# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

# Suppress annoying warnings
warnings.filterwarnings(
    "ignore", message=".*video decoding and encoding capabilities.*deprecated.*"
)
warnings.filterwarnings(
    "ignore", message=".*A new version of Albumentations is available.*"
)

# Add project root to Python path (for loopact and other custom modules)
_project_root = Path(__file__).parent.parent
_project_root_str = str(_project_root.absolute())
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

# Add Isaac-GR00T submodule to Python path (for gr00t.* modules)
_isaac_gr00t_path = _project_root / "third_party" / "Isaac-GR00T"
_isaac_gr00t_str = str(_isaac_gr00t_path.absolute())
if _isaac_gr00t_str not in sys.path:
    sys.path.insert(0, _isaac_gr00t_str)

import torch
import tyro
from transformers import TrainingArguments
from omegaconf import OmegaConf

# Basic imports that don't depend on mode
from gr00t.data.schema import EmbodimentTag
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model
from gr00t.experiment.runner import TrainRunner

# Import shared utilities
from utils.module_fetcher import ModuleFetcher


@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters (required field comes first)
    dataset_path: str = ""
    """Path to the dataset directory"""

    # Config file support
    config: str = ""
    """Path to YAML config file. If provided, settings from YAML will be used as defaults."""

    # Mode selection - mutually exclusive
    baseline: bool = False
    """Use baseline GR00T implementation from submodule (gr00t.*)"""

    feedback: bool = False
    """Use feedback VLA implementation from main repo (loopact.*)"""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: str = "fourier_gr1_arms_only"
    """
    Data configuration to use for training.
    Options: 'so100', 'fourier_gr1_arms_only', 'unitree_g1', 'libero_arm', etc.
    See gr00t/experiment/data_config.py for more details.
    """

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    # Tuning parameters (always stage 1: projector + diffusion only)
    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 12
    """Number of workers for data loading per GPU."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps for training."""

    dataloader_prefetch_factor: int = 4
    """Prefetch factor for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard', 'azure_ml')."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1', 'libero_arm'"""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchvision_av"
    """Video backend to use for training. [torchcodec, decord, torchvision_av]"""


#####################################################################################
# Helper functions
#####################################################################################


def _copy_partial_action_expert_weights(old_dict, new_dict, old_dim, new_dim):
    """
    Copy weights with partial dimension matching for action_dim changes.
    NOTE(Youliang): this is a very experimental implementation to handle action_dim changes. TODO: improve this.
    """
    total_params = copied_params = random_params = 0

    for key, old_tensor in old_dict.items():
        if key not in new_dict:
            continue

        new_tensor = new_dict[key]
        total_params += new_tensor.numel()

        if old_tensor.shape == new_tensor.shape:
            # Same shape: direct copy
            new_tensor.copy_(old_tensor)
            copied_params += old_tensor.numel()
        elif "action_encoder" in key and "W1.weight" in key:
            # Input dimension change: copy [:, :old_dim]
            new_tensor[:, :old_dim] = old_tensor
            copied_params += old_tensor.numel()
            random_params += new_tensor.numel() - old_tensor.numel()
        elif "action_decoder" in key and ("weight" in key or "bias" in key):
            # Output dimension change: copy first old_dim elements of last dimension
            if old_tensor.dim() == 1:
                new_tensor[:old_dim] = old_tensor
            elif old_tensor.dim() == 2:
                new_tensor[:, :old_dim] = old_tensor
            elif old_tensor.dim() == 3:
                new_tensor[:, :, :old_dim] = old_tensor
            copied_params += old_tensor.numel()
            random_params += new_tensor.numel() - old_tensor.numel()
        else:
            # Incompatible shape: keep random initialization
            random_params += new_tensor.numel()

    assert total_params == copied_params + random_params, "Parameter count mismatch"
    random_percentage = (random_params / total_params) * 100 if total_params > 0 else 0
    print(
        f"Weight copy stats: {copied_params:,} copied, {random_params:,} random ({random_percentage:.1f}% randomly initialized)"
    )
    print(f"Action dimensions {old_dim + 1}-{new_dim} will be learned from scratch")
    return new_dict


#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""
    # Determine mode
    if config.baseline and config.feedback:
        raise ValueError("Cannot specify both --baseline and --feedback. Choose one.")
    elif config.baseline:
        mode = "baseline"
    elif config.feedback:
        mode = "feedback"
    else:
        # Default to baseline if neither specified
        mode = "baseline"
        print("Warning: No mode specified. Defaulting to --baseline mode.")

    # Set environment variable for conditional imports in runner.py
    os.environ["GR00T_MODULE_MODE"] = mode

    print(f"\nRunning in {mode.upper()} mode")
    print(
        f"  Using: {'gr00t.* (submodule)' if mode == 'baseline' else 'loopact.* (main repo)'}\n"
    )

    # Initialize module fetcher
    fetcher = ModuleFetcher(mode)

    # Print tuning configuration
    print("Fine-tuning configuration:")
    print(f"  - LLM backbone: {config.tune_llm}")
    print(f"  - Vision tower: {config.tune_visual}")
    print(f"  - Projector: {config.tune_projector}")
    print(f"  - Diffusion model: {config.tune_diffusion_model}")
    print()

    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = fetcher.load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 data loader
    train_dataset = fetcher.LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=config.video_backend,
    )

    # ------------ step 2: load model ------------
    # Get the data config to determine action horizon
    data_action_horizon = len(data_config_cls.action_indices)

    # Get max_action_dim from the last transform
    assert hasattr(transforms, "transforms") and len(transforms.transforms) > 0, (
        "No transforms found"
    )
    last_transform = transforms.transforms[-1]

    assert isinstance(last_transform, fetcher.GR00TTransform), (
        "Last transform must be GR00TTransform"
    )
    assert hasattr(last_transform, "max_action_dim"), (
        "GR00TTransform must have max_action_dim"
    )
    data_max_action_dim = last_transform.max_action_dim

    # Handle tune_stage2 logic
    if mode == "feedback":
        print(
            "Stage 2 training enabled: freezing projector and diffusion model, training feedback_action"
        )
        actual_tune_projector = False
        actual_tune_diffusion_model = False
        tune_feedback = True
    else:
        actual_tune_projector = config.tune_projector
        actual_tune_diffusion_model = config.tune_diffusion_model
        tune_feedback = False

    # Load model
    model = fetcher.Model.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,
        tune_visual=config.tune_visual,
        tune_projector=actual_tune_projector,
        tune_diffusion_model=actual_tune_diffusion_model,
        tune_feedback=tune_feedback,
    )

    # Update action_horizon and max_action_dim to match data config
    # Need to recreate action head with correct config since it was initialized with old config
    action_horizon_mismatch = (
        data_action_horizon != model.action_head.config.action_horizon
    )
    action_dim_mismatch = data_max_action_dim != model.action_head.config.action_dim

    if action_horizon_mismatch or action_dim_mismatch:
        # Store old values for logging
        old_action_horizon = model.action_head.config.action_horizon
        old_action_dim = model.action_head.config.action_dim
        print(
            f"Recreating action head with action_horizon {data_action_horizon} (was {old_action_horizon})"
        )
        if action_dim_mismatch:
            print(
                f"Updating max_action_dim {data_max_action_dim} (was {old_action_dim})"
            )

        # Update the action head config (need to copy to avoid modifying original)
        import copy

        new_action_head_config = copy.deepcopy(model.action_head.config)
        new_action_head_config.action_horizon = data_action_horizon
        new_action_head_config.action_dim = data_max_action_dim

        # Create new action head with updated config
        new_action_head = fetcher.ActionPlanner(new_action_head_config)

        # Copy the weights from the old action head to the new one
        if not action_dim_mismatch:
            print("Copying weights from old action head (compatible dimensions)")
            new_action_head.load_state_dict(
                model.action_head.state_dict(), strict=False
            )
        else:
            print(
                f"Partial weight copy: copying first {old_action_dim} dimensions, initializing last {data_max_action_dim - old_action_dim} dimensions randomly"
            )
            new_action_head.state_dict().update(
                _copy_partial_action_expert_weights(
                    model.action_head.state_dict(),
                    new_action_head.state_dict(),
                    old_action_dim,
                    data_max_action_dim,
                )
            )

        # Replace the action head
        model.action_head = new_action_head

        # Update model config AND the action_head_cfg dictionary that gets saved
        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
        model.config.action_head_cfg["action_horizon"] = data_action_horizon
        model.config.action_head_cfg["action_dim"] = data_max_action_dim

        # Update the main model's action_dim for validation (critical for validate_inputs)
        model.config.action_dim = data_max_action_dim
        model.action_dim = data_max_action_dim

        # Set trainable parameters for the new action head
        # Respect stage-2 overrides when feedback training is enabled.
        model.action_head.set_trainable_parameters(
            tune_projector=actual_tune_projector,
            tune_diffusion_model=actual_tune_diffusion_model,
        )

        # Also recreate feedback_action if in feedback mode (it uses action_dim)
        if mode == "feedback" and action_dim_mismatch:
            print(f"Recreating feedback_action with new action_dim: {data_max_action_dim}")
            FeedbackAction = fetcher.VisualRefinementNetwork
            new_feedback_action = FeedbackAction(new_action_head_config)
            model.feedback_action = new_feedback_action
            model.feedback_action.set_trainable(tune_feedback)

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    # Debug: print trainable parameters to verify freeze behavior
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"TRAINABLE: {name}")
            trainable_params += param.numel()
    print(f"Trainable parameter count: {trainable_params}")

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        # evaluation_strategy="no",
        save_total_limit=5,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.3 run experiment
    experiment.train()


if __name__ == "__main__":
    # Check for --config in command line arguments first
    yaml_config = {}
    if "--config" in sys.argv:
        config_idx = sys.argv.index("--config")
        if config_idx + 1 < len(sys.argv):
            config_file_path = Path(sys.argv[config_idx + 1])
            if not config_file_path.is_absolute():
                config_file_path = Path.cwd() / config_file_path
            if not config_file_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_file_path}")

            # Load YAML config
            yaml_config = OmegaConf.load(str(config_file_path))
            yaml_config = OmegaConf.to_container(yaml_config, resolve=True)
            print(f"📄 Loaded config from: {config_file_path}\n")

            # Remove --config and its value from sys.argv so tyro doesn't see it
            sys.argv.pop(config_idx)  # Remove --config
            sys.argv.pop(config_idx)  # Remove the path

    # Convert YAML config to command-line arguments (will be overridden by actual CLI args)
    if yaml_config:
        yaml_args = []
        for key, value in yaml_config.items():
            # Skip config itself and module_mode (handled separately)
            if key in ["config", "module_mode", "data_config_loader_module"]:
                continue

            arg_key = f"--{key.replace('_', '-')}"

            # Check if this argument is already in sys.argv (CLI takes precedence)
            if arg_key not in sys.argv and f"--{key}" not in sys.argv:
                if isinstance(value, list):
                    yaml_args.append(arg_key)
                    yaml_args.extend([str(v) for v in value])
                elif isinstance(value, bool):
                    if value:  # Only add flag if True
                        yaml_args.append(arg_key)
                else:
                    yaml_args.append(arg_key)
                    yaml_args.append(str(value))

        # Insert YAML args after script name (CLI args will override)
        sys.argv = sys.argv[:1] + yaml_args + sys.argv[1:]

    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)

    # Validate that dataset_path is provided
    if not config.dataset_path:
        raise ValueError(
            "dataset_path is required. Provide it via --dataset-path or in config file."
        )

    # Print the configuration
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        if key != "config":  # Don't print config path twice
            print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert config.num_gpus <= available_gpus, (
        f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    )
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            script_path = Path(__file__).absolute()

            # Use subprocess.run instead of os.system
            raw_args_list = sys.argv[1:]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
                *raw_args_list,
            ]

            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)

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

"""
Script to evaluate a checkpoint with stepped inference (16-step action chunks).
This script loads a checkpoint, runs inference with real observations from a dataset,
performs stepped inference, and plots predicted actions vs ground truth.
"""

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Suppress annoying warnings
warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities.*deprecated.*")
warnings.filterwarnings("ignore", message=".*A new version of Albumentations is available.*")

# Add project root to Python path
_project_root = Path(__file__).parent.parent
_project_root_str = str(_project_root.absolute())
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

# Add Isaac-GR00T submodule to Python path
_isaac_gr00t_path = _project_root / "third_party" / "Isaac-GR00T"
_isaac_gr00t_str = str(_isaac_gr00t_path.absolute())
if _isaac_gr00t_str not in sys.path:
    sys.path.insert(0, _isaac_gr00t_str)

# Import after path setup
from gr00t.data.schema import EmbodimentTag
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING

# Import shared utilities
from utils.module_fetcher import ModuleFetcher


# =============================================================================
# CONFIGURATION — edit these to switch runs
# =============================================================================
MODE = "feedback"            # "baseline" or "feedback"
CHECKPOINT_PATH = "checkpoints/checkpoint-10000"  # Relative to project root
# CHECKPOINT_PATH = "checkpoints/cp_sim_obs"
DATASET_PATH = "experiments/demo_data/pick_up_the_green_toy_data"         # drop_the_ball_data_imp, pick_up_the_green_toy_data
# DATA_CONFIG is auto-selected from MODE: baseline→"libero_arm", feedback→"libero_arm_obs"
DATA_CONFIG = None
TRAJ_ID = 20                  # Which trajectory to evaluate (0-indexed)
MAX_STEPS = None             # None = full trajectory; int = cap at N steps
SAVE_PLOT_PATH = None        # None = show interactively; str path = save to file

# --- Simulation Replay ---
REPLAY_IN_SIM = False         # Set True to replay predicted actions in a popup simulation
REPLAY_ACTION = "predicted"         # gt or predicted
REPLAY_ENV_NAME = "ToyCan"    # Use the exact class name: "ToyCan", "ToyRed", "ToyGreen", "DropBall"
REPLAY_SLOPE_ANGLE = 1.6     # For ToyCan: slope angle in degrees (should match data collection)
REPLAY_STEP_DELAY = 0.3      # Seconds between steps (0.1 = real-time 10Hz, 0.3 = slow motion)
# =============================================================================


@dataclass
class ArgsConfig:
    """Configuration for checkpoint evaluation with stepped inference."""

    # Config file support
    config: str = ""
    """Path to YAML config file. If provided, settings from YAML will be used."""

    # Mode selection - mutually exclusive
    baseline: bool = False
    """Use baseline GR00T implementation from submodule (gr00t.*)"""

    feedback: bool = False
    """Use feedback VLA implementation from main repo (loopact.*)"""

    # Checkpoint and dataset
    checkpoint_path: str = ""
    """Path to the checkpoint directory."""

    dataset_path: str = ""
    """Path to the dataset directory."""

    data_config: str = "libero_arm"
    """Data config to use for loading the model (determines modality config)."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "libero_arm"
    """Embodiment tag to use for the model."""

    # Inference parameters
    action_chunk_size: int = 16
    """Number of action steps to predict at once."""

    denoising_steps: int = 2
    """Number of denoising steps for the diffusion model."""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchvision_av"
    """Video backend to use for loading videos."""

    device: str = "cuda"
    """Device to run inference on."""

    # Output parameters
    save_plot_path: Optional[str] = None
    """Path to save the action plot. If None, will display the plot."""

    traj_id: int = 0
    """Which trajectory to evaluate (0-indexed)."""

    max_steps: Optional[int] = None
    """Maximum number of steps to evaluate. If None, evaluates entire dataset."""

    sim_replay: bool = False
    """If True, replay predicted actions in a popup robosuite simulation after plotting."""


#####################################################################################
# Helper Functions
#####################################################################################


def plot_actions(predicted_actions, ground_truth_actions, action_names=None, save_path=None):
    """
    Plot predicted actions vs ground truth actions.
    
    Args:
        predicted_actions: np.ndarray of shape (num_steps, action_dim)
        ground_truth_actions: np.ndarray of shape (num_steps, action_dim)
        action_names: Optional list of action dimension names
        save_path: Optional path to save the plot
    """
    num_steps, action_dim = predicted_actions.shape
    
    # Create subplots for each action dimension
    fig, axes = plt.subplots(action_dim, 1, figsize=(12, 3 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]
    
    timesteps = np.arange(num_steps)
    
    for i in range(action_dim):
        ax = axes[i]
        ax.plot(timesteps, ground_truth_actions[:, i], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(timesteps, predicted_actions[:, i], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        
        if action_names and i < len(action_names):
            title = f"{action_names[i]}"
        else:
            title = f"Action Dim {i}"
        
        ax.set_title(title)
        ax.set_ylabel('Action Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Timestep')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


#####################################################################################
# Simulation Replay
#####################################################################################


def replay_in_simulation(predicted_actions: np.ndarray, ground_truth_actions: np.ndarray, dataset, traj_start_idx: int):
    """
    Replay predicted actions in a popup robosuite simulation.

    Resets the env with randomize_init=False so the robot starts at exactly
    INIT_QPOS, then overwrites joint positions with the dataset's first-frame
    state.joint_states before stepping through predicted_actions.

    Args:
        predicted_actions: (T, action_dim) array of actions to replay
        dataset: LeRobotSingleDataset instance (provides initial joint state)
        traj_start_idx: start index of the trajectory in the dataset
    """
    import importlib.util
    import robosuite as suite
    from robosuite import load_composite_controller_config

    # Register custom environments if needed
    _robosuite_exp = _project_root / "experiments" / "robosuite"
    if str(_robosuite_exp) not in sys.path:
        sys.path.insert(0, str(_robosuite_exp))

    # 'utils' is already cached as the project-root package (no success_checks).
    # Pre-inject the robosuite utils.success_checks so env/task imports don't fail.
    if "utils.success_checks" not in sys.modules:
        _sc_path = _robosuite_exp / "utils" / "success_checks.py"
        _sc_spec = importlib.util.spec_from_file_location("utils.success_checks", str(_sc_path))
        _sc_mod = importlib.util.module_from_spec(_sc_spec)
        sys.modules["utils.success_checks"] = _sc_mod
        _sc_spec.loader.exec_module(_sc_mod)

    # Register all custom environments by importing each *Env.py file directly.
    # environment/__init__.py is incomplete; each file calls register_env() on import.
    _env_dir = _robosuite_exp / "environment"
    for _env_file in sorted(_env_dir.glob("*Env.py")):
        _mod_name = f"environment.{_env_file.stem}"
        if _mod_name not in sys.modules:
            try:
                _spec = importlib.util.spec_from_file_location(_mod_name, str(_env_file))
                _mod = importlib.util.module_from_spec(_spec)
                sys.modules[_mod_name] = _mod
                _spec.loader.exec_module(_mod)
            except Exception as _e:
                print(f"[REPLAY] Warning: could not load {_env_file.name}: {_e}")

    controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")

    env_kwargs = dict(
        env_name=REPLAY_ENV_NAME,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=10,
        horizon=len(predicted_actions) + 10,
        randomize_init=False,
    )
    if REPLAY_ENV_NAME == "ToyCan":
        env_kwargs["slope_angle"] = REPLAY_SLOPE_ANGLE

    print(f"\n[REPLAY] Creating '{REPLAY_ENV_NAME}' environment...")
    env = suite.make(**env_kwargs)
    env.reset()

    # Set robot joint positions from dataset's first frame
    first_obs = dataset[traj_start_idx]
    if "state.joint_states" in first_obs:
        joint_states = first_obs["state.joint_states"]
        if isinstance(joint_states, torch.Tensor):
            joint_states = joint_states.cpu().numpy()
        joint_states = joint_states.squeeze()

        robot = env.robots[0]
        joint_indices = robot._ref_joint_pos_indexes
        for i, idx in enumerate(joint_indices):
            if i < len(joint_states):
                env.sim.data.qpos[idx] = joint_states[i]
        env.sim.forward()
        print(f"[REPLAY] Initial joints set from dataset: {np.array2string(joint_states, precision=3)}")
    else:
        print("[REPLAY] state.joint_states not found — using env INIT_QPOS")

    # Mirror data collection: open gripper and let controller settle (20 warmup steps).
    # The dataset's first frame is recorded after this warmup, so skipping it causes
    # the gripper to start closed and the arm to sit lower, making it hit the table.
    action_dim = env.action_spec[0].shape[0]
    if action_dim > 6:
        warmup_action = np.zeros(action_dim)
        warmup_action[-1] = -1.0  # open gripper
        for _ in range(20):
            env.step(warmup_action)
    print("[REPLAY] Warmup complete (gripper opened, controller settled)")

    import time
    actions_to_replay = ground_truth_actions if REPLAY_ACTION == "gt" else predicted_actions
    label = "ground truth" if REPLAY_ACTION == "gt" else "predicted"
    print(f"[REPLAY] Replaying {len(actions_to_replay)} steps ({label}) at {REPLAY_STEP_DELAY}s/step. Close the window or Ctrl+C to stop.\n")
    try:
        for action in actions_to_replay:
            env.step(action)
            env.render()
            time.sleep(REPLAY_STEP_DELAY)
    except KeyboardInterrupt:
        print("[REPLAY] Interrupted.")
    finally:
        env.close()
        print("[REPLAY] Done.")


#####################################################################################
# Main Evaluation Function
#####################################################################################


def main(config: ArgsConfig):
    """Main function to evaluate checkpoint with stepped inference."""
    print("=" * 80)
    print("Checkpoint Evaluation with Stepped Inference")
    print("=" * 80)

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
    
    # Set environment variable for conditional imports in Gr00tPolicy._load_model()
    os.environ["GR00T_MODULE_MODE"] = mode
    
    print(f"\nRunning in {mode.upper()} mode")
    print(f"  Using: {'gr00t.* (submodule)' if mode == 'baseline' else 'loopact.* (main repo)'}\n")
    
    # Initialize module fetcher
    fetcher = ModuleFetcher(mode)
    
    # Validate paths
    if not config.checkpoint_path or not os.path.exists(config.checkpoint_path):
        print(f"ERROR: Checkpoint path does not exist: {config.checkpoint_path}")
        sys.exit(1)

    if not config.dataset_path or not os.path.exists(config.dataset_path):
        print(f"ERROR: Dataset path does not exist: {config.dataset_path}")
        sys.exit(1)

    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Data config: {config.data_config}")
    print(f"Embodiment tag: {config.embodiment_tag}")
    print(f"Action chunk size: {config.action_chunk_size}")
    print()

    # Load data config
    try:
        data_config_cls = fetcher.load_data_config(config.data_config)
        modality_config = data_config_cls.modality_config()
        modality_transform = data_config_cls.transform()
    except Exception as e:
        print(f"ERROR: Failed to load data config '{config.data_config}': {e}")
        sys.exit(1)

    # Load policy
    print("Loading policy...")
    try:
        Gr00tPolicy = fetcher.Gr00tPolicy
        policy = Gr00tPolicy(
            model_path=config.checkpoint_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=config.embodiment_tag,
            denoising_steps=config.denoising_steps,
            device=config.device,
        )
        # Set model to eval mode
        policy.model.eval()
        print("✓ Policy loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load policy: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load dataset
    print("\nLoading dataset...")
    try:
        LeRobotSingleDataset = fetcher.LeRobotSingleDataset
        dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path,
            modality_configs=modality_config,
            video_backend=config.video_backend,
            video_backend_kwargs=None,
            transforms=None,
            embodiment_tag=config.embodiment_tag,
        )
        if len(dataset) == 0:
            print(f"ERROR: Dataset is empty")
            sys.exit(1)
        print(f"✓ Dataset loaded: {len(dataset)} total frames, {len(dataset.trajectory_lengths)} trajectories")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Get the selected trajectory for evaluation
    traj_id = config.traj_id
    if traj_id >= len(dataset.trajectory_lengths):
        print(f"ERROR: traj_id {traj_id} out of range (dataset has {len(dataset.trajectory_lengths)} trajectories)")
        sys.exit(1)
    traj_start_idx = sum(dataset.trajectory_lengths[:traj_id])
    traj_length = dataset.trajectory_lengths[traj_id]
    
    # Determine evaluation length
    if config.max_steps is not None:
        eval_length = min(config.max_steps, traj_length)
    else:
        eval_length = traj_length
    
    print(f"\nEvaluating trajectory {traj_id} (length: {traj_length}, evaluating: {eval_length} steps)")
    
    # Collect predictions and ground truth
    predicted_actions_list = []
    ground_truth_actions_list = []
    
    print("\n" + "=" * 80)
    print("Running Stepped Inference...")
    print("=" * 80)

    def _get_gt_action(step_idx):
        """Extract scalar ground truth action at a single dataset step."""
        gt_obs = dataset[traj_start_idx + step_idx]
        for key in gt_obs.keys():
            if 'action' in key.lower() and not key.startswith('video.') and not key.startswith('state.'):
                gt = gt_obs[key]
                if isinstance(gt, torch.Tensor):
                    gt = gt.cpu().numpy()
                if gt.ndim == 3:
                    return gt[0, 0]
                elif gt.ndim == 2:
                    return gt[0]
                return gt
        return None

    with torch.no_grad():
        if mode == "feedback":
            # Feedback: call get_action every step so the feedback decoder sees
            # the current observation at each timestep.
            for step in range(eval_length):
                obs = dataset[traj_start_idx + step]
                action_output = policy.get_action(obs, step)

                # Extract single action (feedback path returns 1 action per step)
                if isinstance(action_output, dict):
                    for key in action_output:
                        if 'action' in key.lower():
                            action_output = action_output[key]
                            break
                if isinstance(action_output, torch.Tensor):
                    action_output = action_output.cpu().numpy()
                action_output = np.squeeze(action_output)
                if action_output.ndim == 2:
                    action_output = action_output[0]

                gt = _get_gt_action(step)
                if gt is None:
                    print(f"Warning: no GT action at step {step}")
                    continue

                predicted_actions_list.append(action_output)
                ground_truth_actions_list.append(gt)

                if step % config.action_chunk_size == 0:
                    print(f"Step {step}: action shape {action_output.shape}")

        else:
            # Baseline: call get_action once per chunk, use all 16 predicted actions.
            current_step = 0
            while current_step < eval_length:
                obs = dataset[traj_start_idx + current_step]
                action_output = policy.get_action(obs)

                if isinstance(action_output, dict):
                    for key in action_output:
                        if 'action' in key.lower():
                            action_output = action_output[key]
                            break
                if isinstance(action_output, torch.Tensor):
                    action_output = action_output.cpu().numpy()
                if action_output.ndim == 3:
                    action_output = action_output[0]

                chunk_size = min(config.action_chunk_size, eval_length - current_step)
                gt_chunk = []
                for i in range(chunk_size):
                    gt = _get_gt_action(current_step + i)
                    if gt is not None:
                        gt_chunk.append(gt)

                if not gt_chunk:
                    print(f"Warning: no GT actions at step {current_step}")
                    break

                gt_chunk = np.array(gt_chunk)
                actual = min(len(action_output), len(gt_chunk))
                predicted_actions_list.append(action_output[:actual])
                ground_truth_actions_list.append(gt_chunk[:actual])
                print(f"Step {current_step}: predicted {action_output.shape}, GT {gt_chunk.shape}, using {actual}")
                current_step += actual
    
    # Concatenate all predictions and ground truth
    predicted_actions = np.stack(predicted_actions_list, axis=0) if mode == "feedback" else np.concatenate(predicted_actions_list, axis=0)
    ground_truth_actions = np.stack(ground_truth_actions_list, axis=0) if mode == "feedback" else np.concatenate(ground_truth_actions_list, axis=0)
    
    print(f"\n✓ Inference complete!")
    print(f"  Total steps evaluated: {predicted_actions.shape[0]}")
    print(f"  Action dimension: {predicted_actions.shape[1]}")
    

    action_names = None
    if hasattr(data_config_cls, 'action_names'):
        action_names = data_config_cls.action_names
    
    plot_actions(
        predicted_actions,
        ground_truth_actions,
        action_names=action_names,
        save_path=config.save_plot_path
    )

    if config.sim_replay:
        replay_in_simulation(predicted_actions, ground_truth_actions, dataset, traj_start_idx)


if __name__ == "__main__":
    checkpoint_abs = str(_project_root / CHECKPOINT_PATH)
    dataset_abs = str(_project_root / DATASET_PATH)

    # Auto-select data config from mode unless overridden
    _data_config = DATA_CONFIG or ("libero_arm_obs" if MODE == "feedback" else "libero_arm")

    config = ArgsConfig(
        baseline=(MODE == "baseline"),
        feedback=(MODE == "feedback"),
        checkpoint_path=checkpoint_abs,
        dataset_path=dataset_abs,
        data_config=_data_config,
        traj_id=TRAJ_ID,
        max_steps=MAX_STEPS,
        save_plot_path=SAVE_PLOT_PATH,
        sim_replay=REPLAY_IN_SIM,
    )

    print("\n" + "=" * 50)
    print("EVALUATION CONFIGURATION:")
    print("=" * 50)
    print(f"mode:            {MODE}")
    print(f"checkpoint:      {CHECKPOINT_PATH}")
    print(f"dataset:         {DATASET_PATH}")
    print(f"data_config:     {_data_config}")
    print(f"traj_id:         {TRAJ_ID}")
    print(f"max_steps:       {MAX_STEPS}")
    print(f"save_plot_path:  {SAVE_PLOT_PATH}")
    print(f"sim_replay:      {REPLAY_IN_SIM}")
    print("=" * 50 + "\n")

    main(config)

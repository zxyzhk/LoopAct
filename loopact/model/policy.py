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

from dataclasses import dataclass, field
from typing import Tuple
import copy
import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_proposal_planner import (
    ActionProposalPlanner,
    ActionProposalPlannerConfig,
)
from .visual_refinement_network import VisualRefinementNetwork
from gr00t.model.backbone import EagleBackbone

import os
import time


BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class LoopActConfig(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(
        init=False, metadata={"help": "Action head configuration."}
    )

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class LoopAct(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = LoopActConfig
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: LoopActConfig,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone(**config.backbone_cfg)
        action_head_cfg = ActionProposalPlannerConfig(**config.action_head_cfg)
        self.action_head = ActionProposalPlanner(action_head_cfg)
        # Feedback module (no shared parameters with action_head)
        self.feedback_action = VisualRefinementNetwork(action_head_cfg)
        # By default, freeze feedback_action (Stage 1 training) - will be set in from_pretrained
        self.visual_refinement_network.set_trainable(False)

        # Track which training stage we're in
        self.tune_feedback = False  # Will be set by from_pretrained

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def train(self, mode=True):
        """
        Override train() to ensure frozen modules stay in eval mode during stage 2 training.
        When tune_feedback=True (stage 2), everything except feedback_action (observer + updater)
        should be in eval mode, even when HuggingFace trainer calls model.train().
        """
        super().train(mode)
        if mode and self.tune_feedback:
            # Stage 2 training: set everything to eval mode except feedback_action
            self.backbone.eval()
            self.action_proposal_planner.eval()
            # Keep feedback_action (observer + updater) in train mode
            # Note: feedback_action.train() is already called by super().train()
            # but we call set_frozen_modules_to_eval_mode() to ensure obs_encoder stays frozen
            self.visual_refinement_network.set_frozen_modules_to_eval_mode()
        return self

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        # print(f"action_horizon: {self.action_horizon}")
        # print(f"action_dim: {self.action_dim}")

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\naction.dtype={action.dtype}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\naction.shape={action.shape}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\ntype(video)={type(video)}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\nvideo.dtype={video.dtype}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\nvideo.shape={video.shape}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\nisinstance(backbone_outputs, BatchFeature)={isinstance(backbone_outputs, BatchFeature)}"
            error_msg += f"\nBACKBONE_FEATURE_KEY in backbone_outputs={BACKBONE_FEATURE_KEY in backbone_outputs}"
            error_msg += f"\nbackbone_outputs[BACKBONE_FEATURE_KEY].shape={backbone_outputs[BACKBONE_FEATURE_KEY].shape}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\nisinstance(action_head_outputs, BatchFeature)={isinstance(action_head_outputs, BatchFeature)}"
            error_msg += (
                f"\nLOSS_KEY in action_head_outputs={LOSS_KEY in action_head_outputs}"
            )
            error_msg += f"\naction_head_outputs[ACTION_KEY].shape={action_head_outputs[ACTION_KEY].shape}"
            error_msg += f"\nself.action_horizon={self.action_horizon}"
            error_msg += f"\nself.action_dim={self.action_dim}"
            raise ValueError(error_msg)

    # TODO: when reuse vlm values, it's passed into backward progress again. Try to Detach

    def forward(
        self,
        inputs: dict,
        window_idx: int = None,
    ) -> BatchFeature:
        if self.tune_feedback:
            return self.forward_stage2(inputs, window_idx)
        else:
            return self.forward_stage1(inputs, window_idx)

    def forward_stage1(
        self,
        inputs: dict,
        window_idx: int = None,
    ) -> BatchFeature:
        """Stage 1: Your original forward logic"""
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_proposal_planner(
            backbone_outputs, action_inputs
        )  # raw 16 action chunk
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        print("Forward Stage 1")
        return action_head_outputs  # loss

    def forward_stage2(
        self,
        inputs: dict,
        window_idx: int = None,
    ) -> BatchFeature:
        """
        Stage 2: Train feedback_action with frozen action_head.
        Gets predicted actions from DiT (with gradients) and updates them with feedback_action.
        """
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Get backbone outputs (frozen)
        with torch.no_grad():
            backbone_outputs = self.backbone(backbone_inputs)

        # Get predicted actions from action_head with gradients enabled
        # Note: encoder/decoder/DiT should already be frozen via requires_grad=False
        action_head_outputs = self.action_proposal_planner.get_predicted_actions(
            backbone_outputs, action_inputs
        )  # [B, 16, action_dim] - pure actions from DiT

        # Apply feedback updater (trainable)
        feedback_action_outputs = self.visual_refinement_network(
            action_head_outputs, window_idx, action_inputs
        )  # [B, 16, action_dim] - updated actions

        self.validate_data(feedback_action_outputs, backbone_outputs, is_training=True)
        print("Forward Stage 2")
        return feedback_action_outputs  # loss

    def _detach_batchfeature(self, bf):
        """
        Return a copy of bf where *every* tensor in bf.data
        has been cloned, detached, and set requires_grad=False.
        We do this by shallow-copying bf and then replacing its .data.
        """
        # 1) Build a new dict of detached tensors
        detached_data = {}
        for k, v in bf.data.items():
            if torch.is_tensor(v):
                # clone so we don’t share storage, detach from any graph,
                # and prevent any future grad requests
                detached_data[k] = v.clone().detach().requires_grad_(False)
            else:
                detached_data[k] = v

        # 2) Shallow-copy the original BatchFeature object
        new_bf = copy.copy(bf)
        # 3) Overwrite its .data with our detached copy
        new_bf.data = detached_data
        return new_bf

    # def get_action(
    #     # a clean version
    #     self,
    #     inputs: dict,
    #     time_step: int,
    # ) -> BatchFeature:
    #     backbone_inputs, action_inputs = self.prepare_input(inputs)
    #     if time_step % 4 == 0:
    #         print(f"im at {time_step}, Run Model")
    #         print("320 320 320")

    #         backbone_outputs = self.backbone(backbone_inputs)
    #         action_head_outputs = self.action_proposal_planner.get_action(
    #             backbone_outputs, action_inputs
    #         )
    #         self._action_context_cache = action_head_outputs
    #         feedback_action_outputs = self.visual_refinement_network.get_action(
    #             action_head_outputs, time_step, action_inputs
    #         )
    #         return feedback_action_outputs
    #     else:
    #         print(f"im at {time_step}, CACHED VLM")
    #         # Reuse cached action_head outputs, only run feedback with fresh observations
    #         if not hasattr(self, "_action_context_cache"):
    #             raise ValueError(
    #                 f"No cached action head outputs available at timestep {time_step}"
    #             )
    #         # Skip action_head.get_action() - just use cached outputs directly with feedback
    #         feedback_action_outputs = self.visual_refinement_network.get_action(
    #             self._action_context_cache, time_step, action_inputs
    #         )
    #         return feedback_action_outputs

    def get_action(
        self,
        inputs: dict,
        time_step: int,
    ) -> BatchFeature:
        timing_enabled = os.getenv("GR00T_TIMING", "0") == "1"

        def maybe_sync():
            if timing_enabled and torch.cuda.is_available():
                torch.cuda.synchronize()

        if timing_enabled:
            maybe_sync()
            start_perf = time.perf_counter()

        if time_step % 16 == 0:
            # print(f"im at {time_step}, Run Model")
            # print("Full VLM + DiT inference")

            backbone_inputs, action_inputs = self.prepare_input(inputs)
            if timing_enabled:
                maybe_sync()
                backbone_start = time.perf_counter()
            backbone_outputs = self.backbone(backbone_inputs)
            if timing_enabled:
                maybe_sync()
                backbone_end = time.perf_counter()

            if timing_enabled:
                maybe_sync()
                action_head_start = time.perf_counter()
            action_head_outputs = self.action_proposal_planner.get_action(
                backbone_outputs, action_inputs
            )
            if timing_enabled:
                maybe_sync()
                action_head_end = time.perf_counter()

            if timing_enabled:
                maybe_sync()
                feedback_action_start = time.perf_counter()
            self._action_context_cache = action_head_outputs
            feedback_action_outputs = self.visual_refinement_network.get_action(
                action_head_outputs, time_step, action_inputs
            )
            if timing_enabled:
                maybe_sync()
                feedback_action_end = time.perf_counter()

            if timing_enabled:
                total_ms = (feedback_action_end - start_perf) * 1000.0
                backbone_ms = (backbone_end - backbone_start) * 1000.0
                action_head_ms = (action_head_end - action_head_start) * 1000.0
                feedback_action_ms = (
                    feedback_action_end - feedback_action_start
                ) * 1000.0
                print("=" * 90)
                print(
                    f"backbone_ms={backbone_ms:.2f} "
                    f"action_head_ms={action_head_ms:.2f} "
                    f"feedback_action_ms={feedback_action_ms:.2f} "
                    f"total_ms={total_ms:.2f}"
                )
                print("=" * 90)

            return feedback_action_outputs

        else:
            # print(f"im at {time_step}, CACHED VLM (feedback only)")
            # Reuse cached action_head outputs, only run feedback with fresh observations
            if not hasattr(self, "_action_context_cache"):
                raise ValueError(
                    f"No cached action head outputs available at timestep {time_step}"
                )
            action_inputs = self.prepare_action_input_only(inputs)
            if timing_enabled:
                maybe_sync()
                cached_process_start = time.perf_counter()
            action_head_outputs = self._action_context_cache
            if timing_enabled:
                maybe_sync()
                cached_process_end = time.perf_counter()

            if timing_enabled:
                maybe_sync()
                feedback_action_start = time.perf_counter()
            feedback_action_outputs = self.visual_refinement_network.get_action(
                action_head_outputs, time_step, action_inputs
            )
            if timing_enabled:
                maybe_sync()
                feedback_action_end = time.perf_counter()

            if timing_enabled:
                total_ms = (feedback_action_end - start_perf) * 1000.0
                cached_process_ms = (cached_process_end - cached_process_start) * 1000.0
                feedback_action_ms = (
                    feedback_action_end - feedback_action_start
                ) * 1000.0
                print("=" * 90)
                print(
                    f"cached_process_ms={cached_process_ms:.2f} "
                    f"feedback_action_ms={feedback_action_ms:.2f} "
                    f"total_ms={total_ms:.2f}"
                )
                print("=" * 90)

            return feedback_action_outputs

    def _filter_action_inputs(self, inputs: dict) -> dict:
        keep_keys = {
            "simple_img",
            "state",
            "action",
            "action_mask",
            "state_mask",
            "embodiment_id",
        }
        return {k: v for k, v in inputs.items() if k in keep_keys}

    def _to_device_with_maybe_dtype(self, x):
        # Only cast to self.compute_dtype if the tensor is floating
        if not torch.is_tensor(x):
            return x
        if torch.is_floating_point(x):
            target_dtype = self.action_proposal_planner.dtype
            if x.device == self.device and x.dtype == target_dtype:
                return x
            return x.to(self.device, dtype=target_dtype)
        # Keep original dtype
        if x.device == self.device:
            return x
        return x.to(self.device)

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_proposal_planner.prepare_input(inputs)

        backbone_inputs = tree.map_structure(
            self._to_device_with_maybe_dtype, backbone_inputs
        )
        action_inputs = tree.map_structure(
            self._to_device_with_maybe_dtype, action_inputs
        )
        return backbone_inputs, action_inputs

    def prepare_action_input_only(self, inputs) -> BatchFeature:
        filtered_inputs = self._filter_action_inputs(inputs)
        self.validate_inputs(filtered_inputs)
        action_inputs = self.action_proposal_planner.prepare_input(filtered_inputs)
        action_inputs = tree.map_structure(
            self._to_device_with_maybe_dtype, action_inputs
        )
        return action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)
        tune_feedback = kwargs.pop("tune_feedback", False)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")
        print(f"Tune feedback action: {tune_feedback}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(
                pretrained_model_name_or_path, repo_type="model"
            )
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_proposal_planner.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        pretrained_model.visual_refinement_network.set_trainable(tune_feedback)

        # Set the training stage flag to control forward routing
        pretrained_model.tune_feedback = tune_feedback

        return pretrained_model

    @property
    def action_proposal_planner(self):
        """LoopAct API; reuse the existing checkpoint storage key."""
        return self.action_head

    @property
    def visual_refinement_network(self):
        """LoopAct API; reuse the existing checkpoint storage key."""
        return self.feedback_action

    @property
    def action_context_cache(self):
        """LoopAct API; reuse the existing checkpoint storage key."""
        return self._action_context_cache


# register
AutoConfig.register("gr00t_n1_5", LoopActConfig)
AutoModel.register(LoopActConfig, LoopAct)


# Serialization-compatible class aliases.
GR00T_N1_5_Config = LoopActConfig
GR00T_N1_5 = LoopAct

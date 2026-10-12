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
Module fetcher utility for dynamically loading baseline or feedback implementations.
"""

import importlib
from typing import Literal


class ModuleFetcher:
    """Fetch modules from either baseline (submodule) or feedback (main repo) implementations."""
    
    def __init__(self, mode: Literal["baseline", "feedback"]):
        """
        Initialize the module fetcher.
        
        Args:
            mode: Either "baseline" (use gr00t.* submodule) or "feedback" (use loopact.*)
        """
        self.mode = mode
        self._modules = {}
        self._load_modules()
    
    def _load_modules(self):
        """Load all required modules based on the mode."""
        if self.mode == "baseline":
            # Import from gr00t submodule
            self._modules["dataset"] = importlib.import_module("gr00t.data.dataset")
            self._modules["model"] = importlib.import_module("gr00t.model.gr00t_n1")
            self._modules["policy"] = importlib.import_module("gr00t.model.policy")
            self._modules["action_head"] = importlib.import_module("gr00t.model.action_head.flow_matching_action_head")
            self._modules["transforms"] = importlib.import_module("gr00t.model.transforms")
            self._modules["data_config"] = importlib.import_module("gr00t.experiment.data_config")
        else:  # feedback mode
            # Import from loopact main repo
            self._modules["dataset"] = importlib.import_module("loopact.data.dataset")
            self._modules["model"] = importlib.import_module("loopact.model.policy")
            self._modules["policy"] = importlib.import_module("gr00t.model.policy")  # Use baseline policy
            self._modules["action_head"] = importlib.import_module("loopact.model.action_proposal_planner")
            self._modules["transforms"] = importlib.import_module("loopact.data.transforms")
            self._modules["data_config"] = importlib.import_module("loopact.data.data_config")
    
    @property
    def Model(self):
        """GR00T baseline or LoopAct, depending on the selected mode."""
        if self.mode == "baseline":
            return self._modules["model"].GR00T_N1_5
        return self._modules["model"].LoopAct

    @property
    def ActionPlanner(self):
        """Baseline action head or LoopAct Action Proposal Planner (APP)."""
        if self.mode == "baseline":
            return self._modules["action_head"].FlowmatchingActionHead
        return self._modules["action_head"].ActionProposalPlanner

    @property
    def VisualRefinementNetwork(self):
        """Visual Refinement Network (VRN), available in feedback mode."""
        if self.mode == "feedback":
            return importlib.import_module("loopact.model.visual_refinement_network").VisualRefinementNetwork
        return None

    @property
    def LeRobotSingleDataset(self):
        """Get the LeRobotSingleDataset class."""
        return self._modules["dataset"].LeRobotSingleDataset
    
    @property
    def GR00T_N1_5(self):
        """Get the GR00T model class."""
        return self.Model
    
    @property
    def Gr00tPolicy(self):
        """Get the Gr00tPolicy class."""
        return self._modules["policy"].Gr00tPolicy
    
    @property
    def FlowmatchingActionHead(self):
        """Get the FlowmatchingActionHead class."""
        return self.ActionPlanner
    
    @property
    def GR00TTransform(self):
        """Get the GR00TTransform class."""
        return self._modules["transforms"].GR00TTransform

    @property
    def FeedbackAction(self):
        """Get the FeedbackAction class (feedback mode only)."""
        if self.mode == "feedback":
            return self.VisualRefinementNetwork
        return None
    
    @property
    def DATA_CONFIG_MAP(self):
        """Get the DATA_CONFIG_MAP (baseline only)."""
        return self._modules["data_config"].DATA_CONFIG_MAP if self.mode == "baseline" else None
    
    def load_data_config(self, config_name: str):
        """
        Load data config by name.
        
        Args:
            config_name: Name of the data configuration to load
            
        Returns:
            Data configuration class instance
        """
        if self.mode == "baseline":
            return self._modules["data_config"].DATA_CONFIG_MAP[config_name]
        else:
            return self._modules["data_config"].load_data_config(config_name)

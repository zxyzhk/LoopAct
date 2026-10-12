import sys
import os
import torch
import numpy as np
import cv2
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation
from pathlib import Path

# Add project root to Python path
_project_root = Path(__file__).parent.parent.parent
_project_root_str = str(_project_root.absolute())
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

# Add Isaac-GR00T submodule to Python path
_isaac_gr00t_path = _project_root / "third_party" / "Isaac-GR00T"
_isaac_gr00t_str = str(_isaac_gr00t_path.absolute())
if _isaac_gr00t_str not in sys.path:
    sys.path.insert(0, _isaac_gr00t_str)

# Import shared utilities
from utils.module_fetcher import ModuleFetcher


class Gr00tLiberoModel:
    def __init__(self, 
                 model_path: str = "change if needed",
                 embodiment_tag="libero_arm",
                 device=None,
                 denoising_steps=4,
                 mode="baseline"):
        """
        Initialize GR00T model for Libero
        
        Args:
            model_path: Path to GR00T model
            embodiment_tag: Tag for the robot embodiment
            device: Device to run model on (default: cuda if available, else cpu)
            mode: 'baseline' or 'feedback' to select model type
        """
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.embodiment_tag = embodiment_tag
        self.model_path = model_path
        self.denoising_steps = denoising_steps
        self.mode = mode
        
        # Set environment variable for conditional imports in Gr00tPolicy._load_model()
        os.environ["GR00T_MODULE_MODE"] = mode
        
        # Initialize module fetcher
        fetcher = ModuleFetcher(mode)
        
        # Load data configuration - use different configs based on mode
        if mode == "feedback":
            data_config_name = "libero_arm_obs"  # Feedback mode needs obs data
        else:
            data_config_name = "libero_arm"  # Baseline mode
        
        data_config_cls = fetcher.load_data_config(data_config_name)
        self.modality_config = data_config_cls.modality_config()
        self.modality_transform = data_config_cls.transform()
        
        # For feedback mode, store last frame for obs.eye_in_hand_rgb
        # (decoder uses obs[:, -1, -1] — only the latest frame matters at inference)
        if mode == "feedback":
            self.obs_frame_buffer = []
            self.max_obs_frames = 1
        
        # Load policy
        Gr00tPolicy = fetcher.Gr00tPolicy
        self.policy = Gr00tPolicy(
            model_path=self.model_path,
            embodiment_tag=self.embodiment_tag,
            modality_config=self.modality_config,
            modality_transform=self.modality_transform,
            device=self.device,
            denoising_steps=self.denoising_steps
        )
        
        # Set target resolution for video data
        self.target_resolution = (256, 256)
        
        print("GR00T model initialized with modality config:")
        for key, value in self.modality_config.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: shape {value.shape}")
            else:
                print(f"{key}: {value}")
    
    def transform_obs(self, step_data, task_description):
        """
        Transform Libero observation to GR00T format
        
        Args:
            step_data: Dictionary containing raw Libero observation data
            task_description: Task description string
            
        Returns:
            Dictionary in GR00T format
        """
        # Process video data
        agentview = np.array(step_data['agentview_image'], dtype=np.uint8)
        if len(agentview.shape) == 3:
            agentview = np.expand_dims(agentview, axis=0)
        
        eye_in_hand = np.array(step_data['robot0_eye_in_hand_image'], dtype=np.uint8)
        if len(eye_in_hand.shape) == 3:
            eye_in_hand = np.expand_dims(eye_in_hand, axis=0)
        
        # For feedback mode, maintain buffer of last 4 frames for obs.eye_in_hand_rgb
        if self.mode == "feedback":
            # Add current frame to buffer
            self.obs_frame_buffer.append(eye_in_hand[0])  # Remove time dimension: (H, W, C)
            # Keep only last 4 frames
            if len(self.obs_frame_buffer) > self.max_obs_frames:
                self.obs_frame_buffer.pop(0)
            # Pad with first frame if we don't have 4 frames yet
            while len(self.obs_frame_buffer) < self.max_obs_frames:
                self.obs_frame_buffer.insert(0, self.obs_frame_buffer[0])
            # Stack to create (4, H, W, C) array
            obs_eye_in_hand = np.stack(self.obs_frame_buffer, axis=0)  # (4, H, W, C)
        
        # Process state data
        joint_states = np.array(step_data['robot0_joint_pos'], dtype=np.float64)
        if len(joint_states.shape) == 1:
            joint_states = np.expand_dims(joint_states, axis=0)
            
        gripper_states = np.array(step_data['robot0_gripper_qpos'], dtype=np.float64)
        if len(gripper_states.shape) == 1:
            gripper_states = np.expand_dims(gripper_states, axis=0)
            
        # Process end-effector data
        ee_pos = np.array(step_data['robot0_eef_pos'], dtype=np.float64)
        ee_ori = T.quat2axisangle(step_data['robot0_eef_quat'])
        
        # Add batch dimensions
        if len(ee_pos.shape) == 1:
            ee_pos = np.expand_dims(ee_pos, axis=0)
        if len(ee_ori.shape) == 1:
            ee_ori = np.expand_dims(ee_ori, axis=0)
            
        ee_states = np.concatenate([ee_pos, ee_ori], axis=1)
        
        # Create GR00T format data
        gr00t_data = {
            'video.agentview_rgb': agentview,  # (1, 128, 128, 3)
            'video.eye_in_hand_rgb': eye_in_hand,  # (1, 128, 128, 3)
            'state.ee_pos': ee_pos,  # (1, 3)
            'state.ee_ori': ee_ori,  # (1, 3)
            'state.ee_states': ee_states,  # (1, 6)
            'state.gripper_states': gripper_states,  # (1, 2)
            'state.joint_states': joint_states,  # (1, 7)
            'annotation.human.action.task_description': [task_description],  # List with task description
        }
        
        # For feedback mode, add obs.eye_in_hand_rgb with 4 frames
        if self.mode == "feedback":
            gr00t_data['obs.eye_in_hand_rgb'] = obs_eye_in_hand  # (4, H, W, C)
        # print(f"task description: {gr00t_data['annotation.human.action.task_description']}")
        
        
        # gr00t_data['video.agentview_rgb'] = np.ones(gr00t_data['video.agentview_rgb'].shape, dtype=np.uint8) * 128
        # gr00t_data['video.eye_in_hand_rgb'] = np.ones(gr00t_data['video.eye_in_hand_rgb'].shape, dtype=np.uint8) * 128
        # gr00t_data['obs.agentview_rgb'] = np.ones(gr00t_data['obs.agentview_rgb'].shape, dtype=np.uint8) * 128
        # gr00t_data['obs.eye_in_hand_rgb'] = np.ones(gr00t_data['obs.eye_in_hand_rgb'].shape, dtype=np.uint8) * 128

        # gr00t_data['state.ee_pos'] = np.random.randn(*gr00t_data['state.ee_pos'].shape).astype(gr00t_data['state.ee_pos'].dtype)
        # gr00t_data['state.ee_ori'] = np.random.randn(*gr00t_data['state.ee_ori'].shape).astype(gr00t_data['state.ee_ori'].dtype)

        # gr00t_data['annotation.human.action.task_description'] = ["do nothing"]

        return gr00t_data

    def get_action(self, step_data, inference_idx, task_description):
        """
        Get action from GR00T model based on observation
        
        Args:
            step_data: Dictionary containing Libero observation data
            inference_idx: Current inference index
            task_description: Task description string
            
        Returns:
            Dictionary containing predicted actions
        """
        gr00t_data = self.transform_obs(step_data, task_description)
        if self.mode == "feedback":
            predicted_action = self.policy.get_action(gr00t_data, inference_idx)
        else:
            predicted_action = self.policy.get_action(gr00t_data)
        action = predicted_action["action.all_actions"]
        return action
    
    def reset(self):
        """Reset the observation frame buffer for a new episode."""
        if self.mode == "feedback":
            self.obs_frame_buffer = []

if __name__ == "__main__":
    # Initialize model
    model = Gr00tLiberoModel()
    
    # Example observation data
    example_obs = {
        'agentview_image': np.zeros((128, 128, 3), dtype=np.uint8),
        'robot0_eye_in_hand_image': np.zeros((128, 128, 3), dtype=np.uint8),
        'robot0_joint_pos': np.zeros(7, dtype=np.float32),
        'robot0_gripper_qpos': np.zeros(2, dtype=np.float32),
        'robot0_eef_pos': np.random.rand(3).astype(np.float32),
        'robot0_eef_quat': np.random.rand(4).astype(np.float32)
    }
    
    # Get action
    time_step = 0
    action = model.get_action(example_obs, time_step, "do your thing!")
    print("Predicted next action:", action)

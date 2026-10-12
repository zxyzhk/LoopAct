"""
Pick Ladle Task Definition

Task: Pick up the ladle from the table and lift it.
Environment: TableLadle
Difficulty: Easy
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# Add parent directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_robosuite_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _robosuite_dir)

from tasks.base import TaskConfig, TaskSuccess
from utils.success_checks import check_object_grasped, check_object_lifted, check_object_lifted_relative


@dataclass
class PickLadleConfig(TaskConfig):
    """Configuration for the Pick Ladle task."""
    
    # Task identification
    task_name: str = "pick_up_ladle"
    task_description: str = "Pick up the ladle from the table"
    task_id: int = 0
    
    # Environment settings
    env_name: str = "TableLadle"
    robot: str = "Panda"
    controller: str = "BASIC"
    control_freq: int = 10  # Match LIBERO dataset (10 Hz)
    
    # Teleoperation settings
    teleoperate_method: str = "spacemouse"
    
    # Rendering and cameras
    render_camera: str = "custom_view"
    camera_names: List[str] = field(default_factory=lambda: ["robot0_eye_in_hand", "agentview"])
    camera_height: int = 128
    camera_width: int = 128
    show_camera_obs: bool = True
    free_view_mode: bool = False
    
    # Task-specific dynamics (ladle moves during task)
    target_joint_name: str = "none"
    target_speed_y: float = -0.020
    target_axis: str = "y"
    
    # Training settings
    randomize_init: bool = True  # Enable LIBERO-style randomization
    save_data: bool = True
    max_episode_steps: int = 800
    
    # Difficulty and tags
    difficulty: str = "easy"
    tags: List[str] = field(default_factory=lambda: ["pick", "single_object", "ladle"])


class PickLadleSuccess(TaskSuccess):
    """Success criteria for the Pick Ladle task."""
    
    def __init__(self):
        super().__init__(
            task_name="pick_up_ladle",
            task_description="Pick up the ladle from the table"
        )
        self.debug_verbose = False
        
        # Define success conditions - check if ladle moved up by at least 5cm
        self.add_condition(
            name="Ladle is lifted",
            check_fn=lambda env, _self=self: check_object_lifted_relative(
                env, "ladle_main", min_lift_height=0.15, verbose=_self.debug_verbose
            ),
            weight=1.0,
            required=True,
        )
        
        self.success_threshold = 1.0  
    
    def check_success(self, env, verbose: bool = False) -> Tuple[bool, float, Dict]:
        """
        Check if the ladle has been successfully picked up.
        
        Args:
            env: The robosuite environment
            verbose: If True, print detailed success information
            
        Returns:
            (success, score, details): Success status, score (0-1), and detailed results
        """
        return self.evaluate_conditions(env, self.success_threshold, verbose)


# Create default instances for easy import
PICK_LADLE_CONFIG = PickLadleConfig()
PICK_LADLE_SUCCESS = PickLadleSuccess()

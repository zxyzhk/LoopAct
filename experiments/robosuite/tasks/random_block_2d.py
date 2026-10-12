"""
Random Block Task Definition

Task: Pick up a block that moves randomly (stops/goes every second).
Environment: GreenBlock
Difficulty: Hard (unpredictable movement requires visual feedback)

This task is designed to test the feedback model's ability to adapt
to unpredictable object movement using visual observations.
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
from utils.success_checks import check_object_lifted_relative


@dataclass
class RandomBlock2DConfig(TaskConfig):
    """Configuration for the Random Block task."""

    # Task identification
    task_name: str = "pick_random_block_2d"
    task_description: str = "Pick up the randomly moving block"
    task_id: int = 1

    # Environment settings
    env_name: str = "GreenBlock"
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

    # Task-specific dynamics (block moves randomly in 2D)
    target_joint_name: str = "block"
    target_axis: str = "y"  # Ignored in 2D mode (speeds define both axes)
    mover_type: str = "random"  # random, pattern

    # Random movement parameters (for RandomMover)
    # 2D speed pairs: [vx, vy] — each decision picks one pair
    decision_interval: float = 2.0
    speeds: List = field(default_factory=lambda: [
        [0, 0],          # stop
        # [0, -0.04],      # vertical
        [0.02, -0.03],  # diagonal
        [0, 0],
        [-0.02, -0.03],   # diagonal
    ])

    # Training settings
    randomize_init: bool = True
    save_data: bool = True
    max_episode_steps: int = 600

    # Difficulty and tags
    difficulty: str = "hard"
    tags: List[str] = field(default_factory=lambda: ["pick", "single_object", "block", "random_movement", "feedback_advantage"])


class RandomBlockSuccess(TaskSuccess):
    """Success criteria for the Random Block task."""

    def __init__(self):
        super().__init__(
            task_name="pick_random_block",
            task_description="Pick up the randomly moving block"
        )

        # Define success condition - check if block lifted by at least 5cm
        self.add_condition(
            name="Block is lifted",
            check_fn=lambda env: check_object_lifted_relative(
                env, "block_main", min_lift_height=0.1, verbose=self.debug_verbose
            ),
            weight=1.0,
            required=True,
        )

        self.success_threshold = 1.0
        self.debug_verbose = False

    def check_success(self, env, verbose: bool = False) -> Tuple[bool, float, Dict]:
        """
        Check if the block has been successfully picked up.

        Args:
            env: The robosuite environment
            verbose: If True, print detailed success information

        Returns:
            (success, score, details): Success status, score (0-1), and detailed results
        """
        return self.evaluate_conditions(env, self.success_threshold, verbose)


# Create default instances for easy import
RANDOM_BLOCK_CONFIG = RandomBlock2DConfig()
RANDOM_BLOCK_SUCCESS = RandomBlockSuccess()


if __name__ == "__main__":
    # Register environment
    sys.path.insert(0, _robosuite_dir)
    from environment.GreenBlockEnv import GreenBlock
    from env_render import EnvRender

    cfg = RANDOM_BLOCK_CONFIG

    runner = EnvRender(
        env_name=cfg.env_name,
        robot=cfg.robot,
        controller=cfg.controller,
        control_freq=cfg.control_freq,
        teleoperate_method=cfg.teleoperate_method,
        render_camera=cfg.render_camera,
        camera_names=cfg.camera_names,
        show_camera_obs=cfg.show_camera_obs,
        free_view_mode=cfg.free_view_mode,
        randomize_init=cfg.randomize_init,
        mover_type=cfg.mover_type,
        target_joint_name=cfg.target_joint_name,
        speeds=cfg.speeds,
        decision_interval=cfg.decision_interval,
        task_name=cfg.task_name,
        task_success=RANDOM_BLOCK_SUCCESS,
    )

    runner.run()

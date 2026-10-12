"""
Random Block Task Definition

Task: Pick up a block that moves randomly (stops/goes every second).
Environment: RedBlock
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
class RandomBlockConfig(TaskConfig):
    """Configuration for the Random Block task."""

    # Task identification
    task_name: str = "pick_random_block"
    task_description: str = "Pick up the randomly moving block"
    task_id: int = 1

    # Environment settings
    env_name: str = "RedBlock"
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
    free_view_mode: bool = True

    # Task-specific dynamics (block moves randomly)
    target_joint_name: str = "block"
    mover_type: str = "pattern"

    # Random movement parameters (for RandomMover)
    # 2D speed pairs: [vx, vy] — each decision picks one pair
    decision_interval: float = 2.0
    speeds: List = field(default_factory=lambda: [
        [0, 0],          # stop
        [0, -0.04],   # diagonal
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


# Variant with faster, more unpredictable movement
@dataclass
class FastRandomBlockConfig(RandomBlockConfig):
    """Configuration for Fast Random Block - more challenging variant."""

    task_name: str = "pick_fast_random_block"
    task_description: str = "Pick up the fast, randomly moving block"

    # Faster and more unpredictable
    decision_interval: float = 0.5  # Decision every 0.5 seconds (more chaotic)
    speeds: List = field(default_factory=lambda: [
        [0, 0],          # stop
        [0.04, -0.05],   # diagonal fast
        [-0.04, -0.05],  # diagonal fast
        [0, -0.05],      # vertical fast
    ])

    difficulty: str = "very_hard"
    tags: List[str] = field(default_factory=lambda: ["pick", "single_object", "block", "fast_random", "feedback_critical"])


# Variant with slow, steady movement (for comparison/baseline)
@dataclass
class SteadyBlockConfig(RandomBlockConfig):
    """Configuration for Steady Block - constant movement, no randomness."""

    task_name: str = "pick_steady_block"
    task_description: str = "Pick up the steadily moving block"

    # Constant movement, no random stops
    mover_type: str = "random"
    decision_interval: float = 999.0  # Effectively never changes
    speeds: List = field(default_factory=lambda: [
        [0, -0.02],     # steady vertical
    ])

    difficulty: str = "medium"
    tags: List[str] = field(default_factory=lambda: ["pick", "single_object", "block", "constant_movement"])


# Create default instances for easy import
RANDOM_BLOCK_CONFIG = RandomBlockConfig()
RANDOM_BLOCK_SUCCESS = RandomBlockSuccess()

FAST_RANDOM_BLOCK_CONFIG = FastRandomBlockConfig()
STEADY_BLOCK_CONFIG = SteadyBlockConfig()


if __name__ == "__main__":
    # Register environment
    sys.path.insert(0, _robosuite_dir)
    from environment.RedBlockEnv import RedBlock
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

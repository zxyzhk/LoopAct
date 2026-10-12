"""
Drop Ball Task Definition

Task: Drop the ball into the cup that rides on a moving cart.
Environment: DropBall
Difficulty: Hard (cart and cup move horizontally; user must time the release)

Geometry (table, robot, distance, camera) mirrors toy_red exactly; on-table
objects are: a moving cart, a cup mounted on the cart, and a ball that the
robot starts with already gripped.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np

# Add parent directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_robosuite_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _robosuite_dir)

from tasks.base import TaskConfig, TaskSuccess


def _ball_in_cup(env, xy_tol: float = 0.030, z_min: float = 0.0, z_max: float = 0.05) -> bool:
    """True iff ball_main lies inside cup_main's interior.

    xy_tol is the cup's inner half-extent minus the ball radius; the ball
    center must be within ±xy_tol of the cup center in X and Y, and within
    (z_min, z_max) above the cup origin in world Z.
    """
    try:
        ball = env.sim.data.get_body_xpos("ball_main")
        cup = env.sim.data.get_body_xpos("cup_main")
        dxy = np.abs(ball[:2] - cup[:2])
        dz = ball[2] - cup[2]
        return bool(dxy[0] < xy_tol and dxy[1] < xy_tol and z_min < dz < z_max)
    except Exception:
        return False


@dataclass
class DropBallConfig(TaskConfig):
    """Configuration for the Drop Ball task."""

    # Task identification
    task_name: str = "drop_ball"
    task_description: str = "Drop the ball into the cup on the moving cart"
    task_id: int = 11

    # Environment settings
    env_name: str = "DropBall"
    robot: str = "Panda"
    controller: str = "BASIC"
    control_freq: int = 10

    # Teleoperation settings
    teleoperate_method: str = "spacemouse"

    # Rendering and cameras
    render_camera: str = "custom_view"
    camera_names: List[str] = field(default_factory=lambda: ["robot0_eye_in_hand", "agentview"])
    camera_height: int = 128
    camera_width: int = 128
    show_camera_obs: bool = True
    free_view_mode: bool = True

    # Cart moves at this fixed speed along -Y for the whole episode.
    # Options: -0.03 (slow), -0.05 (medium), -0.06 (fast)
    target_joint_name: str = "cart"
    target_speed_y: float = -0.05

    # Training settings
    randomize_init: bool = True
    save_data: bool = True
    max_episode_steps: int = 450

    # Difficulty and tags
    difficulty: str = "hard"
    tags: List[str] = field(default_factory=lambda: ["pick", "ball", "cart", "fixed_speed", "feedback_advantage"])


class DropBallSuccess(TaskSuccess):
    """Success criteria for the Drop Ball task."""

    def __init__(self):
        super().__init__(
            task_name="drop_ball",
            task_description="Drop the ball into the cup on the moving cart",
        )

        self.add_condition(
            name="Ball is inside cup",
            check_fn=_ball_in_cup,
            weight=1.0,
            required=True,
        )

        self.success_threshold = 1.0
        self.debug_verbose = False

    def check_success(self, env, verbose: bool = False) -> Tuple[bool, float, Dict]:
        return self.evaluate_conditions(env, self.success_threshold, verbose)


# Create default instances for easy import
DROP_BALL_CONFIG = DropBallConfig()
DROP_BALL_SUCCESS = DropBallSuccess()


if __name__ == "__main__":
    sys.path.insert(0, _robosuite_dir)
    from environment.DropBallEnv import DropBall  # noqa: F401 — triggers register_env()
    from env_render import EnvRender

    cfg = DROP_BALL_CONFIG

    runner = EnvRender(
        **cfg.to_env_kwargs(),
        task_success=DROP_BALL_SUCCESS,
    )

    runner.run()

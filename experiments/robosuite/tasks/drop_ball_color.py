"""
Drop Ball Color Task Definition

Task: Drop the ball into the round cup that rides on a moving cart.
Environment: DropBall2
Difficulty: Hard (cart and cup move horizontally; user must time the release)

Same geometry and robot configuration as drop_ball, except the cup is a round
(octagonal) orange-red shape instead of a square blue one.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np

_current_dir = os.path.dirname(os.path.abspath(__file__))
_robosuite_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _robosuite_dir)

from tasks.base import TaskConfig, TaskSuccess


def _ball_in_cup(env, xy_tol: float = 0.030, z_min: float = 0.0, z_max: float = 0.05) -> bool:
    """True iff ball_main lies inside cup_main's interior (same check as drop_ball)."""
    try:
        ball = env.sim.data.get_body_xpos("ball_main")
        cup = env.sim.data.get_body_xpos("cup_main")
        dxy = np.abs(ball[:2] - cup[:2])
        dz = ball[2] - cup[2]
        return bool(dxy[0] < xy_tol and dxy[1] < xy_tol and z_min < dz < z_max)
    except Exception:
        return False


@dataclass
class DropBallColorConfig(TaskConfig):
    """Configuration for the Drop Ball Color task (round orange-red cup)."""

    task_name: str = "drop_ball_color"
    task_description: str = "Drop the ball into the round cup on the moving cart"
    task_id: int = 12

    env_name: str = "DropBall2"
    robot: str = "Panda"
    controller: str = "BASIC"
    control_freq: int = 10

    teleoperate_method: str = "spacemouse"

    render_camera: str = "custom_view"
    camera_names: List[str] = field(default_factory=lambda: ["robot0_eye_in_hand", "agentview"])
    camera_height: int = 128
    camera_width: int = 128
    show_camera_obs: bool = True
    free_view_mode: bool = True

    target_joint_name: str = "cart"
    target_speed_y: float = -0.05

    randomize_init: bool = True
    save_data: bool = True
    max_episode_steps: int = 450

    difficulty: str = "hard"
    tags: List[str] = field(default_factory=lambda: ["pick", "ball", "cart", "fixed_speed", "round_cup", "feedback_advantage"])


class DropBallColorSuccess(TaskSuccess):
    """Success criteria for the Drop Ball Color task."""

    def __init__(self):
        super().__init__(
            task_name="drop_ball_color",
            task_description="Drop the ball into the round cup on the moving cart",
        )

        self.add_condition(
            name="Ball is inside round cup",
            check_fn=_ball_in_cup,
            weight=1.0,
            required=True,
        )

        self.success_threshold = 1.0
        self.debug_verbose = False

    def check_success(self, env, verbose: bool = False) -> Tuple[bool, float, Dict]:
        return self.evaluate_conditions(env, self.success_threshold, verbose)


DROP_BALL_COLOR_CONFIG = DropBallColorConfig()
DROP_BALL_COLOR_SUCCESS = DropBallColorSuccess()


if __name__ == "__main__":
    sys.path.insert(0, _robosuite_dir)
    from environment.DropBall2Env import DropBall2  # noqa: F401 — triggers register_env()
    from env_render import EnvRender

    cfg = DROP_BALL_COLOR_CONFIG

    runner = EnvRender(
        **cfg.to_env_kwargs(),
        task_success=DROP_BALL_COLOR_SUCCESS,
    )

    runner.run()

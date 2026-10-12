"""
Drop Ball Speed Task Definition

Same as drop_ball but with a higher cart speed (-0.066 m/s) for generalization testing.
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
    try:
        ball = env.sim.data.get_body_xpos("ball_main")
        cup = env.sim.data.get_body_xpos("cup_main")
        dxy = np.abs(ball[:2] - cup[:2])
        dz = ball[2] - cup[2]
        return bool(dxy[0] < xy_tol and dxy[1] < xy_tol and z_min < dz < z_max)
    except Exception:
        return False


@dataclass
class DropBallSpeedConfig(TaskConfig):
    """Drop Ball with unseen high cart speed (-0.066 m/s)."""

    task_name: str = "drop_ball_speed"
    task_description: str = "Drop the ball into the cup on the fast moving cart"
    task_id: int = 18

    env_name: str = "DropBall"
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
    target_speed_y: float = -0.066

    randomize_init: bool = True
    save_data: bool = True
    max_episode_steps: int = 450

    difficulty: str = "hard"
    tags: List[str] = field(default_factory=lambda: ["pick", "ball", "cart", "fixed_speed", "unseen_speed", "feedback_advantage"])


class DropBallSpeedSuccess(TaskSuccess):
    """Success criteria for the Drop Ball Speed task."""

    def __init__(self):
        super().__init__(
            task_name="drop_ball_speed",
            task_description="Drop the ball into the cup on the fast moving cart",
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


DROP_BALL_SPEED_CONFIG = DropBallSpeedConfig()
DROP_BALL_SPEED_SUCCESS = DropBallSpeedSuccess()


if __name__ == "__main__":
    sys.path.insert(0, _robosuite_dir)
    from environment.DropBallEnv import DropBall  # noqa: F401 — triggers register_env()
    from env_render import EnvRender

    cfg = DROP_BALL_SPEED_CONFIG

    runner = EnvRender(
        **cfg.to_env_kwargs(),
        task_success=DROP_BALL_SPEED_SUCCESS,
    )

    runner.run()

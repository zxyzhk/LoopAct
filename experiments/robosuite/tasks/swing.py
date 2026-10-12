"""
Swing Pendulum Task Definition

Task: Catch the ball on a swinging pendulum.
      A rigid stick hangs from a fixed post and is released from ~45°.
      Physics drives the continuous swinging; the robot must intercept
      and grasp the ball while it moves.

Environment: Swing
Difficulty: Hard (moving target requires visual feedback and timing)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

_current_dir = os.path.dirname(os.path.abspath(__file__))
_robosuite_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _robosuite_dir)

from tasks.base import TaskConfig, TaskSuccess
from utils.success_checks import check_object_grasped


@dataclass
class SwingConfig(TaskConfig):
    """Configuration for the Swing Pendulum task."""

    # Task identification
    task_name: str        = "catch_swing_ball"
    task_description: str = "Catch the ball on the swinging pendulum"
    task_id: int          = 10

    # Environment settings
    env_name: str   = "Swing"
    robot: str      = "Panda"
    controller: str = "BASIC"
    control_freq: int = 10

    # Teleoperation
    teleoperate_method: str = "spacemouse"

    # Rendering and cameras
    render_camera: str = "custom_view"
    camera_names: List[str] = field(
        default_factory=lambda: ["robot0_eye_in_hand", "agentview"]
    )
    camera_height: int   = 128
    camera_width: int    = 128
    show_camera_obs: bool = True
    free_view_mode: bool  = True

    # No external mover — the pendulum is driven by physics
    target_joint_name: Optional[str] = None
    mover_type: str = "none"
    speeds: List    = field(default_factory=list)

    # Training settings
    randomize_init: bool  = True
    save_data: bool       = True
    max_episode_steps: int = 600

    # Difficulty and tags
    difficulty: str = "hard"
    tags: List[str] = field(
        default_factory=lambda: [
            "catch",
            "pendulum",
            "swing",
            "moving_target",
            "timing",
            "feedback_advantage",
        ]
    )


class SwingSuccess(TaskSuccess):
    """Success criteria for the Swing Pendulum task."""

    def __init__(self):
        super().__init__(
            task_name="catch_swing_ball",
            task_description="Catch the ball on the swinging pendulum",
        )

        # Ball must be inside the gripper fingers (distance + closed check)
        self.add_condition(
            name="Ball caught by gripper",
            check_fn=lambda env: check_object_grasped(
                env,
                "swing_ball_body",   # body name as placed in arena worldbody
                threshold=0.07,
            ),
            weight=1.0,
            required=True,
        )

        self.success_threshold = 1.0
        self.debug_verbose     = False

    def check_success(self, env, verbose: bool = False) -> Tuple[bool, float, Dict]:
        return self.evaluate_conditions(env, self.success_threshold, verbose)


# Default instances for easy import
SWING_CONFIG  = SwingConfig()
SWING_SUCCESS = SwingSuccess()


if __name__ == "__main__":
    sys.path.insert(0, _robosuite_dir)
    from environment.SwingEnv import Swing   # noqa: F401 — triggers register_env()
    from env_render import EnvRender

    cfg = SWING_CONFIG

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
        task_name=cfg.task_name,
        task_success=SWING_SUCCESS,
    )

    runner.run()

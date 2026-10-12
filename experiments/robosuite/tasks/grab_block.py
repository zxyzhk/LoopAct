"""
Grab Block Task Definition

Task: Pick up a toy robot figurine placed on a raised block platform.
Environment: GrabBlock
Difficulty: Hard (same movement pattern as toy_red)

Ablation variant of toy_red with the object placed at a higher initial position.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

_current_dir = os.path.dirname(os.path.abspath(__file__))
_robosuite_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _robosuite_dir)

from tasks.base import TaskConfig, TaskSuccess
from utils.success_checks import check_object_lifted_relative


@dataclass
class GrabBlockConfig(TaskConfig):
    """Configuration for the Grab Block task."""

    task_name: str = "grab_block"
    task_description: str = "Pick up the red toy"
    task_id: int = 10

    env_name: str = "GrabBlock"
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

    target_joint_name: str = "toybot"
    mover_type: str = "pattern"

    decision_interval: float = 2.0
    speeds: List = field(default_factory=lambda: [
        [0, 0],
        [0, -0.04]
    ])
    inference_patterns: List = field(default_factory=lambda: [
        [0.000, 0.000, 0.000, -0.040, 0.000, 0.000, 0.000, -0.040], # original
        # [-0.040, -0.040, 0.000, -0.040, -0.040, -0.040, -0.040, 0.000, 0.000]
    ])

    randomize_init: bool = True
    save_data: bool = True
    max_episode_steps: int = 600

    sticky_gripper: bool = False

    difficulty: str = "hard"
    tags: List[str] = field(default_factory=lambda: ["pick", "single_object", "grab_block", "random_movement", "feedback_advantage"])


class GrabBlockSuccess(TaskSuccess):
    """Success criteria for the Grab Block task."""

    def __init__(self):
        super().__init__(
            task_name="grab_block",
            task_description="Pick up the toy on the randomly moving high block",
        )

        self.add_condition(
            name="Toy is lifted",
            check_fn=lambda env: check_object_lifted_relative(
                env, "toybot_main", min_lift_height=0.05, verbose=self.debug_verbose
            ),
            weight=1.0,
            required=True,
        )

        self.success_threshold = 1.0
        self.debug_verbose = False

    def check_success(self, env, verbose: bool = False) -> Tuple[bool, float, Dict]:
        return self.evaluate_conditions(env, self.success_threshold, verbose)


GRAB_BLOCK_CONFIG = GrabBlockConfig()
GRAB_BLOCK_SUCCESS = GrabBlockSuccess()


if __name__ == "__main__":
    sys.path.insert(0, _robosuite_dir)
    from environment.GrabBlockEnv import GrabBlock  # noqa: F401 — triggers register_env()
    from env_render import EnvRender

    cfg = GRAB_BLOCK_CONFIG

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
        task_success=GRAB_BLOCK_SUCCESS,
    )

    runner.run()

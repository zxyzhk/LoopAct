"""
Catch Carrot Task Definition

Task: Catch an orange tapered carrot on a tilted table.
Environment: Cone
Difficulty: Configurable via slope angle

Geometry (table, robot, distance, camera) mirrors toy_can exactly; only the
on-table object differs — a frustum-shaped orange carrot (one end smaller than
the other) replaces the soda can, so it rolls in a circular arc rather than
straight downhill.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

_current_dir = os.path.dirname(os.path.abspath(__file__))
_robosuite_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _robosuite_dir)

from tasks.base import TaskConfig, TaskSuccess
from utils.success_checks import check_object_continue_lifted


@dataclass
class CatchCarrotConfig(TaskConfig):
    """Configuration for the CatchCarrot task."""

    task_name: str = "catch_carrot"
    task_description: str = "Catch the carrot on the tilted table"
    task_id: int = 12

    env_name: str = "Cone"
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

    slope_angle: float = 1.8  # degrees: 2.6, 2.0, 1.2
    # big end is fixed at radius 0.028 (dia 56 mm); small end is adjustable
    cone_end: float = 0.0245 # other end: 0.028; try: 0.0185, 0.0215, 0.0245

    randomize_init: bool = True
    save_data: bool = True
    max_episode_steps: int = 450

    difficulty: str = "easy"
    tags: List[str] = field(default_factory=lambda: ["catch_carrot", "carrot", "environment"])

    def to_env_kwargs(self) -> Dict:
        kwargs = super().to_env_kwargs()
        kwargs["slope_angle"] = self.slope_angle
        kwargs["cone_end"] = self.cone_end
        return kwargs


class CatchCarrotSuccess(TaskSuccess):
    """Success criteria for the CatchCarrot task.

    The carrot rolls in an arc down the tilted table from its initial position,
    so success is detected by checking whether the object's height is
    continuously increasing (i.e. being lifted by the robot).
    """

    def __init__(self):
        super().__init__(
            task_name="catch_carrot",
            task_description="Catch the orange carrot on the tilted table",
        )
        self.debug_verbose = False

        self.add_condition(
            name="Carrot is being lifted",
            check_fn=lambda env: check_object_continue_lifted(
                env, "cone_main", verbose=self.debug_verbose
            ),
            weight=1.0,
            required=True,
        )

        self.success_threshold = 1.0

    def check_success(self, env, verbose: bool = False) -> Tuple[bool, float, Dict]:
        return self.evaluate_conditions(env, self.success_threshold, verbose)


CATCH_CARROT_CONFIG = CatchCarrotConfig()
CATCH_CARROT_SUCCESS = CatchCarrotSuccess()


if __name__ == "__main__":
    sys.path.insert(0, _robosuite_dir)
    from environment.ConeEnv import Cone  # noqa: F401 — triggers register_env()
    from env_render import EnvRender

    cfg = CATCH_CARROT_CONFIG

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
        slope_angle=cfg.slope_angle,
        cone_end=cfg.cone_end,
        task_name=cfg.task_name,
        task_success=CATCH_CARROT_SUCCESS,
    )

    runner.run()

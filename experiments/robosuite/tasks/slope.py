"""
Slope Task Definition

Task: Pick up a cylinder rolling down a wedge ramp.
Environment: Slope
Difficulty: Configurable via slope angle

The slope angle can be changed to vary difficulty. Steeper angles
make the cylinder roll faster, requiring quicker visual feedback responses.
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
from utils.success_checks import check_object_lifted_off_surface


@dataclass
class SlopeConfig(TaskConfig):
    """Configuration for the Slope environment task."""

    # Task identification
    task_name: str = "slope"
    task_description: str = "Pick up the cylinder on the slope"
    task_id: int = 4

    # Environment settings
    env_name: str = "Slope"
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

    # Slope parameters
    slope_angle: float = 1.5  # degrees

    # Training settings
    randomize_init: bool = True
    save_data: bool = True
    max_episode_steps: int = 600

    # Difficulty and tags
    difficulty: str = "easy"
    tags: List[str] = field(default_factory=lambda: ["slope", "environment"])

    def to_env_kwargs(self) -> Dict:
        """Override to include slope_angle."""
        kwargs = super().to_env_kwargs()
        kwargs["slope_angle"] = self.slope_angle
        return kwargs


class SlopeSuccess(TaskSuccess):
    """Success criteria for the Slope task.

    The cylinder rolls down the slope from its initial position,
    so success is detected by checking whether the object's height
    is continuously increasing (i.e. being lifted by the robot).
    """

    def __init__(self):
        super().__init__(
            task_name="slope",
            task_description="Pick up the cylinder on the slope",
        )
        self.debug_verbose = False

        self.add_condition(
            name="Cylinder is being lifted",
            check_fn=lambda env: check_object_lifted_off_surface(
                env, "cylinder_main",
                surface_geoms=["slope_collision", "table_collision"],
                verbose=self.debug_verbose,
            ),
            weight=1.0,
            required=True,
        )

        self.success_threshold = 1.0

    def check_success(self, env, verbose: bool = False) -> Tuple[bool, float, Dict]:
        return self.evaluate_conditions(env, self.success_threshold, verbose)


# Create default instances for easy import
SLOPE_CONFIG = SlopeConfig()
SLOPE_SUCCESS = SlopeSuccess()


if __name__ == "__main__":
    # Register environment
    sys.path.insert(0, _robosuite_dir)
    from environment.SlopeEnv import Slope
    from env_render import EnvRender

    cfg = SLOPE_CONFIG

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
        task_name=cfg.task_name,
        task_success=SLOPE_SUCCESS,
    )

    runner.run()

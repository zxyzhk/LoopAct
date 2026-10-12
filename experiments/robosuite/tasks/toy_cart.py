"""
Toy Cart Task Definition

Task: Pick up a red block from a flat cart that moves randomly (stops/slides every few seconds).
Environment: ToyCart
Difficulty: Hard (unpredictable movement requires visual feedback)

Mirrors random_block but with a flat cart object instead of a block.
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
class ToyCartConfig(TaskConfig):
    """Configuration for the Toy Cart task."""

    # Task identification
    task_name: str = "pick_toy_cart"
    task_description: str = "Pick up the red block from the back of the moving toy cart"
    task_id: int = 5

    # Environment settings
    env_name: str = "ToyCart"
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
    free_view_mode: bool = False

    # Cart moves randomly; block rides along on the bed
    target_joint_name: str = "truck"
    mover_type: str = "pattern"

    # 2D speed pairs: [vx, vy] — each decision picks one pair
    decision_interval: float = 2.0
    speeds: List = field(default_factory=lambda: [
        [0, 0],        # stop
        [0, -0.04],    # slide forward
    ])

    # Training settings
    randomize_init: bool = True
    save_data: bool = True
    max_episode_steps: int = 600

    # Difficulty and tags
    difficulty: str = "hard"
    tags: List[str] = field(default_factory=lambda: ["pick", "single_object", "cart", "random_movement", "feedback_advantage"])


class ToyCartSuccess(TaskSuccess):
    """Success criteria for the Toy Cart task."""

    def __init__(self):
        super().__init__(
            task_name="pick_toy_cart",
            task_description="Pick up the red block from the back of the moving toy cart",
        )

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
        return self.evaluate_conditions(env, self.success_threshold, verbose)


# Create default instances for easy import
TOY_CART_CONFIG = ToyCartConfig()
TOY_CART_SUCCESS = ToyCartSuccess()


if __name__ == "__main__":
    sys.path.insert(0, _robosuite_dir)
    from environment.ToyCartEnv import ToyCart  # noqa: F401 — triggers register_env()
    from env_render import EnvRender

    cfg = TOY_CART_CONFIG

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
        task_success=TOY_CART_SUCCESS,
    )

    runner.run()

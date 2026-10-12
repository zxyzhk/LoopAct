"""
Block Cart Task Definition

Task: Pick up the blue block riding on a moving cart.
Environment: BlockCart

Geometry mirrors toy_red exactly. The target object is a blue block
sitting on top of a moving cart (single rigid body), with the same
physical collision footprint as the toy robot and the same total height
above the table surface — making this a direct visual swap for
baseline vs feedback comparison.
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
class BlockCartConfig(TaskConfig):
    """Configuration for the Block Cart task."""

    task_name: str = "block_cart"
    task_description: str = "Pick up the blue block on the moving cart"
    task_id: int = 12

    env_name: str = "BlockCart"
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

    target_joint_name: str = "toycart"
    mover_type: str = "pattern"

    decision_interval: float = 2.0
    speeds: List = field(default_factory=lambda: [
        [0, 0],
        [0, -0.04],
    ])

    randomize_init: bool = True
    save_data: bool = True
    max_episode_steps: int = 600

    difficulty: str = "hard"
    tags: List[str] = field(default_factory=lambda: ["pick", "block", "cart", "moving", "feedback_advantage"])


class BlockCartSuccess(TaskSuccess):
    """Success: blue block lifted >= 10 cm above its initial height."""

    def __init__(self):
        super().__init__(
            task_name="block_cart",
            task_description="Pick up the blue block on the moving cart",
        )
        self.debug_verbose = False

        self.add_condition(
            name="Blue block is lifted",
            check_fn=lambda env: check_object_lifted_relative(
                env, "blue_block_main", min_lift_height=0.1, verbose=self.debug_verbose
            ),
            weight=1.0,
            required=True,
        )

        self.success_threshold = 1.0

    def check_success(self, env, verbose: bool = False) -> Tuple[bool, float, Dict]:
        return self.evaluate_conditions(env, self.success_threshold, verbose)


BLOCK_CART_CONFIG = BlockCartConfig()
BLOCK_CART_SUCCESS = BlockCartSuccess()

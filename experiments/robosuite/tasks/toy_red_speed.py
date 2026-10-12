"""
Toy Red Speed Task Definition

Same as toy_red but with an unseen speed pattern (0.300 m/s phase) for generalization testing.
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
class ToyRedSpeedConfig(TaskConfig):
    """Toy Red with unseen high-speed inference pattern."""

    task_name: str = "pick_toy_red_speed"
    task_description: str = "Pick up the red toy"
    task_id: int = 17

    env_name: str = "ToyRed"
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
        [0.000, -0.050, 0.000, 0.000, 0.300, -0.030, 0.000, -0.050],
    ])

    randomize_init: bool = True
    save_data: bool = True
    max_episode_steps: int = 600

    difficulty: str = "hard"
    tags: List[str] = field(default_factory=lambda: ["pick", "single_object", "toy_red", "random_movement", "unseen_speed"])


class ToyRedSpeedSuccess(TaskSuccess):
    """Success criteria for the Toy Red Speed task."""

    def __init__(self):
        super().__init__(
            task_name="pick_toy_red_speed",
            task_description="Pick up the randomly moving red toy robot (unseen speed)",
        )

        self.add_condition(
            name="Red toy robot is lifted",
            check_fn=lambda env: check_object_lifted_relative(
                env, "toybot_main", min_lift_height=0.1, verbose=self.debug_verbose
            ),
            weight=1.0,
            required=True,
        )

        self.success_threshold = 1.0
        self.debug_verbose = False

    def check_success(self, env, verbose: bool = False) -> Tuple[bool, float, Dict]:
        return self.evaluate_conditions(env, self.success_threshold, verbose)


TOY_RED_SPEED_CONFIG = ToyRedSpeedConfig()
TOY_RED_SPEED_SUCCESS = ToyRedSpeedSuccess()

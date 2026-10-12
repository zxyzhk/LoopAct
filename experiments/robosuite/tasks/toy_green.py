"""
Toy Green Task Definition

Task: Pick up a green toy robot figurine that moves randomly (stops/slides in 2D every few seconds).
Environment: ToyGreen
Difficulty: Hard (unpredictable movement requires visual feedback)

Same as toy_red but with a green-coloured figurine and 2D diagonal movement pattern.
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
class ToyGreenConfig(TaskConfig):
    """Configuration for the Toy Green Robot task."""

    # Task identification
    task_name: str = "pick_toy_green"
    task_description: str = "Pick up the green toy"
    task_id: int = 9

    # Environment settings
    env_name: str = "ToyGreen"
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

    # Toy robot moves randomly in 2D (2D diagonal, matches toy_robot mover)
    target_joint_name: str = "toybot"
    mover_type: str = "pattern"

    # 2D speed pairs: [vx, vy] — each decision picks one pair
    decision_interval: float = 2.0
    speeds: List = field(default_factory=lambda: [
        [0, 0],           # stop
        [0, 0],
        [0.012, -0.038],    # diagonal
        [-0.012, -0.038],   # diagonal
    ])
    inference_patterns: List = field(default_factory=lambda: [ # original
        [
            [ 0.012, -0.038], [0.000, 0.000], [ 0.012, -0.038], [-0.012, -0.038], [ 0.012, -0.038],
            [ 0.000,  0.000], [-0.012, -0.038], [-0.012, -0.038], [0.000,  0.000], [ 0.000,  0.000],
        ],
    ])

    # inference_patterns: List = field(default_factory=lambda: [
    #     [
    #         [ -0.012, -0.038],[0.000,  0.000],[ 0.012, -0.038], [-0.012, -0.038], [ 0.012, -0.038],
    #         [ 0.000,  0.000], [-0.012, -0.038], [-0.012, -0.038], [0.000,  0.000], [ 0.000,  0.000],
    #     ],
    # ])

    # Training settings
    randomize_init: bool = True
    save_data: bool = True
    max_episode_steps: int = 300

    # Difficulty and tags
    difficulty: str = "hard"
    tags: List[str] = field(default_factory=lambda: ["pick", "single_object", "toy_green", "random_movement", "feedback_advantage"])


class ToyGreenSuccess(TaskSuccess):
    """Success criteria for the Toy Green Robot task."""

    def __init__(self):
        super().__init__(
            task_name="pick_toy_green",
            task_description="Pick up the randomly moving green toy",
        )

        self.add_condition(
            name="Green toy robot is lifted",
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


# Create default instances for easy import
TOY_GREEN_CONFIG = ToyGreenConfig()
TOY_GREEN_SUCCESS = ToyGreenSuccess()


if __name__ == "__main__":
    sys.path.insert(0, _robosuite_dir)
    from environment.ToyGreenEnv import ToyGreen  # noqa: F401 — triggers register_env()
    from env_render import EnvRender

    cfg = TOY_GREEN_CONFIG

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
        task_success=TOY_GREEN_SUCCESS,
    )

    runner.run()

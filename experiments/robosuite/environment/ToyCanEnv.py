#!/usr/bin/env python3
"""
ToyCan Environment - Tilted table with objects.

The slope angle is set at environment creation via the slope_angle parameter.
The table itself is tilted to create the incline.
"""

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import Task

import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_robosuite_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _robosuite_dir)

from xml_objects import SodaCanObject
from robosuite.environments.base import register_env
from robosuite.utils.mjcf_utils import find_elements, new_element, array_to_string
from dataclasses import dataclass, field
from typing import List, Optional
from env_render import EnvRender
from tasks.base import TaskSuccess


class ToyCan(ManipulationEnv):
    """Environment with a tilted table."""

    # Default Panda pose — q1 lowered from -0.3 to -0.5 to reduce initial z travel
    INIT_QPOS = np.array([0.0, -0.1, 0.0, -1.9, 0.0, 1.8, 0.785])

    ROBOT_QPOS_NOISE = 0.05

    def __init__(
        self,
        slope_angle: float = 5.0,
        randomize_init: bool = True,
        task_name: Optional[str] = None,
        task_success: Optional[TaskSuccess] = None,
        **kwargs
    ):
        # Initialize attributes BEFORE super().__init__()
        self.slope_angle = slope_angle  # degrees
        self.randomize_init = randomize_init
        self.object_base_positions = {}
        self.object_base_rotations = {}

        # Task setup
        self.task_name = task_name or "toy_can"
        self.task_success = task_success
        self.current_success = False
        self.success_checked = False

        super().__init__(**kwargs)

        if self.task_success:
            print(f"\n[TASK] Loaded: {self.task_success.task_description}")

    def _reset_internal(self):
        """Reset with optional randomization."""
        super()._reset_internal()

        # Override agentview camera with custom viewer position
        if hasattr(self.sim.model, 'camera_names'):
            for cam_name in self.sim.model.camera_names:
                if cam_name == 'agentview':
                    cam_id = self.sim.model.camera_name2id(cam_name)
                    self.sim.model.cam_pos[cam_id] = np.array([1.1536, -0.0076, 1.7745])
                    self.sim.model.cam_quat[cam_id] = np.array([0.6260, 0.3337, 0.3316, 0.6219])
                    self.sim.model.cam_fovy[cam_id] = 50.0
                    break

        print(f"[INIT] Slope angle: {self.slope_angle:.1f} degrees")

        # Lay cylinder on its side on the tilted table
        joint_name = next(
            (jname for jname in self.sim.model.joint_names if "cylinder" in jname),
            None,
        )
        if joint_name:
            base_pos = self.object_base_positions["cylinder"]
            # 90° around Y: rotates cylinder axis from Z to X (laying on side)
            quat = np.array([0.7071, 0.0, 0.7071, 0.0])
            full_qpos = np.concatenate([base_pos, quat])
            self.sim.data.set_joint_qpos(joint_name, full_qpos)

        # Set initial robot joint positions
        robot = self.robots[0]
        robot_qpos = []
        if hasattr(robot, 'init_qpos') and self.INIT_QPOS is not None:
            robot_qpos_indices = robot._ref_joint_pos_indexes

            for i, idx in enumerate(robot_qpos_indices):
                if i < len(self.INIT_QPOS):
                    if self.randomize_init:
                        noise = np.random.uniform(-self.ROBOT_QPOS_NOISE, self.ROBOT_QPOS_NOISE)
                        qpos = self.INIT_QPOS[i] + noise
                        self.sim.data.qpos[idx] = qpos
                        robot_qpos.append(qpos)
                    else:
                        self.sim.data.qpos[idx] = self.INIT_QPOS[i]
                        robot_qpos.append(self.INIT_QPOS[i])

        if robot_qpos:
            print(f"[INIT] Robot joints: {np.array2string(np.array(robot_qpos), precision=3, suppress_small=True)}")

        self.sim.forward()

    def _load_model(self):
        super()._load_model()

        # Arena setup
        self.table_offset = np.array((0.0, 0, 0.875))
        self.table_size = np.array((1.3, 2.0, 0.05))
        arena = TableArena(
            table_full_size=self.table_size,
            table_offset=self.table_offset,
            has_legs=True
        )
        arena.set_origin([0.0, 0.0, 0.0])
        arena.worldbody.append(
            new_element(
                "camera",
                name="custom_view",
                mode="fixed",
                pos="-1.67298636 0.79797356 1.75123554",
                quat="-0.44254917 -0.29815212 0.47254251 0.70139798",
                fovy="50",
            )
        )

        # --- Lower table surface friction ---
        for geom in arena.worldbody.iter("geom"):
            if geom.get("name") == "table_collision":
                geom.set("friction", "0.1 0.005 0.0001")
                break

        # --- Tilt the table ---
        angle_rad = np.radians(self.slope_angle)
        # Rotate around X-axis so the table slopes along Y
        tilt_quat = np.array([np.cos(angle_rad / 2), np.sin(angle_rad / 2), 0, 0])
        arena.table_body.set("quat", array_to_string(tilt_quat))

        table_top_z = self.table_offset[2]

        print(f"[MODEL] Tilted table: angle={self.slope_angle:.1f}deg")

        # --- Soda can on the tilted table ---
        cyl_radius = 0.028  # matches soda_can.xml collision geom radius
        can_obj = SodaCanObject(name="cylinder")

        cyl_x = -0.2
        cyl_y = 0.2
        cyl_z = table_top_z + cyl_y * np.sin(angle_rad) + cyl_radius + 0.005
        cyl_pos = f"{cyl_x} {cyl_y} {cyl_z}"

        obj_xml = can_obj.get_obj()
        obj_xml.set("pos", cyl_pos)

        # Store base position for reset
        self.object_base_positions["cylinder"] = np.array([cyl_x, cyl_y, cyl_z])

        # Store joint name
        self.object_joints = {}
        joint_name = None
        joints = getattr(can_obj, "joints", None)
        if joints:
            if isinstance(joints[0], str):
                joint_name = joints[0]
            else:
                joint_names = getattr(can_obj, "joint_names", None)
                if joint_names:
                    joint_name = joint_names[0]
        self.cylinder_joint = joint_name
        self.object_joints["cylinder"] = joint_name

        objects = [("cylinder", can_obj, cyl_pos)]

        # Set robot base position
        robot_model = self.robots[0].robot_model
        root_body_name = robot_model.root_body
        if root_body_name:
            body_elements = find_elements(
                root=robot_model.root,
                tags="body",
                attribs={"name": root_body_name},
                return_first=True
            )
            if body_elements is not None:
                body_elements.set("pos", f"-0.78 0.0 {table_top_z + 0.02}")

        # Build Task
        mujoco_objects = [obj for _, obj, _ in objects]
        self.model = Task(
            mujoco_arena=arena,
            mujoco_robots=self.robots[0].robot_model,
            mujoco_objects=mujoco_objects,
        )

    def reward(self, action):
        return 0.0

    def check_success(self, verbose: bool = False):
        """Check if the current task has been completed successfully."""
        if self.task_success is None:
            return False, 0.0, {}

        success, score, details = self.task_success.check_success(self, verbose=verbose)
        self.current_success = success
        self.success_checked = True

        return success, score, details

    def get_task_language(self) -> str:
        """Get natural language description of current task."""
        if self.task_success:
            return self.task_success.task_description
        return self.task_name

    def get_task_id(self) -> int:
        """Get task ID for multi-task learning."""
        return 0


@dataclass
class ToyCanEnvConfig:
    env_name: str = "ToyCan"
    robot: str = "Panda"
    controller: str = "BASIC"
    control_freq: int = 10
    teleoperate_method: str = "spacemouse"
    slope_angle: float = 2.0  # degrees
    render_camera: str = "custom_view"
    camera_names: List[str] = field(default_factory=lambda: ["robot0_eye_in_hand", "agentview"])
    show_camera_obs: bool = True
    free_view_mode: bool = True
    randomize_init: bool = True
    save_data: bool = True
    task_name: str = "toy_can"


TOY_CAN_ENV_CONFIG = ToyCanEnvConfig()

register_env(ToyCan)


if __name__ == "__main__":
    cfg = TOY_CAN_ENV_CONFIG

    runner = EnvRender(
        env_name=cfg.env_name,
        robot=cfg.robot,
        controller=cfg.controller,
        control_freq=cfg.control_freq,
        teleoperate_method=cfg.teleoperate_method,
        slope_angle=cfg.slope_angle,
        render_camera=cfg.render_camera,
        camera_names=cfg.camera_names,
        show_camera_obs=cfg.show_camera_obs,
        free_view_mode=cfg.free_view_mode,
        randomize_init=cfg.randomize_init,
    )

    runner.run()

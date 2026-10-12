#!/usr/bin/env python3
"""
Cone Environment - Tilted table with a yellow tapered cone.

Identical geometry to ToyCanEnv (table, robot, distance, camera, slope) — only
the on-table object differs: a frustum-shaped (tapered) cone replaces the soda
can. Because the two ends have different radii, the cone traces a circular arc
as it rolls down the slope, drifting toward one side of the table.
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

from xml_objects import ConeObject
from robosuite.environments.base import register_env
from robosuite.utils.mjcf_utils import find_elements, new_element, array_to_string
from dataclasses import dataclass, field
from typing import List, Optional
from env_render import EnvRender
from tasks.base import TaskSuccess


class Cone(ManipulationEnv):
    """Environment with a tilted table and a yellow tapered rolling cone."""

    # Default Panda pose (matches ToyCan)
    INIT_QPOS = np.array([0.0, -0.3, 0.0, -1.9, 0.0, 1.8, 0.785])

    ROBOT_QPOS_NOISE = 0.05

    # Big-end radius of the cone (matches cone.xml seg0)
    CONE_BIG_RADIUS = 0.028

    def __init__(
        self,
        slope_angle: float = 5.0,
        cone_end: float = 0.0185,
        randomize_init: bool = True,
        task_name: Optional[str] = None,
        task_success: Optional[TaskSuccess] = None,
        **kwargs
    ):
        self.slope_angle = slope_angle  # degrees
        self.cone_end = cone_end        # small end radius
        self.randomize_init = randomize_init
        self.object_base_positions = {}
        self.object_base_rotations = {}

        self.task_name = task_name or "catch_carrot"
        self.task_success = task_success
        self.current_success = False
        self.success_checked = False

        super().__init__(**kwargs)

        if self.task_success:
            print(f"\n[TASK] Loaded: {self.task_success.task_description}")

    def _reset_internal(self):
        super()._reset_internal()

        if hasattr(self.sim.model, 'camera_names'):
            for cam_name in self.sim.model.camera_names:
                if cam_name == 'agentview':
                    cam_id = self.sim.model.camera_name2id(cam_name)
                    self.sim.model.cam_pos[cam_id] = np.array([1.1536, -0.0076, 1.7745])
                    self.sim.model.cam_quat[cam_id] = np.array([0.6260, 0.3337, 0.3316, 0.6219])
                    self.sim.model.cam_fovy[cam_id] = 50.0
                    break

        print(f"[INIT] Slope angle: {self.slope_angle:.1f} degrees")

        # Lay the cone on its side: -90° around Y maps the local long axis (Z)
        # to world X, with the small end at -X (robot side) and big end at +X.
        # The cone then pivots around the small end as it rolls down -Y, so the
        # big end sweeps an outer arc and the cone curves toward -X (the robot).
        joint_name = self.object_joints.get("cone")
        if joint_name:
            base_pos = self.object_base_positions["cone"]
            quat = np.array([0.7071, 0.0, -0.7071, 0.0])
            full_qpos = np.concatenate([base_pos, quat])
            self.sim.data.set_joint_qpos(joint_name, full_qpos)

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

        # Arena setup — identical to ToyCan
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

        # Lower table surface friction (so the cone rolls down)
        for geom in arena.worldbody.iter("geom"):
            if geom.get("name") == "table_collision":
                geom.set("friction", "0.1 0.005 0.0001")
                break

        angle_rad = np.radians(self.slope_angle)
        tilt_quat = np.array([np.cos(angle_rad / 2), np.sin(angle_rad / 2), 0, 0])
        arena.table_body.set("quat", array_to_string(tilt_quat))

        table_top_z = self.table_offset[2]

        print(f"[MODEL] Tilted table: angle={self.slope_angle:.1f}deg")

        # Cone replaces the soda can; same XY placement as ToyCan. Use the
        # big-end radius for the resting height since that's the widest
        # cross-section along the contact line when laid on its side.
        cone_obj = ConeObject(name="cone")

        cone_x = -0.2
        cone_y = 0.2
        cone_z = table_top_z + cone_y * np.sin(angle_rad) + self.CONE_BIG_RADIUS + 0.005
        cone_pos = f"{cone_x} {cone_y} {cone_z}"

        obj_xml = cone_obj.get_obj()
        obj_xml.set("pos", cone_pos)

        # Apply parametric small-end radius: rewrite all 10 segment geoms so
        # the taper runs linearly from CONE_BIG_RADIUS (seg0) to cone_end (seg9).
        n = 10
        seg_half_h = 0.0065
        for elem in obj_xml.iter("geom"):
            name = elem.get("name", "")
            for i in range(n):
                if f"seg{i}_col" in name or f"seg{i}_vis" in name:
                    r = self.CONE_BIG_RADIUS + (self.cone_end - self.CONE_BIG_RADIUS) * i / (n - 1)
                    elem.set("size", f"{r:.4f} {seg_half_h}")
                    break
            if "small_cap" in name:
                elem.set("size", f"{self.cone_end:.4f} 0.003")

        self.object_base_positions["cone"] = np.array([cone_x, cone_y, cone_z])

        self.object_joints = {}
        joint_name = None
        joints = getattr(cone_obj, "joints", None)
        if joints:
            if isinstance(joints[0], str):
                joint_name = joints[0]
            else:
                joint_names = getattr(cone_obj, "joint_names", None)
                if joint_names:
                    joint_name = joint_names[0]
        self.cone_joint = joint_name
        self.object_joints["cone"] = joint_name

        objects = [("cone", cone_obj, cone_pos)]

        # Robot base position (matches ToyCan)
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

        mujoco_objects = [obj for _, obj, _ in objects]
        self.model = Task(
            mujoco_arena=arena,
            mujoco_robots=self.robots[0].robot_model,
            mujoco_objects=mujoco_objects,
        )

    def reward(self, action):
        return 0.0

    def check_success(self, verbose: bool = False):
        if self.task_success is None:
            return False, 0.0, {}

        success, score, details = self.task_success.check_success(self, verbose=verbose)
        self.current_success = success
        self.success_checked = True

        return success, score, details

    def get_task_language(self) -> str:
        if self.task_success:
            return self.task_success.task_description
        return self.task_name

    def get_task_id(self) -> int:
        return 0


@dataclass
class ConeEnvConfig:
    env_name: str = "Cone"
    robot: str = "Panda"
    controller: str = "BASIC"
    control_freq: int = 10
    teleoperate_method: str = "spacemouse"
    slope_angle: float = 2.0
    render_camera: str = "custom_view"
    camera_names: List[str] = field(default_factory=lambda: ["robot0_eye_in_hand", "agentview"])
    show_camera_obs: bool = True
    free_view_mode: bool = True
    randomize_init: bool = True
    save_data: bool = True
    task_name: str = "catch_carrot"


CONE_ENV_CONFIG = ConeEnvConfig()

register_env(Cone)


if __name__ == "__main__":
    cfg = CONE_ENV_CONFIG

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

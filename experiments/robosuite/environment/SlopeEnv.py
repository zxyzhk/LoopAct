#!/usr/bin/env python3
"""
Slope Environment - Table with a triangular wedge ramp.

The slope angle is set at environment creation via the slope_angle parameter.
The wedge mesh is generated dynamically from the angle.
Objects can be added later to test interactions with the inclined surface.
"""

import numpy as np
import xml.etree.ElementTree as ET
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import Task
from robosuite.models.objects.primitive import CylinderObject

import sys
import os
if __name__ == "__main__":
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _robosuite_dir = os.path.dirname(_current_dir)
    sys.path.insert(0, _robosuite_dir)

from robosuite.environments.base import register_env
from robosuite.utils.mjcf_utils import find_elements, new_element, new_body
from dataclasses import dataclass, field
from typing import List, Optional
from env_render import EnvRender
from tasks.base import TaskSuccess


class Slope(ManipulationEnv):
    """Environment with a table and a triangular wedge ramp."""

    # Default Panda pose
    INIT_QPOS = np.array([0.0, -0.3, 0.0, -1.9, 0.0, 1.8, 0.785])

    ROBOT_QPOS_NOISE = 0.05

    # Slope position on table (x, y)
    SLOPE_POS_XY = np.array([-0.1, 0.0])

    def __init__(
        self,
        slope_angle: float = 5.0,
        slope_half_length: float = 0.5,
        slope_half_width: float = 0.10,
        randomize_init: bool = True,
        task_name: Optional[str] = None,
        task_success: Optional[TaskSuccess] = None,
        **kwargs
    ):
        # Initialize attributes BEFORE super().__init__()
        self.slope_angle = slope_angle  # degrees
        self.slope_half_length = slope_half_length
        self.slope_half_width = slope_half_width
        self.randomize_init = randomize_init
        self.object_base_positions = {}
        self.object_base_rotations = {}

        # Task setup
        self.task_name = task_name or "slope"
        self.task_success = task_success
        self.current_success = False
        self.success_checked = False

        super().__init__(**kwargs)

        if self.task_success:
            print(f"\n[TASK] Loaded: {self.task_success.task_description}")

    def _make_wedge_vertices(self):
        """
        Generate the 6 vertices of a triangular wedge (ramp) based on slope_angle.

        Cross-section (side view along Y):

            high end (+X)
               /|
              / |  H
             /  |
            /___|
        low end   2*L

        Returns:
            vertex_str: Space-separated vertex string for MuJoCo mesh
            height: The computed wedge height
        """
        L = self.slope_half_length
        W = self.slope_half_width
        H = 2 * L * np.tan(np.radians(self.slope_angle))

        vertices = [
            (-L, -W, 0.0), (-L,  W, 0.0),   # Low end (ground level)
            ( L, -W, 0.0), ( L,  W, 0.0),   # Bottom of high end
            ( L, -W, H),   ( L,  W, H),     # Top of high end
        ]
        vertex_str = " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
        return vertex_str, H

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

        # Lay cylinder on its side on the slope
        joint_name = self.object_joints.get("cylinder")
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

        # Arena setup (same as OneBlockEnv)
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

        # --- Wedge mesh slope ---
        # Generate triangular wedge vertices from slope_angle
        vertex_str, wedge_height = self._make_wedge_vertices()

        # Add mesh and material to arena's asset section
        mesh_elem = ET.SubElement(arena.asset, "mesh")
        mesh_elem.set("name", "slope_wedge_mesh")
        mesh_elem.set("vertex", vertex_str)

        material_elem = ET.SubElement(arena.asset, "material")
        material_elem.set("name", "slope_mat")
        material_elem.set("rgba", "0.72 0.53 0.35 1")

        # Place slope body on the table surface
        table_top_z = self.table_offset[2]
        slope_pos = [self.SLOPE_POS_XY[0], self.SLOPE_POS_XY[1], table_top_z]
        # quat: 90deg around Z = [0.7071, 0, 0, 0.7071] (slope runs along Y instead of X)
        slope_body = new_body(name="slope_body", pos=slope_pos, quat=[0.7071, 0, 0, 0.7071])

        # Collision geom (group 0) - for physics
        slope_body.append(new_element(
            "geom", "slope_collision",
            type="mesh", mesh="slope_wedge_mesh",
            group="0", friction="0.8 0.01 0.001",
        ))
        # Visual geom (group 1) - for rendering
        slope_body.append(new_element(
            "geom", "slope_visual",
            type="mesh", mesh="slope_wedge_mesh",
            group="1", contype="0", conaffinity="0",
            material="slope_mat",
        ))
        arena.worldbody.append(slope_body)

        print(f"[MODEL] Wedge slope: angle={self.slope_angle:.1f}deg, "
              f"length={2*self.slope_half_length:.2f}m, "
              f"width={2*self.slope_half_width:.2f}m, "
              f"height={wedge_height:.3f}m")

        # --- Cylinder on the slope ---
        cyl_radius = 0.02
        cyl_half_len = 0.03
        cylinder_obj = CylinderObject(
            name="cylinder",
            size=[cyl_radius, cyl_half_len],
            rgba=[0.8, 0.2, 0.2, 1.0],  # Red
            density=500,
            friction=[0.8, 0.01, 0.001],  # Lower friction so it rolls
        )

        # Compute initial position: 80% up the slope
        cyl_frac = 0.90
        L = self.slope_half_length
        body_x = L * (2 * cyl_frac - 1)
        z_surface = (body_x + L) * np.tan(np.radians(self.slope_angle))
        # Transform slope body frame → world (90° Z rotation: body_x → world_y)
        cyl_x = self.SLOPE_POS_XY[0]
        cyl_y = self.SLOPE_POS_XY[1] + body_x
        cyl_z = table_top_z + z_surface + cyl_radius + 0.005
        cyl_pos = f"{cyl_x} {cyl_y} {cyl_z}"

        obj_xml = cylinder_obj.get_obj()
        obj_xml.set("pos", cyl_pos)

        # Store base position for reset
        self.object_base_positions["cylinder"] = np.array([cyl_x, cyl_y, cyl_z])

        # Store joint name
        self.object_joints = {}
        joint_name = None
        joints = getattr(cylinder_obj, "joints", None)
        if joints:
            if isinstance(joints[0], str):
                joint_name = joints[0]
            else:
                joint_names = getattr(cylinder_obj, "joint_names", None)
                if joint_names:
                    joint_name = joint_names[0]
        self.cylinder_joint = joint_name
        self.object_joints["cylinder"] = joint_name

        objects = [("cylinder", cylinder_obj, cyl_pos)]

        # Set robot base position (same as OneBlockEnv)
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
                body_elements.set("pos", f"-0.7 0.0 {table_top_z}")

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
class SlopeEnvConfig:
    env_name: str = "Slope"
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
    task_name: str = "slope"


SLOPE_ENV_CONFIG = SlopeEnvConfig()

register_env(Slope)


if __name__ == "__main__":
    cfg = SLOPE_ENV_CONFIG

    runner = EnvRender(
        env_name=cfg.env_name,
        robot=cfg.robot,
        controller=cfg.controller,
        control_freq=cfg.control_freq,
        teleoperate_method=cfg.teleoperate_method,
        slope_angle = cfg.slope_angle,
        render_camera=cfg.render_camera,
        camera_names=cfg.camera_names,
        show_camera_obs=cfg.show_camera_obs,
        free_view_mode=cfg.free_view_mode,
        randomize_init=cfg.randomize_init,
    )

    runner.run()

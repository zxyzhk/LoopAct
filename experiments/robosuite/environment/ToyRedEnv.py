#!/usr/bin/env python3
"""
ToyRed Environment - Same as ToyRobotEnv but with a red-coloured figurine.
"""

import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_robosuite_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _robosuite_dir)

import numpy as np
from robosuite.environments.base import register_env
from robosuite.models.tasks import Task
from robosuite.models.objects.primitive import BoxObject, BallObject
from xml_objects import ToyRobotRedObject, ToyCartObject
from environment.ToyRobotEnv import ToyRobot


class ToyRed(ToyRobot):
    """ToyRobot environment with a red figurine instead of the default blue one."""

    INIT_QPOS = np.array([-0.007, -0.283, -0.05, -1.86, -0.012, 1.821, 0.755])
    OBJECT_POS_NOISE_XY = 0.03
    ROBOT_QPOS_NOISE = 0.03

    def _load_model(self):
        # Call grandparent to set up arena and robot, skipping ToyRobot._load_model
        from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
        from robosuite.models.arenas import TableArena
        from robosuite.utils.mjcf_utils import find_elements, new_element

        ManipulationEnv._load_model(self)

        self.table_offset = np.array((0.0, 0, 0.757))
        self.robot_base_z = 0.875  # keep arm at original height independent of table
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

        table_surface_z = self.table_offset[2]                        # 0.757 — actual table surface
        table_top_z     = table_surface_z + self.table_size[2] / 2  # 0.782 — matches HighBlockEnv

        toybot_obj = ToyRobotRedObject(name="toybot")

        # Distractor cluster on the right side of the table (away from toybot's path)
        cart_obj = ToyCartObject(name="cart")
        # cart platform top at local z=0.016; green box centre = 0.016 + half_z(0.025)
        green_box_obj = BoxObject(
            name="green_box",
            size=[0.025, 0.025, 0.025],
            rgba=[0.15, 0.75, 0.20, 1.0],
            density=500,
            friction=(1.5, 0.01, 0.001),
        )
        blue_box_obj = BoxObject(
            name="blue_box",
            size=[0.025, 0.025, 0.025],
            rgba=[0.15, 0.30, 0.90, 1.0],
            density=500,
            friction=(1.5, 0.01, 0.001),
        )
        orange_ball_obj = BallObject(
            name="orange_ball",
            size=[0.038],
            rgba=[0.95, 0.50, 0.05, 1.0],
            density=500,
            friction=(1.5, 0.01, 0.001),
        )

        # 30° counter-clockwise around Z: quat [w, x, y, z], negate z for CCW
        _rot30z_ccw = f"{np.cos(np.pi/12):.6f} 0 0 {-np.sin(np.pi/12):.6f}"
        _object_quats = {"cart": _rot30z_ccw, "green_box": _rot30z_ccw}

        # Original base positions (before fixed-pose snapshot)
        # cart_x, cart_y = -0.4, 0.4
        # objects = [
        #     ("toybot",      toybot_obj,      f"-0.10  0.20 {table_top_z:.4f}"),
        #     ("cart",        cart_obj,        f"{cart_x}  {cart_y}  {table_surface_z:.4f}"),
        #     ("green_box",   green_box_obj,   f"{cart_x}  {cart_y}  {table_surface_z + 0.041:.4f}"),
        #     ("blue_box",    blue_box_obj,    f"-0.5   0.10  {table_surface_z + 0.025:.4f}"),
        #     ("orange_ball", orange_ball_obj, f"-0.08   0.4  {table_surface_z + 0.025:.4f}"),
        # ]

        # Fixed snapshot pose (episode 20/20, seed 980242470)
        objects = [
            ("toybot",      toybot_obj,      f"-0.101  0.231 {table_top_z:.4f}"),
            ("cart",        cart_obj,        f"-0.424  0.418  {table_surface_z:.4f}"),
            ("green_box",   green_box_obj,   f"-0.403  0.387  {table_surface_z + 0.041:.4f}"),
            ("blue_box",    blue_box_obj,    f"-0.474  0.097  {table_surface_z + 0.025:.4f}"),
            ("orange_ball", orange_ball_obj, f"-0.119  0.369  {table_surface_z + 0.025:.4f}"),
        ]

        self.object_joints = {}

        for obj_name, obj, position in objects:
            obj_xml = obj.get_obj()
            obj_xml.set("pos", position)
            if obj_name in _object_quats:
                obj_xml.set("quat", _object_quats[obj_name])

            pos_array = np.array([float(x) for x in position.split()])
            self.object_base_positions[obj_name] = pos_array

            joint_name = None
            joints = getattr(obj, "joints", None)
            if joints:
                if isinstance(joints[0], str):
                    joint_name = joints[0]
                else:
                    joint_names = getattr(obj, "joint_names", None)
                    if joint_names:
                        joint_name = joint_names[0]

            setattr(self, f"{obj_name}_joint", joint_name)
            self.object_joints[obj_name] = joint_name

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
                body_elements.set("pos", f"-0.78 0.0 {self.robot_base_z}")

        mujoco_objects = [obj for _, obj, _ in objects]
        self.model = Task(
            mujoco_arena=arena,
            mujoco_robots=self.robots[0].robot_model,
            mujoco_objects=mujoco_objects,
        )


register_env(ToyRed)

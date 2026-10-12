#!/usr/bin/env python3
"""
GrabBlock Environment — red block on a visual-only cart.
Block (toybot) is the PatternMover target. The cart (highcart) is a separate
visual body that tracks the block's XY each step until the block is lifted.
"""

import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_robosuite_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _robosuite_dir)

import numpy as np
from robosuite.environments.base import register_env
from robosuite.models.tasks import Task
from robosuite.models.objects.primitive import BoxObject
from xml_objects import BigCartObject, HighBlockRedObject, HighBlockCartObject
from environment.ToyRobotEnv import ToyRobot


class GrabBlock(ToyRobot):
    """Red block on a moving visual cart. Cart stays on table when block is lifted."""

    INIT_QPOS = np.array([-0.007, -0.283, -0.05, -1.86, -0.012, 1.821, 0.755])
    OBJECT_POS_NOISE_XY = 0.03
    ROBOT_QPOS_NOISE = 0.03

    BLOCK_LIFT_THRESHOLD = 0.03

    def __init__(self, **kwargs):
        self._block_initial_z = None
        self._block_ever_lifted = False
        super().__init__(**kwargs)

    def _load_model(self):
        from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
        from robosuite.models.arenas import TableArena
        from robosuite.utils.mjcf_utils import find_elements, new_element

        ManipulationEnv._load_model(self)

        self.table_offset = np.array((0.0, 0, 0.757))
        self.robot_base_z = 0.875
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

        table_surface_z = self.table_offset[2]  # 0.757
        table_top_z = table_surface_z + self.table_size[2] / 2  # 0.782

        block_x, block_y = -0.0801, 0.231

        toybot_obj = HighBlockRedObject(name="toybot")
        highcart_obj = HighBlockCartObject(name="highcart")

        distractor_cart_obj = BigCartObject(name="cart")
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
        _rot30z_ccw = f"{np.cos(np.pi/12):.6f} 0 0 {-np.sin(np.pi/12):.6f}"
        _object_quats = {"cart": _rot30z_ccw, "green_box": _rot30z_ccw}

        objects = [
            ("toybot",    toybot_obj,           f"{block_x}  {block_y}  {table_top_z:.4f}"),
            ("highcart",  highcart_obj,          f"{block_x}  {block_y}  {table_top_z:.4f}"),
            ("cart",      distractor_cart_obj,   f"-0.424  0.418  {table_surface_z:.4f}"),
            ("green_box", green_box_obj,         f"-0.403  0.387  {table_surface_z + 0.041:.4f}"),
            ("blue_box",  blue_box_obj,          f"-0.474  0.097  {table_surface_z + 0.025:.4f}"),
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

    def _reset_internal(self):
        super()._reset_internal()

        block_joint = self.object_joints.get("toybot")
        cart_joint = self.object_joints.get("highcart")

        if block_joint and cart_joint:
            try:
                block_qpos = self.sim.data.get_joint_qpos(block_joint)
                cart_qpos = self.sim.data.get_joint_qpos(cart_joint).copy()
                cart_qpos[0] = block_qpos[0]
                cart_qpos[1] = block_qpos[1]
                self.sim.data.set_joint_qpos(cart_joint, cart_qpos)
                self.sim.forward()
            except Exception as e:
                print(f"[WARN] Could not snap cart to block: {e}")

        self._block_ever_lifted = False
        try:
            self._block_initial_z = self.sim.data.get_body_xpos("toybot_main")[2]
            self.object_initial_heights["toybot_main"] = self._block_initial_z
        except Exception:
            self._block_initial_z = None

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        self._sync_cart_to_block()

    def _sync_cart_to_block(self):
        """Match cart XY to block XY each step until block is lifted."""
        block_joint = self.object_joints.get("toybot")
        cart_joint = self.object_joints.get("highcart")
        if not block_joint or not cart_joint:
            return

        if not self._block_ever_lifted:
            try:
                block_z = self.sim.data.get_body_xpos("toybot_main")[2]
                if self._block_initial_z is not None and block_z > self._block_initial_z + self.BLOCK_LIFT_THRESHOLD:
                    self._block_ever_lifted = True
            except Exception:
                pass

        if self._block_ever_lifted:
            return

        try:
            block_qpos = self.sim.data.get_joint_qpos(block_joint)
            cart_qpos = self.sim.data.get_joint_qpos(cart_joint).copy()
            cart_qpos[0] = block_qpos[0]
            cart_qpos[1] = block_qpos[1]
            self.sim.data.set_joint_qpos(cart_joint, cart_qpos)
        except Exception:
            pass


register_env(GrabBlock)

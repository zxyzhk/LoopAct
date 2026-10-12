#!/usr/bin/env python3
"""
BlockCart Environment — ToyRed geometry with a blue block on a moving cart.

The cart (ToyCartObject) is the PatternMover target. The blue block is a
separate free body placed on top of the cart. _sync_block_to_cart copies
the cart's XY displacement to the block every step until the robot lifts it.
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
from xml_objects import ToyCartObject
from environment.ToyRobotEnv import ToyRobot


class BlockCart(ToyRobot):
    """ToyRed-equivalent env: moving cart carrying a blue block."""

    INIT_QPOS = np.array([-0.007, -0.283, -0.05, -1.86, -0.012, 1.821, 0.755])
    OBJECT_POS_NOISE_XY = 0.01
    ROBOT_QPOS_NOISE = 0.01

    # cart platform top at local z=0.016; block half-height=0.0935
    BLOCK_OFFSET_Z = 0.1095
    BLOCK_LIFT_THRESHOLD = 0.03

    def __init__(self, **kwargs):
        self._last_cart_xy = None
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
            has_legs=True,
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

        table_surface_z = self.table_offset[2]                       # 0.757
        table_top_z     = table_surface_z + self.table_size[2] / 2  # 0.782

        cart_x, cart_y = -0.101, 0.231

        cart_obj = ToyCartObject(name="toycart")

        blue_block_obj = BoxObject(
            name="blue_block",
            size=[0.05, 0.02, 0.0935],
            rgba=[0.15, 0.30, 0.90, 1.0],
            density=500,
            friction=(2.0, 0.01, 0.001),
        )
        green_box_obj = BoxObject(
            name="green_box",
            size=[0.025, 0.025, 0.025],
            rgba=[0.15, 0.75, 0.20, 1.0],
            density=500,
            friction=(1.5, 0.01, 0.001),
        )
        pink_box_obj = BoxObject(
            name="pink_box",
            size=[0.025, 0.025, 0.025],
            rgba=[0.95, 0.20, 0.60, 1.0],
            density=500,
            friction=(1.5, 0.01, 0.001),
        )
        orange_box_obj = BoxObject(
            name="orange_box",
            size=[0.025, 0.025, 0.025],
            rgba=[0.95, 0.50, 0.05, 1.0],
            density=500,
            friction=(1.5, 0.01, 0.001),
        )

        objects = [
            ("toycart",    cart_obj,       f"{cart_x}  {cart_y} {table_surface_z:.4f}"),
            ("blue_block", blue_block_obj, f"{cart_x}  {cart_y} {table_surface_z + self.BLOCK_OFFSET_Z:.4f}"),
            ("green_box",  green_box_obj,  f"-0.403  0.387 {table_top_z:.4f}"),
            ("pink_box",   pink_box_obj,   f"-0.474  0.097 {table_top_z:.4f}"),
            ("orange_box", orange_box_obj, f"-0.119  0.369 {table_top_z:.4f}"),
        ]

        self.object_joints = {}

        for obj_name, obj, position in objects:
            obj_xml = obj.get_obj()
            obj_xml.set("pos", position)

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
                return_first=True,
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

        # After super() places objects (with independent XY noise), reposition
        # the block to sit exactly on top of the cart.
        cart_joint = self.object_joints.get("toycart")
        block_joint = self.object_joints.get("blue_block")

        if cart_joint and block_joint:
            try:
                cart_qpos = self.sim.data.get_joint_qpos(cart_joint)
                block_pos = cart_qpos[:3].copy()
                block_pos[2] = cart_qpos[2] + self.BLOCK_OFFSET_Z
                self.sim.data.set_joint_qpos(
                    block_joint,
                    np.concatenate([block_pos, [1.0, 0.0, 0.0, 0.0]]),
                )
                self.sim.forward()
            except Exception as e:
                print(f"[WARN] Could not place block on cart: {e}")

        # Init block-follows-cart tracking
        self._block_ever_lifted = False
        if cart_joint:
            try:
                self._last_cart_xy = self.sim.data.get_joint_qpos(cart_joint)[:2].copy()
            except Exception:
                self._last_cart_xy = None

        try:
            self._block_initial_z = self.sim.data.get_body_xpos("blue_block_main")[2]
            self.object_initial_heights["blue_block_main"] = self._block_initial_z
        except Exception:
            self._block_initial_z = None

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        self._sync_block_to_cart()

    def _sync_block_to_cart(self):
        """Copy cart XY delta to block each step until block is lifted."""
        cart_joint = self.object_joints.get("toycart")
        block_joint = self.object_joints.get("blue_block")
        if not cart_joint or not block_joint or self._last_cart_xy is None:
            return

        if not self._block_ever_lifted:
            try:
                block_z = self.sim.data.get_body_xpos("blue_block_main")[2]
                if self._block_initial_z is not None and block_z > self._block_initial_z + self.BLOCK_LIFT_THRESHOLD:
                    self._block_ever_lifted = True
            except Exception:
                pass

        if self._block_ever_lifted:
            return

        cart_qpos = self.sim.data.get_joint_qpos(cart_joint)
        curr_cart_xy = cart_qpos[:2].copy()
        delta = curr_cart_xy - self._last_cart_xy
        self._last_cart_xy = curr_cart_xy

        if np.any(np.abs(delta) > 1e-9):
            block_qpos = self.sim.data.get_joint_qpos(block_joint).copy()
            block_qpos[0] += delta[0]
            block_qpos[1] += delta[1]
            self.sim.data.set_joint_qpos(block_joint, block_qpos)


register_env(BlockCart)


if __name__ == "__main__":
    from tasks.block_cart import BLOCK_CART_CONFIG, BLOCK_CART_SUCCESS
    from env_render import EnvRender

    cfg = BLOCK_CART_CONFIG

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
        target_joint_name=cfg.target_joint_name,
        mover_type=cfg.mover_type,
        speeds=cfg.speeds,
        decision_interval=cfg.decision_interval,
        task_name=cfg.task_name,
        task_success=BLOCK_CART_SUCCESS,
    )

    runner.run()

#!/usr/bin/env python3
"""
ToyCart Environment - Table with a moving flat cart.

Same robot and table setup as GreenBlockEnv, but with a flat cart
built from MuJoCo primitives sitting on the table.
"""

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import Task

import sys
import os
if __name__ == "__main__":
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _robosuite_dir = os.path.dirname(_current_dir)
    sys.path.insert(0, _robosuite_dir)

from robosuite.environments.base import register_env
from robosuite.utils.mjcf_utils import find_elements, new_element
from robosuite.models.objects.primitive import BoxObject
from dataclasses import dataclass, field
from typing import List, Optional
from env_render import EnvRender
from tasks.base import TaskSuccess
from xml_objects import ToyCartObject


class ToyCart(ManipulationEnv):
    """Environment with a table, a flat cart that moves, and a block on its bed."""

    # Default Panda pose
    INIT_QPOS = np.array([0.0, -0.3, 0.0, -1.9, 0.0, 1.8, 0.785])

    # Randomization ranges
    OBJECT_POS_NOISE_XY = 0.05
    ROBOT_QPOS_NOISE = 0.05

    # Block offset from cart origin (cart is a flat slab, no rotation needed):
    #   cart platform top at local z = 0.016 (centre 0.008 + half-height 0.008)
    #   block half-height = 0.030
    BLOCK_OFFSET_X = 0.0
    BLOCK_OFFSET_Y = 0.0
    BLOCK_OFFSET_Z = 0.046  # cart_top (0.016) + block_half_h (0.030)

    # Threshold above initial z to consider block "lifted" (stop syncing)
    BLOCK_LIFT_THRESHOLD = 0.03

    def __init__(
        self,
        randomize_init: bool = True,
        task_name: Optional[str] = None,
        task_success: Optional[TaskSuccess] = None,
        **kwargs
    ):
        self.randomize_init = randomize_init
        self.object_base_positions = {}
        self.object_base_rotations = {}

        self.task_name = task_name or "pick_truck"
        self.task_success = task_success
        self.current_success = False
        self.success_checked = False

        # Block-follows-truck state
        self._last_truck_xy = None
        self._block_initial_z = None
        self._block_ever_lifted = False  # one-way latch: never re-sync after pickup

        super().__init__(**kwargs)

        if self.task_success:
            print(f"\n[TASK] Loaded: {self.task_success.task_description}")

    def _reset_internal(self):
        """Reset with randomized truck position; block placed on truck bed."""
        super()._reset_internal()

        if not hasattr(self, 'object_initial_heights'):
            self.object_initial_heights = {}

        # Override agentview camera
        if hasattr(self.sim.model, 'camera_names'):
            for cam_name in self.sim.model.camera_names:
                if cam_name == 'agentview':
                    cam_id = self.sim.model.camera_name2id(cam_name)
                    self.sim.model.cam_pos[cam_id] = np.array([1.1536, -0.0076, 1.7745])
                    self.sim.model.cam_quat[cam_id] = np.array([0.6260, 0.3337, 0.3316, 0.6219])
                    self.sim.model.cam_fovy[cam_id] = 50.0
                    break

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

        # --- Truck position (with randomization) ---
        truck_joint = self.object_joints.get("truck")
        truck_base = self.object_base_positions["truck"]
        truck_quat = self.object_base_rotations.get("truck", np.array([1.0, 0.0, 0.0, 0.0]))

        if self.randomize_init and truck_joint:
            noise_x = np.random.uniform(-self.OBJECT_POS_NOISE_XY, self.OBJECT_POS_NOISE_XY)
            noise_y = np.random.uniform(-self.OBJECT_POS_NOISE_XY, self.OBJECT_POS_NOISE_XY)
            truck_pos = truck_base.copy()
            truck_pos[0] += noise_x
            truck_pos[1] += noise_y
        else:
            truck_pos = truck_base.copy()

        if truck_joint:
            try:
                self.sim.data.set_joint_qpos(truck_joint, np.concatenate([truck_pos, truck_quat]))
            except Exception as e:
                print(f"[DEBUG] Could not set truck joint: {e}")

        print(f"[INIT] Truck position: [{truck_pos[0]:+.3f}, {truck_pos[1]:+.3f}, {truck_pos[2]:.3f}]")

        # --- Block position: computed from truck position (no independent noise) ---
        block_joint = self.object_joints.get("block")
        block_pos = np.array([
            truck_pos[0] + self.BLOCK_OFFSET_X,
            truck_pos[1] + self.BLOCK_OFFSET_Y,
            truck_pos[2] + self.BLOCK_OFFSET_Z,
        ])
        block_quat = np.array([1.0, 0.0, 0.0, 0.0])

        if block_joint:
            try:
                self.sim.data.set_joint_qpos(block_joint, np.concatenate([block_pos, block_quat]))
            except Exception as e:
                print(f"[DEBUG] Could not set block joint: {e}")

        print(f"[INIT] Block position: [{block_pos[0]:+.3f}, {block_pos[1]:+.3f}, {block_pos[2]:.3f}]")

        self.sim.forward()

        # Initialize block-follows-truck tracking
        self._block_ever_lifted = False
        if truck_joint:
            self._last_truck_xy = self.sim.data.get_joint_qpos(truck_joint)[:2].copy()
        else:
            self._last_truck_xy = truck_pos[:2].copy()

        # Store initial heights
        self.object_initial_heights = {}
        try:
            self.object_initial_heights["truck_main"] = self.sim.data.get_body_xpos("truck_main")[2]
            self.object_initial_heights["block_main"] = self.sim.data.get_body_xpos("block_main")[2]
            self._block_initial_z = self.object_initial_heights["block_main"]
            print(f"[INIT] Initial block height: {self._block_initial_z:.3f}m")
        except Exception as e:
            print(f"[WARN] Could not store initial heights: {e}")
            self._block_initial_z = block_pos[2]

    def _pre_action(self, action, policy_step=False):
        """Sync block position with truck movement before each sim step."""
        super()._pre_action(action, policy_step)
        self._sync_block_to_truck()

    def _sync_block_to_truck(self):
        """Apply the truck's XY displacement to the block (until block is lifted)."""
        truck_joint = self.object_joints.get("truck")
        block_joint = self.object_joints.get("block")
        if not truck_joint or not block_joint or self._last_truck_xy is None:
            return

        # One-way latch: once lifted, never sync again (even if dropped back)
        if not self._block_ever_lifted:
            try:
                block_z = self.sim.data.get_body_xpos("block_main")[2]
                if self._block_initial_z is not None and block_z > self._block_initial_z + self.BLOCK_LIFT_THRESHOLD:
                    self._block_ever_lifted = True
            except Exception:
                pass

        if self._block_ever_lifted:
            return

        truck_qpos = self.sim.data.get_joint_qpos(truck_joint)
        curr_truck_xy = truck_qpos[:2].copy()
        delta = curr_truck_xy - self._last_truck_xy
        self._last_truck_xy = curr_truck_xy

        if np.any(np.abs(delta) > 1e-9):
            block_qpos = self.sim.data.get_joint_qpos(block_joint).copy()
            block_qpos[0] += delta[0]
            block_qpos[1] += delta[1]
            self.sim.data.set_joint_qpos(block_joint, block_qpos)

    def _load_model(self):
        super()._load_model()

        # Arena setup (same as GreenBlock)
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

        # Create the toy truck
        table_top_z = self.table_offset[2]

        truck_obj = ToyCartObject(name="truck")

        # Block on the cart platform.
        # Cart origin at (cart_x, cart_y, table_top_z).
        # Cart platform top at local z=0.016; block centre at z=0.016+0.030=0.046.
        truck_base_x = -0.12
        truck_base_y = 0.2
        truck_base_z = table_top_z  # cart sits flush on table

        block_x = truck_base_x + self.BLOCK_OFFSET_X
        block_y = truck_base_y + self.BLOCK_OFFSET_Y
        block_z = truck_base_z + self.BLOCK_OFFSET_Z

        block_obj = BoxObject(
            name="block",
            size=[0.05, 0.02, 0.03],
            rgba=[0.8, 0.2, 0.2, 1.0],
            density=500,
            friction=(2.0, 0.01, 0.001),
        )

        objects = [
            ("truck", truck_obj, f"{truck_base_x} {truck_base_y} {truck_base_z}"),
            ("block", block_obj, f"{block_x} {block_y} {block_z}"),
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
            print(f"[WARNING] No task success checker loaded for '{self.task_name}'")
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
class ToyCartConfig:
    env_name: str = "ToyCart"
    robot: str = "Panda"
    controller: str = "BASIC"
    control_freq: int = 10
    teleoperate_method: str = "spacemouse"
    target_joint_name: str = "truck"
    target_speed_y: float = 0.05
    target_axis: str = "x"
    render_camera: str = "custom_view"
    camera_names: List[str] = field(default_factory=lambda: ["robot0_eye_in_hand", "agentview"])
    show_camera_obs: bool = True
    free_view_mode: bool = False
    randomize_init: bool = True
    save_data: bool = True
    task_name: str = "pick_truck"


TOY_CART_CONFIG = ToyCartConfig()

register_env(ToyCart)


if __name__ == "__main__":
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
        record_path="demo.hdf5",
        target_joint_name=cfg.target_joint_name,
        target_speed_y=cfg.target_speed_y,
        target_axis=cfg.target_axis,
    )

    runner.run()

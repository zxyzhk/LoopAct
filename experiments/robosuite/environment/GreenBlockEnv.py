#!/usr/bin/env python3
"""
GreenBlock Environment - Simple table with a single moving block.

The block moves randomly (stop/go behavior every second) to test
visual feedback capabilities of the model.
"""

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import Task
from robosuite.models.objects.primitive import BoxObject

import sys
import os
if __name__ == "__main__":
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _robosuite_dir = os.path.dirname(_current_dir)
    sys.path.insert(0, _robosuite_dir)

from robosuite.environments.base import register_env
from robosuite.utils.mjcf_utils import find_elements, new_element
from dataclasses import dataclass, field
from typing import List, Optional
from env_render import EnvRender
from tasks.base import TaskSuccess


class GreenBlock(ManipulationEnv):
    """Environment with a table and a single block that can move randomly."""

    # Default Panda pose
    INIT_QPOS = np.array([0.0, -0.3, 0.0, -1.9, 0.0, 1.8, 0.785])

    # Randomization ranges
    OBJECT_POS_NOISE_XY = 0.05  # +/-5cm in x and y
    ROBOT_QPOS_NOISE = 0.05

    def __init__(
        self,
        randomize_init: bool = True,
        task_name: Optional[str] = None,
        task_success: Optional[TaskSuccess] = None,
        **kwargs
    ):
        # Initialize randomization attributes BEFORE super().__init__()
        self.randomize_init = randomize_init
        self.object_base_positions = {}
        self.object_base_rotations = {}

        # Task setup
        self.task_name = task_name or "pick_block"
        self.task_success = task_success
        self.current_success = False
        self.success_checked = False

        super().__init__(**kwargs)

        if self.task_success:
            print(f"\n[TASK] Loaded: {self.task_success.task_description}")

    def _reset_internal(self):
        """Reset with randomized initial positions."""
        super()._reset_internal()

        # Initialize storage for initial object heights
        if not hasattr(self, 'object_initial_heights'):
            self.object_initial_heights = {}

        # Override agentview camera with custom viewer position
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

        # Randomize object positions
        object_positions = {}
        if self.randomize_init:
            for obj_name, base_pos in self.object_base_positions.items():
                joint_name = self.object_joints.get(obj_name)
                if joint_name:
                    try:
                        noise_x = np.random.uniform(-self.OBJECT_POS_NOISE_XY, self.OBJECT_POS_NOISE_XY)
                        noise_y = np.random.uniform(-self.OBJECT_POS_NOISE_XY, self.OBJECT_POS_NOISE_XY)
                        new_pos = base_pos.copy()
                        new_pos[0] += noise_x
                        new_pos[1] += noise_y

                        self.sim.data.set_joint_qpos(joint_name, new_pos)
                        object_positions[obj_name] = new_pos
                    except Exception as e:
                        print(f"[DEBUG] Could not set joint position for {obj_name}: {e}")
                        object_positions[obj_name] = new_pos
                else:
                    object_positions[obj_name] = base_pos
        else:
            for obj_name, base_pos in self.object_base_positions.items():
                object_positions[obj_name] = base_pos

        if object_positions:
            print("[INIT] Object positions:")
            for obj_name, pos in object_positions.items():
                print(f"  {obj_name:8s}: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f}]")

        self.sim.forward()

        # Store initial heights for success checking
        self.object_initial_heights = {}
        try:
            self.object_initial_heights["block_main"] = self.sim.data.get_body_xpos("block_main")[2]
            print(f"[INIT] Initial block height: {self.object_initial_heights['block_main']:.3f}m")
        except Exception as e:
            print(f"[WARN] Could not store initial heights: {e}")

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

        # Create a simple colored block
        table_top_z = self.table_offset[2]

        block_obj = BoxObject(
            name="block",
            size=[0.05, 0.02, 0.03],  # 6cm cube
            # rgba = [0.8, 0.2, 0.2, 1.0], # Red color
            rgba=[0.2, 0.8, 0.2, 1.0],  # Green color
            density=500,
            friction=(2.0, 0.01, 0.001),  # Higher friction for better grip
        )

        # Objects list
        objects = [
            ("block", block_obj, f"-0.1 0.2 {table_top_z + 0.03}"),
            # ("block", block_obj, f"-0.2 0.2 {table_top_z + 0.03}"),
        ]

        # Dictionary to store object name -> joint name mapping
        self.object_joints = {}

        for obj_name, obj, position in objects:
            obj_xml = obj.get_obj()
            obj_xml.set("pos", position)

            # Store base position for randomization
            pos_array = np.array([float(x) for x in position.split()])
            self.object_base_positions[obj_name] = pos_array

            # Store joint for this object
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
class GreenBlockConfig:
    env_name: str = "GreenBlock"
    robot: str = "Panda"
    controller: str = "BASIC"
    control_freq: int = 10
    teleoperate_method: str = "spacemouse"
    target_joint_name: str = "block"  # The block will move
    target_speed_y: float = 0.05  # Move to the right (positive x)
    target_axis: str = "x"
    render_camera: str = "custom_view"
    camera_names: List[str] = field(default_factory=lambda: ["robot0_eye_in_hand", "agentview"])
    show_camera_obs: bool = True
    free_view_mode: bool = False
    randomize_init: bool = True
    save_data: bool = True
    task_name: str = "pick_block"


ONE_BLOCK_CONFIG = GreenBlockConfig()

register_env(GreenBlock)


if __name__ == "__main__":
    cfg = ONE_BLOCK_CONFIG

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

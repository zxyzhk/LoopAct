#!/usr/bin/env python3
"""
Custom Environment for SpaceMouse Teleoperation
Uses TeleopWrapper to integrate MouseTeleoperate and TargetMover tools
"""

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import Task
from robosuite.models.objects.primitive import BallObject, BoxObject

# Use absolute imports that work when robosuite/ is in sys.path
import sys
import os
if __name__ == "__main__":
    # Add parent directory to path when running directly
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _robosuite_dir = os.path.dirname(_current_dir)
    sys.path.insert(0, _robosuite_dir)

from xml_objects import PlateObject, PotObject, LadleObject
from robosuite.environments.base import register_env
from robosuite.utils.mjcf_utils import find_elements, new_element
from dataclasses import dataclass, field
from typing import List, Optional
# Import tools
from env_render import EnvRender
from tasks.base import TaskSuccess


class TableLadle(ManipulationEnv):
    """Custom environment with a table and sphere object"""
    
    # Default Panda pose is roughly: [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    INIT_QPOS = np.array([0.0, -0.3, 0.0, -1.9, 0.0, 1.8, 0.785])
    
    # Randomization ranges for LIBERO-style initialization (in meters/radians)
    OBJECT_POS_NOISE_XY = 0.05  # ±5cm in x and y
    OBJECT_ROT_NOISE_Z = 0.3    # ±~17 degrees in z-axis rotation
    ROBOT_QPOS_NOISE = 0.05     # Small noise to robot joints
    
    def __init__(self, randomize_init=True, task_name: Optional[str] = None, 
                 task_success: Optional[TaskSuccess] = None, **kwargs):
        # Initialize randomization attributes BEFORE super().__init__()
        # because super().__init__() calls _load_model() which uses these
        self.randomize_init = randomize_init
        self.object_base_positions = {}
        self.object_base_rotations = {}
        
        # Task setup
        self.task_name = task_name or "pick_up_ladle"  # Default task
        self.task_success = task_success  # TaskSuccess instance
        self.current_success = False
        self.success_checked = False
        
        super().__init__(**kwargs)
        
        # Print task info
        if self.task_success:
            print(f"\n[TASK] Loaded: {self.task_success.task_description}")
    
    def _reset_internal(self):
        """Reset with randomized initial positions (LIBERO-style)"""
        super()._reset_internal()
        
        # Initialize storage for initial object heights (for success checking)
        if not hasattr(self, 'object_initial_heights'):
            self.object_initial_heights = {}
        
        # Override agentview camera with custom viewer position
        if hasattr(self.sim.model, 'camera_names'):
            for cam_name in self.sim.model.camera_names:
                if cam_name == 'agentview':
                    cam_id = self.sim.model.camera_name2id(cam_name)
                    # Set to custom viewer camera position
                    self.sim.model.cam_pos[cam_id] = np.array([1.1536, -0.0076, 1.7745])
                    self.sim.model.cam_quat[cam_id] = np.array([0.6260, 0.3337, 0.3316, 0.6219])
                    self.sim.model.cam_fovy[cam_id] = 50.0
                    
                    print(f"\n=== AgentView Camera (Custom View) ===")
                    print(f"Position: [{self.sim.model.cam_pos[cam_id][0]:+.6f}, {self.sim.model.cam_pos[cam_id][1]:+.6f}, {self.sim.model.cam_pos[cam_id][2]:+.6f}]")
                    print(f"Quaternion: [{self.sim.model.cam_quat[cam_id][0]:+.6f}, {self.sim.model.cam_quat[cam_id][1]:+.6f}, {self.sim.model.cam_quat[cam_id][2]:+.6f}, {self.sim.model.cam_quat[cam_id][3]:+.6f}]")
                    print(f"FOV Y: {self.sim.model.cam_fovy[cam_id]:.2f} degrees\n")
                    break
        
        # Set initial robot joint positions (with optional randomization)
        robot = self.robots[0]
        robot_qpos = []
        if hasattr(robot, 'init_qpos') and self.INIT_QPOS is not None:
            robot_qpos_indices = robot._ref_joint_pos_indexes
            
            for i, idx in enumerate(robot_qpos_indices):
                if i < len(self.INIT_QPOS):
                    if self.randomize_init:
                        # Add small random noise to each joint
                        noise = np.random.uniform(-self.ROBOT_QPOS_NOISE, self.ROBOT_QPOS_NOISE)
                        qpos = self.INIT_QPOS[i] + noise
                        self.sim.data.qpos[idx] = qpos
                        robot_qpos.append(qpos)
                    else:
                        self.sim.data.qpos[idx] = self.INIT_QPOS[i]
                        robot_qpos.append(self.INIT_QPOS[i])
        
        # Print robot initial joint positions
        if robot_qpos:
            print(f"[INIT] Robot joints: {np.array2string(np.array(robot_qpos), precision=3, suppress_small=True)}")
        
        # Randomize object positions (if enabled)
        object_positions = {}
        if self.randomize_init:
            for obj_name, base_pos in self.object_base_positions.items():
                joint_name = self.object_joints.get(obj_name)
                if joint_name:
                    try:
                        # Add random noise to x, y position (keep z the same)
                        noise_x = np.random.uniform(-self.OBJECT_POS_NOISE_XY, self.OBJECT_POS_NOISE_XY)
                        noise_y = np.random.uniform(-self.OBJECT_POS_NOISE_XY, self.OBJECT_POS_NOISE_XY)
                        new_pos = base_pos.copy()
                        new_pos[0] += noise_x
                        new_pos[1] += noise_y
                        
                        # Set position
                        self.sim.data.set_joint_qpos(joint_name, new_pos)
                        object_positions[obj_name] = new_pos
                    except Exception as e:
                        # If set_joint_qpos fails, just store the randomized position for display
                        print(f"[DEBUG] Could not set joint position for {obj_name}: {e}")
                        object_positions[obj_name] = new_pos
                else:
                    # No joint name, just show base position
                    object_positions[obj_name] = base_pos
        else:
            # Store base positions for printing even when not randomizing
            for obj_name, base_pos in self.object_base_positions.items():
                object_positions[obj_name] = base_pos
        
        # Print object positions
        if object_positions:
            print("[INIT] Object positions:")
            for obj_name, pos in object_positions.items():
                print(f"  {obj_name:8s}: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f}]")
        else:
            print("[DEBUG] No object positions to print (object_base_positions empty)")
        
        # Forward the simulation to update all changes
        self.sim.forward()
        
        # Store initial heights of objects (after forward) for success checking
        self.object_initial_heights = {}
        try:
            self.object_initial_heights["ladle_main"] = self.sim.data.get_body_xpos("ladle_main")[2]
            self.object_initial_heights["plate_main"] = self.sim.data.get_body_xpos("plate_main")[2]
            self.object_initial_heights["pot_main"] = self.sim.data.get_body_xpos("pot_main")[2]
            print(f"[INIT] Initial object heights stored:")
            print(f"  ladle: {self.object_initial_heights['ladle_main']:.3f}m")
            print(f"  plate: {self.object_initial_heights['plate_main']:.3f}m")
            print(f"  pot: {self.object_initial_heights['pot_main']:.3f}m")
        except Exception as e:
            print(f"[WARN] Could not store initial heights: {e}")

    def _load_model(self):
        super()._load_model()

        # 1) Arena (use TableArena with larger table)
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
                fovy="50",  # Increased from 40 to zoom out
            )
        )

        # 2) Objects as MujocoObjects
        # Mix of primitive objects and XML objects from models/objects (e.g., dinnerware)
        pot_obj = PotObject(name="pot")
        plate_obj = PlateObject(name="plate")
        ladle_obj = LadleObject(name="ladle")
        
        # Note: XML objects from xml_objects.py have naming prefix conflicts
        # To use XML objects, they need to be fixed or used with different names
        # For now, using primitive objects that work correctly
        
        table_top_z = self.table_offset[2]  # 0.7
        
        # Store all objects in a list for processing: (name, object, position)
        objects = [

            ("plate", plate_obj, f"0.2 0.0 {table_top_z + 0.02}"),
            ("ladle", ladle_obj, f"-0.1 0.3 {table_top_z + 0.012}"),
            ("pot", pot_obj, f"0.0 -0.4 {table_top_z + 0.012}"),
        ]
        
        # Dictionary to store object name -> joint name mapping
        self.object_joints = {}
        
        # Process each object: set position and store joint
        for obj_name, obj, position in objects:
            obj_xml = obj.get_obj()
            obj_xml.set("pos", position)
            
            # Store base position for randomization
            pos_array = np.array([float(x) for x in position.split()])
            self.object_base_positions[obj_name] = pos_array
            
            if obj_name == "ladle":
                obj_xml.set("euler", "0 1 0")
                self.object_base_rotations[obj_name] = np.array([0, 1, 0])
            
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
            
            # Create attribute: self.sphere_joint, self.square_joint, etc.
            setattr(self, f"{obj_name}_joint", joint_name)
            
            # Store in mapping for easy lookup (simple names only: "sphere", "square")
            self.object_joints[obj_name] = joint_name

        # 3) Set robot base position
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
                # Move robot closer to table in X direction (table is at x=1.0, so 0.4 brings it closer)
                body_elements.set("pos", f"-0.7 0.0 {table_top_z}")

        # 4) Build Task
        # Extract objects from the list for Task
        mujoco_objects = [obj for _, obj, _ in objects]
        self.model = Task(
            mujoco_arena=arena,
            mujoco_robots=self.robots[0].robot_model,
            mujoco_objects=mujoco_objects,
        )

    def reward(self, action):
        return 0.0
    
    def check_success(self, verbose: bool = False):
        """
        Check if the current task has been completed successfully.
        
        Args:
            verbose: If True, print detailed success information
            
        Returns:
            (success, score, details): Success status, score (0-1), and detailed results
        """
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
        # Task ID can be extracted from task_name or set explicitly
        return 0

@dataclass
class TableLadleConfig:
    env_name: str = "TableLadle"
    robot: str = "Panda"
    controller: str = "BASIC"
    control_freq: int = 10  # Match LIBERO dataset (10 Hz)
    teleoperate_method: str = "spacemouse"
    target_joint_name: str = "none"
    target_speed_y: float = -0.005
    target_axis: str = "y"
    render_camera: str = "custom_view"
    camera_names: List[str] = field(default_factory=lambda: ["robot0_eye_in_hand", "agentview"])
    show_camera_obs: bool = True
    free_view_mode: bool = False
    randomize_init: bool = True  # Enable/disable randomization
    save_data: bool = True  # Enable/disable data saving
    task_name: str = "pick_up_ladle"  # Task name for success checking

TABLE_LADLE_CONFIG = TableLadleConfig()

register_env(TableLadle)


if __name__ == "__main__":
    # Create and run teleoperation using the wrapper
    # The wrapper integrates MouseTeleoperate (robot arm control) and TargetMover (object movement)
    
    # Use config for all settings
    cfg = TABLE_LADLE_CONFIG
    
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

    # Run teleoperation
    runner.run()

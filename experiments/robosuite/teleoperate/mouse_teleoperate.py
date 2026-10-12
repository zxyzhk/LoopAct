#!/usr/bin/env python3
"""
MouseTeleoperate - Tool for SpaceMouse teleoperation in robosuite
"""

import numpy as np
import robosuite as suite
import time
from pynput import keyboard

from robosuite import load_composite_controller_config
from robosuite.devices.spacemouse import SpaceMouse


class MouseTeleoperate:
    """SpaceMouse teleoperation class for Robosuite - controls robot arm location"""
    
    def __init__(
        self,
        env_name,
        robot="Panda",
        controller="BASIC",
        vendor_id=0x256f,
        product_id=0xc652,
        pos_sensitivity=50.0,
        rot_sensitivity=10.0,
        arm="right",
        control_freq=10,  # Match LIBERO dataset (10 Hz)
        horizon=float('inf'),
        render_camera=None,
        use_camera_obs=False,
        has_offscreen_renderer=False,
        camera_names=None,
        camera_heights=None,
        camera_widths=None,
        randomize_init=True,
        task_name=None,
        task_success=None,
        slope_angle=None,  # Slope angle for ToyCan environment (degrees)
    ):
        """
        Initialize SpaceMouse teleoperation
        
        Args:
            env_name: Name of the robosuite environment
            robot: Robot name (e.g., "Panda")
            controller: Controller type (e.g., "BASIC")
            vendor_id: SpaceMouse vendor ID
            product_id: SpaceMouse product ID
            pos_sensitivity: Position control sensitivity
            rot_sensitivity: Rotation control sensitivity
            arm: Which arm to control ("right" or "left")
            control_freq: Control frequency in Hz
            horizon: Episode horizon (use float('inf') for infinite)
        """
        self.env_name = env_name
        self.robot = robot
        self.controller = controller
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.arm = arm
        self.control_freq = control_freq
        self.horizon = horizon
        self.render_camera = render_camera
        self.use_camera_obs = use_camera_obs
        self.has_offscreen_renderer = has_offscreen_renderer
        self.camera_names = camera_names
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths
        self.randomize_init = randomize_init
        self.task_name = task_name
        self.task_success = task_success
        self.slope_angle = slope_angle
        
        # State variables
        self.env = None
        self.device = None
        self.pressed_keys = set()
        self.gripper_cmd = 0.0
        self.keyboard_listener = None
        self.action_dim = None
        
        print("=" * 60)
        print("SPACEMOUSE TELEOPERATION FOR ROBOSUITE")
        print("=" * 60)
        
    def setup_environment(self, env=None):
        """
        Create and configure the robosuite environment
        
        Args:
            env: Optional pre-created environment. If None, creates a new one.
        """
        if env is not None:
            self.env = env
            self.env.reset()
            self.action_dim = self.env.action_spec[0].shape[0]
            print(f"[INFO] Using provided environment. Action dimension: {self.action_dim}")
            return
        
        print("\n[INFO] Setting up environment...")
        
        # Load controller config
        controller_config = load_composite_controller_config(
            controller=self.controller,
            robot=self.robot
        )
        
        # Create environment
        env_kwargs = {
            "env_name": self.env_name,
            "robots": self.robot,
            "controller_configs": controller_config,
            "has_renderer": True,
            "has_offscreen_renderer": self.has_offscreen_renderer,
            "use_camera_obs": self.use_camera_obs,
            "control_freq": self.control_freq,
            "render_camera": self.render_camera,
            "horizon": self.horizon,
            "camera_names": self.camera_names,
            "camera_heights": self.camera_heights,
            "camera_widths": self.camera_widths,
            "randomize_init": self.randomize_init,
        }
        
        # Add task_name if provided (for custom environments that support it)
        if self.task_name is not None:
            env_kwargs["task_name"] = self.task_name
        
        # Add task_success if provided (for custom environments that support it)
        if self.task_success is not None:
            env_kwargs["task_success"] = self.task_success
        
        # Add slope_angle if provided (for ToyCan environment)
        if self.slope_angle is not None:
            env_kwargs["slope_angle"] = self.slope_angle
        
        self.env = suite.make(**env_kwargs)
        
        # Reset and get action dimension
        self.env.reset()
        self.action_dim = self.env.action_spec[0].shape[0]
        print(f"[INFO] Action dimension: {self.action_dim}")
        
        # Open gripper at start
        self._open_gripper_steps(steps=20)
        self.env.render()
        
    def setup_spacemouse(self):
        """Initialize the SpaceMouse device"""
        if self.env is None:
            raise RuntimeError("Environment must be set up before SpaceMouse")
            
        print("\n[INFO] Initializing SpaceMouse...")
        print(f"  Vendor ID:  {hex(self.vendor_id)} ({self.vendor_id})")
        print(f"  Product ID: {hex(self.product_id)} ({self.product_id})")
        print(f"  Position Sensitivity: {self.pos_sensitivity}")
        print(f"  Rotation Sensitivity: {self.rot_sensitivity}")
        
        try:
            self.device = SpaceMouse(
                env=self.env,
                vendor_id=self.vendor_id,
                product_id=self.product_id,
                pos_sensitivity=self.pos_sensitivity,
                rot_sensitivity=self.rot_sensitivity,
            )
            print("\n[OK] SpaceMouse device created successfully!")
        except Exception as e:
            print(f"\n[ERROR] Failed to create SpaceMouse device: {e}")
            import traceback
            traceback.print_exc()
            raise
            
    def setup_keyboard(self):
        """Setup keyboard listener for gripper control"""
        def on_press(key):
            try:
                self.pressed_keys.add(key.char)
            except AttributeError:
                pass

        def on_release(key):
            try:
                self.pressed_keys.discard(key.char)
            except AttributeError:
                pass

        self.keyboard_listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        self.keyboard_listener.start()
        
    def _open_gripper_steps(self, steps=20):
        """Open the gripper over multiple steps"""
        if self.action_dim is None or self.action_dim <= 6:
            return
        action = np.zeros(self.action_dim)
        action[-1] = -1.0
        for _ in range(steps):
            self.env.step(action)
            
    def get_action(self):
        """
        Get action from SpaceMouse state (location change for robot arm)
        
        Returns:
            action: numpy array of action values (location change)
            reset: boolean indicating if reset was requested
        """
        if self.device is None:
            return None, False
            
        # Get raw state from SpaceMouse device
        state = self.device.get_controller_state()
        
        # Check for reset signal
        if state['reset']:
            return None, True
        
        # Build action vector (location change)
        action = np.zeros(self.action_dim)
        
        # Position control (first 3 dimensions) - location change
        dpos = state['dpos']
        action[0] = -dpos[0]
        action[1] = -dpos[1]
        action[2] = dpos[2]
        
        # Rotation control (next 3 dimensions if available)
        if self.action_dim > 3:
            if 'raw_drotation' in state:
                raw_drot = state['raw_drotation']
                if isinstance(raw_drot, np.ndarray) and raw_drot.shape == (3,):
                    # Map rotation axes
                    action[3] = -raw_drot[1]  # Roll
                    action[4] = -raw_drot[0]  # Pitch
                    action[5] = -raw_drot[2]  # Yaw
            elif 'drotation' in state:
                drot = state['drotation']
                if isinstance(drot, np.ndarray) and drot.shape == (3,):
                    action[3:6] = drot
        
        # Gripper control (last dimension) - using keyboard
        if self.action_dim > 6:
            if ',' in self.pressed_keys:
                self.gripper_cmd = -1.0  # Open gripper
            elif '.' in self.pressed_keys:
                self.gripper_cmd = 1.0   # Close gripper
            action[-1] = self.gripper_cmd
        
        return action, False

    def start(self):
        """Start device control"""
        if self.device is None:
            raise RuntimeError("SpaceMouse must be set up before starting")
        self.device.start_control()

    def reset_gripper(self):
        """Reset gripper command to open state"""
        self.gripper_cmd = -1.0  # Open gripper
            
    def cleanup(self):
        """Clean up resources"""
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        if self.env:
            self.env.close()
        print("[INFO] Environment closed.")

#!/usr/bin/env python3
"""
KeyboardTeleoperate - Tool for keyboard teleoperation in robosuite
"""

import numpy as np
import robosuite as suite
import time
from pynput import keyboard

from robosuite import load_composite_controller_config


class KeyboardTeleoperate:
    """Keyboard teleoperation class for Robosuite - controls robot arm location"""
    
    def __init__(
        self,
        env_name,
        robot="Panda",
        controller="BASIC",
        control_freq=10,  # Match LIBERO dataset (10 Hz)
        horizon=float('inf'),
        speed=0.05,
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
        Initialize Keyboard teleoperation
        
        Args:
            env_name: Name of the robosuite environment
            robot: Robot name (e.g., "Panda")
            controller: Controller type (e.g., "BASIC")
            control_freq: Control frequency in Hz
            horizon: Episode horizon (use float('inf') for infinite)
            speed: Movement speed for keyboard controls
        """
        self.env_name = env_name
        self.robot = robot
        self.controller = controller
        self.control_freq = control_freq
        self.horizon = horizon
        self.speed = speed
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
        self.pressed_keys = set()
        self.keyboard_listener = None
        self.action_dim = None
        
        print("=" * 60)
        print("KEYBOARD TELEOPERATION FOR ROBOSUITE")
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
        
    def setup_keyboard(self):
        """Setup keyboard listener for robot control"""
        def on_press(key):
            try:
                self.pressed_keys.add(key.char)
            except AttributeError:
                # Handle special keys
                if key == keyboard.Key.esc:
                    self.pressed_keys.add('esc')
                else:
                    self.pressed_keys.add(key)

        def on_release(key):
            try:
                self.pressed_keys.discard(key.char)
            except AttributeError:
                if key == keyboard.Key.esc:
                    self.pressed_keys.discard('esc')
                else:
                    self.pressed_keys.discard(key)

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
        Get action from keyboard state (location change for robot arm)
        
        Returns:
            action: numpy array of action values (location change)
            reset: boolean indicating if reset was requested (ESC key)
        """
        if self.keyboard_listener is None:
            return None, False
        
        # Check for ESC key (reset signal)
        if 'esc' in self.pressed_keys:
            return None, True
        
        # Build action vector (location change)
        action = np.zeros(self.action_dim)
        
        # Map keys to robot actions
        # For OSC_POSE controller: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        key_map = {
            # End-effector position control
            'i': (0, -self.speed),   # Y forward
            'k': (0, self.speed),    # Y backward
            'j': (1, -self.speed),   # X left
            'l': (1, self.speed),    # X right
            'u': (2, -self.speed),   # Z up
            'o': (2, self.speed),    # Z down
            
            # Rotation control (if available)
            'q': (3, -self.speed) if self.action_dim > 3 else None,  # Roll left
            'e': (3, self.speed) if self.action_dim > 3 else None,  # Roll right
            'w': (4, -self.speed) if self.action_dim > 4 else None,  # Pitch up
            's': (4, self.speed) if self.action_dim > 4 else None,  # Pitch down
            'a': (5, -self.speed) if self.action_dim > 5 else None,  # Yaw left
            'd': (5, self.speed) if self.action_dim > 5 else None,  # Yaw right
            
            # Gripper control
            'm': (self.action_dim - 1, self.speed) if self.action_dim > 6 else None,  # Open
            '.': (self.action_dim - 1, -self.speed) if self.action_dim > 6 else None,  # Close
        }
        
        # Apply actions based on pressed keys
        for key in self.pressed_keys:
            if key in key_map:
                mapping = key_map[key]
                if mapping is not None:
                    idx, value = mapping
                    if idx < self.action_dim:
                        action[idx] += value
        
        return action, False

    def start(self):
        """Start keyboard control (no-op for keyboard, but kept for API consistency)"""
        pass

    def reset_gripper(self):
        """Reset gripper state (no-op for keyboard, gripper is stateless)"""
        pass
            
    def cleanup(self):
        """Clean up resources"""
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        if self.env:
            self.env.close()
        print("[INFO] Environment closed.")

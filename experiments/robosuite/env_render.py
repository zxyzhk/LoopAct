#!/usr/bin/env python3
"""
EnvRender - Wrapper to integrate MouseTeleoperate and LinearMover
"""

import os
import time
import numpy as np
from hdf5_recorder import Hdf5Recorder
from teleoperate.mouse_teleoperate import MouseTeleoperate
from teleoperate.keyboard_teleoperate import KeyboardTeleoperate
from target_mover.linear_move import LinearMover
from target_mover.random_move import RandomMover
from target_mover.pattern_move import PatternMover


class EnvRender:
    """Wrapper class to integrate robot arm control and object movement"""
    
    def __init__(
        self,
        env_name,
        robot="Panda",
        controller="BASIC",
        teleoperate_method="spacemouse", # spacemouse or keyboard

        vendor_id=0x256f,
        product_id=0xc652,
        pos_sensitivity=50.0,
        rot_sensitivity=10.0,
        arm="right",

        keyboard_speed=0.05,

        control_freq=10,  # Match LIBERO dataset (10 Hz)
        horizon=float('inf'),

        target_joint_name=None,
        target_speed_y=0.0,
        target_axis='y',
        mover_type='linear',  # 'linear' or 'random'
        decision_interval=1.0,  # For random mover: seconds between speed changes
        speeds=None,  # For random mover: list of speeds to choose from

        render_camera="custom_view", # agentview or custom_view
        camera_names=None,
        camera_height=256,
        camera_width=256,
        show_camera_obs=True,
        camera_window_positions=None,
        camera_window_padding=240,

        record_path=None,
        record_camera_names=None,
        record_camera_height=128,
        record_camera_width=128,
        record_stop_key="p",
        camera_print_key="c",
        free_view_mode=False,
        randomize_init=True,  # Enable LIBERO-style randomization
        task_name=None,  # Task name for success checking
        task_success=None,  # TaskSuccess instance for success checking
        env=None,  # Optional pre-created environment
        slope_angle=None,  # Slope angle for ToyCan environment (degrees)
        action_z_cap=None,  # If set, clips action[2] (delta-z) to ±this value during collection
    ):
        """
        Initialize teleoperation wrapper
        
        Args:
            env_name: Name of the robosuite environment
            robot: Robot name (e.g., "Panda")
            controller: Controller type (e.g., "BASIC")
            teleoperate_method: "spacemouse" or "keyboard"
            vendor_id: SpaceMouse vendor ID (only for spacemouse)
            product_id: SpaceMouse product ID (only for spacemouse)
            pos_sensitivity: Position control sensitivity for robot arm (only for spacemouse)
            rot_sensitivity: Rotation control sensitivity for robot arm (only for spacemouse)
            arm: Which arm to control ("right" or "left") (only for spacemouse)
            keyboard_speed: Movement speed for keyboard controls (only for keyboard)
            control_freq: Control frequency in Hz
            horizon: Episode horizon (use float('inf') for infinite)
            target_joint_name: Name of joint/object to move (None to disable)
            target_speed_y: Speed for moving target along Y axis (units per second)
            target_axis: Axis for target movement ('x', 'y', or 'z')
            env: Optional pre-created environment
        """
        # Initialize robot arm controller based on method
        self.teleoperate_method = teleoperate_method.lower()
        self.render_camera = None if free_view_mode else render_camera
        self.camera_names = list(camera_names) if camera_names else []
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.show_camera_obs = show_camera_obs
        self.camera_window_positions = camera_window_positions or {}
        self.camera_window_padding = camera_window_padding
        self.record_path = record_path
        self.record_camera_names = list(record_camera_names) if record_camera_names else []
        self.record_camera_height = record_camera_height
        self.record_camera_width = record_camera_width
        self.record_stop_key = record_stop_key
        self.camera_print_key = camera_print_key
        self.free_view_mode = free_view_mode
        self.record_enabled = self.record_path is not None
        self.recorder = Hdf5Recorder(self.record_path) if self.record_enabled else None
        self._cv2 = None
        self.randomize_init = randomize_init
        self.control_freq = control_freq
        self.frame_time = 1.0 / control_freq  # Time per frame in seconds
        self.action_z_cap = action_z_cap

        if self.record_enabled and not self.record_camera_names:
            self.record_camera_names = ["agentview", "robot0_eye_in_hand"]
        if self.record_enabled:
            for name in ["agentview", "robot0_eye_in_hand"]:
                if name not in self.record_camera_names:
                    self.record_camera_names.append(name)

        if self.show_camera_obs and not self.camera_names:
            self.camera_names = ["agentview", "robot0_eye_in_hand"]
        if self.show_camera_obs and not self.camera_window_positions:
            # Position camera windows on the right side to avoid overlap with MuJoCo viewer
            # Adjust x_offset based on your screen resolution (e.g., 1920 - 300 for 1920px wide screen)
            x_offset = 2500  # Move windows to the right side of screen
            self.camera_window_positions = {
                "agentview": (x_offset, 0),
                "robot0_eye_in_hand": (x_offset, self.camera_height + self.camera_window_padding),
            }

        if self.show_camera_obs:
            try:
                import cv2  # type: ignore
                self._cv2 = cv2
            except Exception as exc:
                self.show_camera_obs = False
                print(f"[WARN] OpenCV not available, disabling camera windows: {exc}")
        
        env_camera_names = []
        for name in self.camera_names + self.record_camera_names:
            if name not in env_camera_names:
                env_camera_names.append(name)

        use_camera_obs = self.show_camera_obs or self.record_enabled
        has_offscreen_renderer = self.show_camera_obs or self.record_enabled
        camera_height = self.camera_height if self.show_camera_obs else self.record_camera_height
        camera_width = self.camera_width if self.show_camera_obs else self.record_camera_width

        if self.teleoperate_method == "spacemouse":
            self.teleop = MouseTeleoperate(
                env_name=env_name,
                robot=robot,
                controller=controller,
                vendor_id=vendor_id,
                product_id=product_id,
                pos_sensitivity=pos_sensitivity,
                rot_sensitivity=rot_sensitivity,
                arm=arm,
                control_freq=control_freq,
                horizon=horizon,
                render_camera=self.render_camera,
                use_camera_obs=use_camera_obs,
                has_offscreen_renderer=has_offscreen_renderer,
                camera_names=env_camera_names if use_camera_obs else None,
                camera_heights=camera_height if use_camera_obs else None,
                camera_widths=camera_width if use_camera_obs else None,
                randomize_init=self.randomize_init,
                task_name=task_name,
                task_success=task_success,
                slope_angle=slope_angle,
            )
        elif self.teleoperate_method == "keyboard":
            self.teleop = KeyboardTeleoperate(
                env_name=env_name,
                robot=robot,
                controller=controller,
                control_freq=control_freq,
                horizon=horizon,
                speed=keyboard_speed,
                render_camera=self.render_camera,
                use_camera_obs=use_camera_obs,
                has_offscreen_renderer=has_offscreen_renderer,
                camera_names=env_camera_names if use_camera_obs else None,
                camera_heights=camera_height if use_camera_obs else None,
                camera_widths=camera_width if use_camera_obs else None,
                randomize_init=self.randomize_init,
                task_name=task_name,
                task_success=task_success,
                slope_angle=slope_angle,
            )
        else:
            raise ValueError(f"Unknown teleoperate_method: {teleoperate_method}. Must be 'spacemouse' or 'keyboard'")
        
        # Setup environment (shared between both controllers)
        self.teleop.setup_environment(env=env)
        
        # Update recorder with task information from environment
        if self.record_enabled and hasattr(self.teleop.env, 'task_name'):
            task_name = self.teleop.env.task_name
            task_description = self.teleop.env.get_task_language() if hasattr(self.teleop.env, 'get_task_language') else ""
            self.recorder = Hdf5Recorder(self.record_path, task_name=task_name, task_description=task_description)
            print(f"[RECORD] Task: {task_description}")
        
        # Print available objects for movement
        if hasattr(self.teleop.env, "object_joints"):
            available_objects = [name for name, joint in self.teleop.env.object_joints.items() 
                                if joint is not None]
            if available_objects:
                print(f"[INFO] Available objects for movement: {', '.join(available_objects)}")
        
        # Resolve target joint name from object label
        # If target_joint_name is None, no object will move (no auto-detection)
        resolved_joint_name = None
        if target_joint_name is not None:
            # Try to resolve using object_joints mapping (simple names: "sphere", "square")
            if hasattr(self.teleop.env, "object_joints"):
                resolved_joint_name = self.teleop.env.object_joints.get(target_joint_name, None)
            
            # Fallback: try direct attribute access (e.g., env.sphere_joint)
            if resolved_joint_name is None:
                resolved_joint_name = getattr(self.teleop.env, f"{target_joint_name}_joint", None)
            
            # If still None, try using target_joint_name directly as joint name
            if resolved_joint_name is None:
                resolved_joint_name = target_joint_name
        
        # Initialize target mover if joint name is available
        self.target_mover = None
        if resolved_joint_name is not None:
            if mover_type == 'random':
                self.target_mover = RandomMover(
                    env=self.teleop.env,
                    joint_name=resolved_joint_name,
                    control_freq=control_freq,
                    speeds=speeds,
                    axis=target_axis,
                    decision_interval=decision_interval,
                )
                print(f"[INFO] Target mover initialized: object={target_joint_name}, type=random, speeds={speeds}, interval={decision_interval}s")
            elif mover_type == 'pattern':
                self.target_mover = PatternMover(
                    env=self.teleop.env,
                    joint_name=resolved_joint_name,
                    control_freq=control_freq,
                    patterns=[speeds] if speeds is not None else None,
                    axis=target_axis,
                    decision_interval=decision_interval,
                )
                print(f"[INFO] Target mover initialized: object={target_joint_name}, type=pattern, speeds={speeds}, interval={decision_interval}s")
            else:
                self.target_mover = LinearMover(
                    env=self.teleop.env,
                    joint_name=resolved_joint_name,
                    control_freq=control_freq,
                    speed_y=target_speed_y,
                    axis=target_axis,
                )
                print(f"[INFO] Target mover initialized: object={target_joint_name}, joint={resolved_joint_name}, speed={target_speed_y} m/s, axis={target_axis}")
        else:
            if target_joint_name is None:
                print(f"[INFO] No target mover: target_joint_name is None (no object will move)")
            else:
                print(f"[INFO] No target mover: could not resolve target_joint_name='{target_joint_name}'")
        
        # Setup teleoperation device
        if self.teleoperate_method == "spacemouse":
            self.teleop.setup_spacemouse()
        self.teleop.setup_keyboard()  # Keyboard setup needed for both (gripper control for spacemouse)


    def _get_obs_image(self, obs, keys):
        if obs is None:
            return None
        for key in keys:
            if key in obs:
                return obs[key]
        return None

    def _show_camera_obs(self, obs):
        if not self.show_camera_obs or self._cv2 is None:
            return
        if not self.camera_names:
            return

        for camera_name in self.camera_names:
            if camera_name == "agentview":
                keys = ["agentview_image", "agentview_rgb"]
            elif camera_name == "robot0_eye_in_hand":
                keys = ["robot0_eye_in_hand_image", "eye_in_hand_image", "eye_in_hand_rgb"]
            else:
                keys = [f"{camera_name}_image", f"{camera_name}_rgb"]

            frame = self._get_obs_image(obs, keys)
            if frame is None:
                continue

            bgr = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
            
            # Transform eye-in-hand view for display only
            if camera_name == "robot0_eye_in_hand":
                bgr = self._cv2.flip(bgr, 1)  # 1. Flip horizontally (mirror left/right)
                bgr = self._cv2.rotate(bgr, self._cv2.ROTATE_180)  # 2. Rotate 180 degrees
            
            self._cv2.namedWindow(camera_name, self._cv2.WINDOW_NORMAL)
            if camera_name in self.camera_window_positions:
                x, y = self.camera_window_positions[camera_name]
                self._cv2.moveWindow(camera_name, int(x), int(y))
            self._cv2.imshow(camera_name, bgr)

        self._cv2.waitKey(1)

    def _rotation_matrix_to_quat(self, rot):
        trace = np.trace(rot)
        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rot[2, 1] - rot[1, 2]) * s
            y = (rot[0, 2] - rot[2, 0]) * s
            z = (rot[1, 0] - rot[0, 1]) * s
        else:
            if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
                w = (rot[2, 1] - rot[1, 2]) / s
                x = 0.25 * s
                y = (rot[0, 1] + rot[1, 0]) / s
                z = (rot[0, 2] + rot[2, 0]) / s
            elif rot[1, 1] > rot[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
                w = (rot[0, 2] - rot[2, 0]) / s
                x = (rot[0, 1] + rot[1, 0]) / s
                y = 0.25 * s
                z = (rot[1, 2] + rot[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
                w = (rot[1, 0] - rot[0, 1]) / s
                x = (rot[0, 2] + rot[2, 0]) / s
                y = (rot[1, 2] + rot[2, 1]) / s
                z = 0.25 * s
        return np.array([w, x, y, z], dtype=np.float64)

    def _print_viewer_camera(self):
        viewer = getattr(self.teleop.env, "viewer", None)
        if viewer is None:
            print("[WARN] Viewer camera not available.")
            return

        cam = None
        for candidate in (
            viewer,
            getattr(viewer, "cam", None),
            getattr(viewer, "_viewer", None),
            getattr(viewer, "viewer", None),
            getattr(viewer, "_render_context", None),
            getattr(viewer, "render_context", None),
            getattr(viewer, "_mjviewer", None),
            getattr(viewer, "mjviewer", None),
        ):
            if candidate is None:
                continue
            if hasattr(candidate, "cam"):
                cam = candidate.cam
                break
            if all(hasattr(candidate, attr) for attr in ["lookat", "distance", "azimuth", "elevation"]):
                cam = candidate
                break
            if all(hasattr(candidate, attr) for attr in ["pos", "quat"]):
                cam = candidate
                break

        if cam is None:
            print("[WARN] Viewer camera not available.")
            return
        if hasattr(cam, "pos") and hasattr(cam, "quat"):
            pos = np.asarray(cam.pos, dtype=np.float64)
            quat = np.asarray(cam.quat, dtype=np.float64)
            fovy = getattr(cam, "fovy", 60)
            print(f"[INFO] Viewer camera pos={pos} quat={quat}")
            print(
                "[INFO] XML camera: "
                f"<camera mode=\"fixed\" name=\"custom_view\" pos=\"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\" "
                f"quat=\"{quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}\" fovy=\"{fovy}\"/>"
            )
            return

        if not all(hasattr(cam, attr) for attr in ["lookat", "distance", "azimuth", "elevation"]):
            print("[WARN] Viewer camera fields not available.")
            return

        lookat = np.asarray(cam.lookat, dtype=np.float64)
        distance = float(cam.distance)
        azimuth = float(cam.azimuth)
        elevation = float(cam.elevation)

        az = np.deg2rad(azimuth)
        el = np.deg2rad(elevation)
        forward = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ])
        pos = lookat - forward * distance

        up = np.array([0.0, 0.0, 1.0])
        z_axis = -forward
        right = np.cross(up, z_axis)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= right_norm
        up_axis = np.cross(z_axis, right)

        rot = np.stack([right, up_axis, z_axis], axis=1)
        quat = self._rotation_matrix_to_quat(rot)
        fovy = getattr(cam, "fovy", 60)
        print(
            "[INFO] Viewer camera (derived) "
            f"pos={pos} quat={quat} "
            f"(lookat={lookat}, distance={distance}, azimuth={azimuth}, elevation={elevation})"
        )
        print(
            "[INFO] XML camera: "
            f"<camera mode=\"fixed\" name=\"custom_view\" pos=\"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\" "
            f"quat=\"{quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}\" fovy=\"{fovy}\"/>"
        )

    def _reset_recording(self):
        if not self.record_enabled:
            return
        self.recorder.reset()

    def _record_step(self, obs, action, reward, done):
        if not self.record_enabled:
            return
        self.recorder.record_step(self.teleop.env, obs, action, reward, done)

    def _save_recording(self):
        if not self.record_enabled:
            return
        
        # Check task success before saving
        if hasattr(self.teleop.env, 'check_success'):
            try:
                success, score, details = self.teleop.env.check_success(verbose=True)
                self.recorder.set_task_success(success, score)
                print(f"\n[TASK] Success: {success}, Score: {score:.2%}")
            except Exception as e:
                print(f"[WARNING] Could not check task success: {e}")
        
        self.recorder.save()
        
    def run(self):
        """Main teleoperation loop - integrates robot arm and object movement"""
        try:
            # Start device control
            self.teleop.start()
            
            # Print controls based on teleoperation method
            print("\n" + "=" * 60)
            print("CONTROLS")
            print("=" * 60)
            
            if self.teleoperate_method == "spacemouse":
                print("  SpaceMouse Translation: Move robot arm position (X, Y, Z)")
                print("  SpaceMouse Rotation:    Rotate robot arm (Roll, Pitch, Yaw)")
                print("  Keyboard , (comma):     Open gripper")
                print("  Keyboard . (period):    Close gripper")
                print("  SpaceMouse Reset:       Reset environment")
            else:  # keyboard
                print("  IJKL:                   Move end-effector in XY plane")
                print("  U/O:                    Move end-effector up/down (Z)")
                print("  Q/E:                    Rotate roll")
                print("  W/S:                    Rotate pitch")
                print("  A/D:                    Rotate yaw")
                print("  M/.:                    Open/Close gripper")
                print("  ESC:                    Reset environment")
            
            if self.target_mover:
                speeds = getattr(self.target_mover, 'speeds', None)
                print(f"  Target Object:          speeds={speeds}")
            print("  Ctrl+C:                 Exit")
            print("=" * 60)
            
            method_msg = "Move the SpaceMouse" if self.teleoperate_method == "spacemouse" else "Use keyboard keys"
            print(f"\n[OK] Teleoperation started. {method_msg} to control the robot!")
            if self.record_enabled:
                print(f"[OK] Recording enabled. Press '{self.record_stop_key}' to stop and save.")
                self._reset_recording()
            
            # Main loop
            step = 0
            loop_start_time = time.time()
            while True:
                pressed_keys = getattr(self.teleop, "pressed_keys", set())
                if self.camera_print_key and self.camera_print_key in pressed_keys:
                    pressed_keys.discard(self.camera_print_key)
                    self._print_viewer_camera()

                # Get robot arm action from teleoperation device (location change)
                action, reset = self.teleop.get_action()
                
                # Handle reset
                if reset:
                    reset_source = "SpaceMouse" if self.teleoperate_method == "spacemouse" else "ESC key"
                    print(f"\n[INFO] Reset signal received ({reset_source}), resetting environment...")
                    obs = self.teleop.env.reset()
                    if self.target_mover:
                        self.target_mover.reset()
                    self.teleop._open_gripper_steps()
                    self.teleop.env.render()
                    if self.record_enabled:
                        self._reset_recording()
                    self._show_camera_obs(obs)
                    continue
                
                # Update target object position (location change)
                if self.target_mover:
                    self.target_mover.step()
                
                # Apply robot arm action (location change)
                obs, reward, done, info = self.teleop.env.step(action)
                self.teleop.env.render()
                self._show_camera_obs(obs)
                self._record_step(obs, action, reward, done)
                
                step += 1
                if step % 100 == 0:
                    print(f"Step {step} - Teleoperation active")
                
                if done:
                    print("\n[INFO] Episode done, resetting...")
                    obs = self.teleop.env.reset()
                    if self.target_mover:
                        self.target_mover.reset()
                    self.teleop.env.render()
                    self._show_camera_obs(obs)
                    if self.record_enabled:
                        print("[INFO] Recording stopped (episode done). Saving...")
                        self._save_recording()
                        break

                if self.record_enabled and self.record_stop_key:
                    if self.record_stop_key in getattr(self.teleop, "pressed_keys", set()):
                        print("\n[INFO] Recording stopped (stop key). Saving...")
                        self._save_recording()
                        break

                # Frame rate limiting to match control_freq
                elapsed = time.time() - loop_start_time
                sleep_time = self.frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                loop_start_time = time.time()
                
        except KeyboardInterrupt:
            print("\n\n[INFO] Stopped by user (Ctrl+C)")
            if self.record_enabled and self.recorder and self.recorder.actions:
                print("[INFO] Saving recording after interrupt...")
                self._save_recording()
        except Exception as e:
            print(f"\n[ERROR] Runtime error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources"""
        self.teleop.cleanup()
        if self._cv2 is not None:
            self._cv2.destroyAllWindows()
        print("[INFO] Teleoperation wrapper closed.")

    def collect_demos(
        self,
        output_dir=".",
        prefix="demo",
        save_key="l",
        discard_key="k",
    ):
        if not self.record_enabled or self.recorder is None:
            raise RuntimeError("Recording is not enabled. Set record_path to collect demos.")

        os.makedirs(output_dir, exist_ok=True)

        def reset_episode():
            obs = self.teleop.env.reset()
            if self.target_mover:
                self.target_mover.reset()
            self.teleop.reset_gripper()  # Reset gripper command state
            self.teleop._open_gripper_steps()
            self.teleop.env.render()
            self._show_camera_obs(obs)
            self.recorder.reset()

        try:
            self.teleop.start()
            print("\n[OK] Demo collector started. Press Ctrl+C to quit.")
            print(f"[INFO] Press '{save_key}' to save, '{discard_key}' to discard.")

            demo_index = 0
            awaiting_decision = False
            reset_episode()
            print(f"[INFO] Start demo {demo_index}")
            print(f"[INFO] Recording at {self.control_freq} Hz")

            loop_start_time = time.time()
            while True:
                pressed_keys = getattr(self.teleop, "pressed_keys", set())
                if self.camera_print_key and self.camera_print_key in pressed_keys:
                    pressed_keys.discard(self.camera_print_key)
                    self._print_viewer_camera()
                if save_key in pressed_keys:
                    pressed_keys.discard(save_key)
                    if self.recorder.actions:
                        output_path = os.path.join(output_dir, f"{prefix}_{demo_index}.hdf5")
                        self.recorder.save(output_path)
                        print(f"[OK] Saved to {output_path}")
                        demo_index += 1
                    else:
                        print("[WARN] No steps recorded; nothing to save.")
                    awaiting_decision = False
                    reset_episode()
                    print("=" * 60)
                    print(f"[INFO] Start demo {demo_index}")
                    continue

                if discard_key in pressed_keys:
                    pressed_keys.discard(discard_key)
                    self.recorder.reset()
                    print("[INFO] Previous run discarded.")

                    awaiting_decision = False
                    reset_episode()
                    print("=" * 60)
                    print(f"[INFO] Start demo {demo_index}")

                    continue

                if awaiting_decision:
                    time.sleep(0.01)
                    continue

                action, reset = self.teleop.get_action()

                if reset:
                    print("\n[INFO] Reset signal received; restarting demo.")
                    reset_episode()
                    continue

                if action is None:
                    time.sleep(0.01)
                    continue

                if self.action_z_cap is not None:
                    action[2] = np.clip(action[2], -self.action_z_cap, self.action_z_cap)

                if self.target_mover:
                    self.target_mover.step()

                obs, reward, done, info = self.teleop.env.step(action)
                self.teleop.env.render()
                self._show_camera_obs(obs)
                self.recorder.record_step(self.teleop.env, obs, action, reward, done)

                if done:
                    print("\n[INFO] Episode done. Press save/discard to continue.")
                    awaiting_decision = True

                # Frame rate limiting to match control_freq
                elapsed = time.time() - loop_start_time
                sleep_time = self.frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                loop_start_time = time.time()
        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user (Ctrl+C).")
        finally:
            self.cleanup()
    
    def set_target_speed(self, speed_y):
        """Change target movement speed dynamically"""
        if self.target_mover:
            self.target_mover.set_speed(speed_y)
            print(f"[INFO] Target speed changed to {speed_y} m/s")
    
    def set_target_axis(self, axis):
        """Change target movement axis dynamically"""
        if self.target_mover:
            self.target_mover.set_axis(axis)
            print(f"[INFO] Target axis changed to {axis}")

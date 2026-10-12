#!/usr/bin/env python3
"""
LIBERO-style Inference Wrapper for Robot Policy Evaluation

This wrapper provides a clean interface for running inference with robot policies,
similar to LIBERO's structure. It handles:
- Environment creation and configuration
- Episode management with randomization
- Observation collection and preprocessing
- Action execution with trained models
- Success metrics and logging
"""

import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config
import robosuite.macros as macros
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
from typing import Dict, List, Optional, Tuple, Callable
import time
import os
import sys
from datetime import datetime
import cv2

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
robosuite_dir = os.path.dirname(current_dir)
sys.path.insert(0, robosuite_dir)

from target_mover.linear_move import LinearMover
from target_mover.random_move import RandomMover
from target_mover.pattern_move import PatternMover


class InferenceWrapper:
    """
    LIBERO-style wrapper for running inference on robosuite environments.

    Example usage:
        from your_model import YourModel
        model = YourModel(checkpoint_path="path/to/checkpoint")
        wrapper = InferenceWrapper(
            env_name="TableLadle",
            robot="Panda",
            control_freq=10,
            model=model
        )

        # Run episodes
        results = wrapper.run_episode(max_steps=500)
    """
    
    def __init__(
        self,
        env_name: str = "TableLadle",
        robot: str = "Panda",
        controller: str = "BASIC",
        control_freq: int = 10,
        horizon: int = 400,
        camera_names: Optional[List[str]] = None,
        camera_heights: int = 240,
        camera_widths: int = 320,
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        use_camera_obs: bool = True,
        randomize_init: bool = True,
        target_joint_name: Optional[str] = None,
        target_speed_y: float = 0.0,
        target_axis: str = 'y',
        mover_type: str = 'linear',  # 'linear', 'random', or 'pattern'
        decision_interval: float = 1.0,  # For random/pattern mover: seconds between speed changes
        speeds: Optional[List[float]] = None,  # For random mover: list of speeds to choose from
        inference_patterns: Optional[List] = None,  # For pattern mover: task-specific speed sequences
        pattern_id: int = 0,  # For pattern mover: which pattern to use
        task_name: Optional[str] = None,
        task_description: Optional[str] = None,
        task_success = None,
        model = None,  # Your trained model (must have get_action method)
        video_resolution: Optional[int] = None,  # Render agentview video frames at this size; None = use obs resolution
        **kwargs
    ):
        """
        Initialize the inference wrapper.
        
        Args:
            env_name: Name of the environment (e.g., "TableLadle")
            robot: Robot name (e.g., "Panda")
            controller: Controller type (e.g., "BASIC", "OSC_POSE")
            control_freq: Control frequency in Hz (LIBERO uses 10Hz)
            horizon: Maximum timesteps per episode
            camera_names: List of camera names for observations
            camera_heights: Height of camera images
            camera_widths: Width of camera images
            has_renderer: Whether to show on-screen rendering
            has_offscreen_renderer: Whether to render off-screen for observations
            use_camera_obs: Whether to include camera observations
            randomize_init: Whether to randomize initial positions (LIBERO-style)
            target_joint_name: Name of object to move (e.g., "ladle"), None to disable
            target_speed_y: Speed for moving target (units per second)
            target_axis: Axis for target movement ('x', 'y', or 'z')
            task_description: Natural language task description for model
            **kwargs: Additional environment parameters
        """
        self.env_name = env_name
        self.robot = robot
        self.controller = controller
        self.control_freq = control_freq
        self.horizon = horizon
        self.randomize_init = randomize_init
        self.task_description = task_description or ""

        # High-resolution video rendering (decoupled from observations / inference)
        self.video_resolution = video_resolution

        # Store model (required for inference)
        self.model = model
        
        # Default cameras for LIBERO-style setup
        if camera_names is None:
            camera_names = ["agentview", "robot0_eye_in_hand"]
        self.camera_names = camera_names
        
        # Load controller configuration
        controller_config = load_composite_controller_config(
            controller=controller,
            robot=robot
        )
        
        # Create environment
        env_kwargs = {
            "env_name": env_name,
            "robots": robot,
            "controller_configs": controller_config,
            "has_renderer": has_renderer,
            "has_offscreen_renderer": has_offscreen_renderer,
            "use_camera_obs": use_camera_obs,
            "control_freq": control_freq,
            "horizon": horizon,
            "camera_names": camera_names,
            "camera_heights": camera_heights,
            "camera_widths": camera_widths,
            "randomize_init": randomize_init,
        }
        
        # Add task_name if provided
        if task_name is not None:
            env_kwargs["task_name"] = task_name
        
        # Add task_success if provided
        if task_success is not None:
            env_kwargs["task_success"] = task_success
        
        # Add any additional kwargs
        env_kwargs.update(kwargs)
        
        self.env = suite.make(**env_kwargs)
        
        # Get action space dimensions
        self.action_dim = self.env.action_dim
        
        # Initialize target mover (for moving objects like in data collection)
        self.target_mover = None
        resolved_joint_name = None
        if target_joint_name is not None:
            # Try to resolve using object_joints mapping
            if hasattr(self.env, "object_joints"):
                resolved_joint_name = self.env.object_joints.get(target_joint_name, None)
            
            # Fallback: try direct attribute access
            if resolved_joint_name is None:
                resolved_joint_name = getattr(self.env, f"{target_joint_name}_joint", None)
            
            # If still None, use target_joint_name directly
            if resolved_joint_name is None:
                resolved_joint_name = target_joint_name
            
            # Store pattern_id for later use
            self.pattern_id = pattern_id

            # Create target mover based on type
            if resolved_joint_name is not None:
                if mover_type == 'random':
                    self.target_mover = RandomMover(
                        env=self.env,
                        joint_name=resolved_joint_name,
                        control_freq=control_freq,
                        speeds=speeds,
                        axis=target_axis,
                        decision_interval=decision_interval,
                    )
                elif mover_type == 'pattern':
                    self.target_mover = PatternMover(
                        env=self.env,
                        joint_name=resolved_joint_name,
                        control_freq=control_freq,
                        patterns=inference_patterns,
                        pattern_id=pattern_id,
                        axis=target_axis,
                        decision_interval=decision_interval,
                    )
                else:
                    self.target_mover = LinearMover(
                        env=self.env,
                        joint_name=resolved_joint_name,
                        control_freq=control_freq,
                        speed_y=target_speed_y,
                        axis=target_axis,
                    )

        print(f"[Wrapper] Environment '{env_name}' created")
        print(f"[Wrapper] Action dimension: {self.action_dim}")
        print(f"[Wrapper] Control frequency: {control_freq} Hz")
        print(f"[Wrapper] Cameras: {camera_names}")
        print(f"[Wrapper] Randomization: {randomize_init}")
        if self.model is not None:
            print(f"[Wrapper] Model: {type(self.model).__name__} (loaded)")
        else:
            print(f"[Wrapper] WARNING: No model provided")
        if self.target_mover:
            if mover_type == 'random':
                print(f"[Wrapper] Target mover: {target_joint_name} (type=random, speeds={speeds}, interval={decision_interval}s)")
            elif mover_type == 'pattern':
                print(f"[Wrapper] Target mover: {target_joint_name} (type=pattern, pattern_id={pattern_id}, interval={decision_interval}s)")
            else:
                print(f"[Wrapper] Target mover: {target_joint_name} (type=linear, speed={target_speed_y} m/s, axis={target_axis})")
        else:
            print(f"[Wrapper] Target mover: disabled")
    
    def reset(self, mover_seed: Optional[int] = None, pattern_id: Optional[int] = None) -> Dict:
        """
        Reset the environment with randomization.

        Args:
            mover_seed: Seed for target mover randomness (for reproducibility)
            pattern_id: Pattern ID for PatternMover (for controlled testing)

        Returns:
            Initial observation dictionary
        """
        obs = self.env.reset()
        # Clear success-check state from previous episode
        if hasattr(self.env, "_lift_history"):
            self.env._lift_history = {}
        if hasattr(self.env, "_surface_lift_history"):
            self.env._surface_lift_history = {}
        if self.target_mover:
            if isinstance(self.target_mover, PatternMover) and pattern_id is not None:
                self.target_mover.reset(pattern_id=pattern_id)
            elif isinstance(self.target_mover, LinearMover):
                self.target_mover.reset()
            else:
                self.target_mover.reset(seed=mover_seed)

        # Reset model state if it has a reset method (for action chunking)
        if self.model is not None and hasattr(self.model, 'reset'):
            self.model.reset()

        return obs
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action array to execute
            
        Returns:
            obs: Observation dictionary
            reward: Reward value
            done: Whether episode is done
            info: Additional information
        """
        # Step target mover before environment step (like in data collection)
        if self.target_mover:
            self.target_mover.step()
        
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
    
    def _save_videos(self, video_frames: Dict[str, List], video_dir: str, episode_idx: int, success: bool = False) -> List[str]:
        """
        Save video frames to MP4 files.

        Args:
            video_frames: Dictionary mapping camera names to lists of frames
            video_dir: Directory to save videos
            episode_idx: Episode index for naming
            success: Whether the episode was successful (included in filename)

        Returns:
            List of saved video file paths
        """
        os.makedirs(video_dir, exist_ok=True)
        video_paths = []

        if success:
            result_str = "success"
        else:
            result_str = "fail"

        for cam_name, frames in video_frames.items():
            if len(frames) == 0:
                continue

            # Create video filename with success/fail
            video_path = os.path.join(video_dir, f"episode_{episode_idx:04d}_{cam_name}_{result_str}.mp4")
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Create video writer (MP4 codec)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.control_freq
            writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            # Write frames
            for frame in frames:
                writer.write(frame)
            
            writer.release()
            video_paths.append(video_path)
        
        return video_paths
    
    def _capture_agentview_frame(self, obs: Dict) -> Optional[np.ndarray]:
        """
        Produce a BGR agentview frame for video, applying the same orientation
        transform used elsewhere (matches env_render.py display).

        If ``video_resolution`` is set, render a dedicated high-resolution frame
        from the current sim state via ``sim.render`` instead of using the
        observation image. This is purely for video — the model still receives
        the native-resolution observation, so inference is unaffected.
        """
        if self.video_resolution is not None:
            # Same camera and sim state that produced the observation, just at
            # higher resolution. robosuite grows its offscreen framebuffer on demand.
            # Match the obs orientation via the active image convention (obs is
            # rendered as sim.render(...)[::convention]).
            convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
            frame = self.env.sim.render(
                camera_name="agentview",
                width=self.video_resolution,
                height=self.video_resolution,
                depth=False,
            )[::convention]
            frame = np.ascontiguousarray(frame)
        else:
            if "agentview_image" not in obs:
                return None
            frame = obs["agentview_image"]

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Apply same transformation as eye-in-hand display in env_render.py
        frame_bgr = cv2.flip(frame_bgr, 1)  # 1. Flip horizontally (mirror left/right)
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)  # 2. Rotate 180 degrees
        return frame_bgr

    def get_action(self, obs: Dict, step: int = 0) -> np.ndarray:
        """
        Get action from the model.

        This is the main interface for getting actions - similar to LIBERO runner.

        Args:
            obs: Current observation dictionary
            step: Current step number (useful for action chunking)

        Returns:
            Action array to execute

        Usage:
            action = wrapper.get_action(obs, step)
        """
        if self.model is None:
            raise ValueError("No model provided. InferenceWrapper requires a model for inference.")

        # Use model to get action
        # Your model should have a get_action(obs, ...) method
        if hasattr(self.model, 'get_action'):
            # Try to pass task_description if model supports it
            try:
                return self.model.get_action(obs, step, task_description=self.task_description)
            except TypeError:
                # Fallback if model doesn't accept task_description
                return self.model.get_action(obs, step)
        elif hasattr(self.model, 'predict'):
            # Fallback for models with predict() method
            return self.model.predict(obs)
        elif callable(self.model):
            # Fallback for function-based models
            return self.model(obs)
        else:
            raise ValueError(f"Model {type(self.model).__name__} must have 'get_action' or 'predict' method")
    
    def run_episode(
        self,
        max_steps: Optional[int] = None,
        render: bool = False,
        verbose: bool = False,
        save_video: bool = False,
        video_dir: Optional[str] = None,
        episode_idx: int = 0,
        policy_fn: Optional[Callable] = None,  # Deprecated, use model instead
        early_termination: bool = False,  # Stop early on success
        success_check_freq: int = 10,  # How often to check for success (every N steps)
        mover_seed: Optional[int] = None,  # Seed for target mover randomness
        pattern_id: Optional[int] = None,  # Pattern ID for PatternMover
    ) -> Dict:
        """
        Run a single episode with the model.

        Args:
            max_steps: Maximum steps per episode. If None, uses horizon.
            render: Whether to render the environment
            verbose: Whether to print step information
            save_video: Whether to save videos from cameras
            video_dir: Directory to save videos (required if save_video=True)
            episode_idx: Episode index for naming video files
            policy_fn: [DEPRECATED] Use model parameter in __init__ instead
            early_termination: If True, stop episode as soon as success is detected
            success_check_freq: How often to check for success (every N steps)
            mover_seed: Seed for target mover randomness (for reproducibility)
            pattern_id: Pattern ID for PatternMover (for controlled testing)

        Returns:
            Dictionary containing episode results:
                - observations: List of observations
                - actions: List of actions
                - rewards: List of rewards
                - success: Whether task succeeded
                - steps: Number of steps taken
                - runtime: Episode runtime in seconds
                - video_paths: List of saved video paths (if save_video=True)
        """
        # Backward compatibility: if policy_fn is provided, use it
        use_policy_fn = policy_fn is not None
        
        if max_steps is None:
            max_steps = self.horizon

        # Reset environment
        obs = self.reset(mover_seed=mover_seed, pattern_id=pattern_id)
        
        # Storage for episode data
        observations = [obs]
        actions = []
        rewards = []
        
        # Storage for video frames (if saving video)
        # Only save agentview camera
        video_frames = {"agentview": []} if save_video else None
        
        done = False
        step_count = 0
        start_time = time.time()
        if verbose:
            print(f"\n[Episode Start] Max steps: {max_steps}")

        # Collect initial frame for video (agentview only, with transformation)
        if save_video and video_frames is not None:
            frame_bgr = self._capture_agentview_frame(obs)
            if frame_bgr is not None:
                video_frames["agentview"].append(frame_bgr)

        # Run episode
        while not done and step_count < max_steps:
            # Get action from model
            if use_policy_fn:
                # Backward compatibility
                action = policy_fn(obs)
            else:
                # Use get_action() method (recommended)
                action = self.get_action(obs, step_count)

            actions.append(action)

            # Execute action
            obs, reward, done, info = self.step(action)
            observations.append(obs)
            rewards.append(reward)
            
            # Collect frames for video (agentview only, with transformation)
            if save_video and video_frames is not None:
                frame_bgr = self._capture_agentview_frame(obs)
                if frame_bgr is not None:
                    video_frames["agentview"].append(frame_bgr)
            
            if render:
                self.env.render()

            step_count += 1

            if verbose and step_count % 50 == 0:
                print(f"  Step {step_count}/{max_steps}")

            # Early termination: check for success periodically (silent)
            if early_termination and step_count % success_check_freq == 0:
                if hasattr(self.env, 'check_success'):
                    success_check, _, _ = self.env.check_success(verbose=False)
                    if success_check:
                        done = True

        runtime = time.time() - start_time

        # Check success - use check_success if available, otherwise fall back to info
        success = False
        if step_count > 0:
            if hasattr(self.env, 'check_success'):
                success, _, _ = self.env.check_success(verbose=False)
            else:
                success = info.get("success", False)
        
        # Save videos if requested
        video_paths = []
        if save_video and video_frames is not None and video_dir is not None:
            video_paths = self._save_videos(video_frames, video_dir, episode_idx, success=success)

        results = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "success": success,
            "steps": step_count,
            "runtime": runtime,
            "info": info if step_count > 0 else {},
            "video_paths": video_paths,
        }
        
        if verbose:
            print(f"[Episode End] Steps: {step_count}, Runtime: {runtime:.2f}s, Success: {success}")
            if video_paths:
                print(f"[Video] Saved {len(video_paths)} video(s)")
        
        return results
    
    def run_episodes(
        self,
        num_episodes: int,
        max_steps: Optional[int] = None,
        render: bool = False,
        verbose: bool = True,
        save_video: bool = False,
        video_base_dir: Optional[str] = None,
        policy_fn: Optional[Callable] = None  # Deprecated, use model instead
    ) -> Dict:
        """
        Run multiple episodes and collect statistics.
        
        Args:
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            render: Whether to render
            verbose: Whether to print progress
            save_video: Whether to save videos
            video_base_dir: Base directory for videos (e.g., "inference/run_data")
                           A timestamped folder will be created under this
            policy_fn: [DEPRECATED] Use model parameter in __init__ instead
            
        Returns:
            Dictionary with aggregate statistics:
                - episodes: List of individual episode results
                - success_rate: Success rate across episodes
                - avg_steps: Average steps per episode
                - avg_reward: Average total reward per episode
                - avg_runtime: Average runtime per episode
                - video_dir: Path to the video directory (if save_video=True)
        """
        model_type = type(self.model).__name__ if self.model else "None"
        print(f"\n{'='*60}")
        print(f"Running {num_episodes} episodes with {model_type} policy")
        print(f"{'='*60}\n")
        
        # Create timestamped video directory if saving videos
        video_dir = None
        if save_video:
            if video_base_dir is None:
                # Default to inference/run_data in current directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                video_base_dir = os.path.join(os.path.dirname(script_dir), "inference", "run_data")
            
            # Create timestamped folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_dir = os.path.join(video_base_dir, timestamp)
            os.makedirs(video_dir, exist_ok=True)
            
            if verbose:
                print(f"[Video] Saving to: {video_dir}\n")
        
        episodes = []
        for ep in range(num_episodes):
            if verbose:
                print(f"Episode {ep + 1}/{num_episodes}")
            
            result = self.run_episode(
                max_steps=max_steps,
                render=render,
                verbose=verbose,
                save_video=save_video,
                video_dir=video_dir,
                episode_idx=ep,
                policy_fn=policy_fn  # For backward compatibility
            )
            episodes.append(result)
            
            if verbose:
                print(f"  → Steps: {result['steps']}, Reward: {sum(result['rewards']):.2f}, Success: {result['success']}\n")
        
        # Compute statistics
        success_rate = np.mean([ep["success"] for ep in episodes])
        avg_steps = np.mean([ep["steps"] for ep in episodes])
        avg_reward = np.mean([sum(ep["rewards"]) for ep in episodes])
        avg_runtime = np.mean([ep["runtime"] for ep in episodes])
        
        results = {
            "episodes": episodes,
            "num_episodes": num_episodes,
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "avg_runtime": avg_runtime,
            "video_dir": video_dir
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY ({num_episodes} episodes)")
        print(f"{'='*60}")
        print(f"Success Rate:     {success_rate*100:.1f}%")
        print(f"Avg Steps:        {avg_steps:.1f}")
        print(f"Avg Total Reward: {avg_reward:.2f}")
        print(f"Avg Runtime:      {avg_runtime:.2f}s")
        if video_dir:
            total_videos = sum(len(ep.get("video_paths", [])) for ep in episodes)
            print(f"Videos Saved:     {total_videos} to {video_dir}")
        print(f"{'='*60}\n")
        
        return results
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

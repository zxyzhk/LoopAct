import sys
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from datetime import datetime

# Add LIBERO submodule to Python path
LIBERO_REPO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "third_party",
    "LIBERO"
)
if os.path.isdir(LIBERO_REPO_PATH) and LIBERO_REPO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_REPO_PATH)

from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

from model_loader import Gr00tLiberoModel


class RunLibero:
    def __init__(self,
                 benchmark_name,
                 task_id,  # specific task you want to test
                 model_path,
                 embodiment_tag,
                 video_dir,
                 mode="baseline",
                 denoising_steps=4,
                 action_horizon=16,
                 img_height=128,
                 img_width=128,
                 max_episode_length=300,
                 fps=10,
                 video_resolution=None,
                 device="cuda"):
        """
        Initialize the runner for GR00T model
        
        Args:
            benchmark_name: name of the libero benchmark
            task_id: specific task ID to test
            model_path: path to checkpoint
            embodiment_tag: embodiment tag for model
            video_dir: directory to save videos
            mode: 'baseline' or 'feedback'
            action_horizon: action chunk size (16 for GR00T)
            img_height: height of camera images
            img_width: width of camera images
            max_episode_length: maximum number of steps per episode
            fps: frames per second for video recording
            device: device to run model on
        """
        self.benchmark_name = benchmark_name
        self.task_id = task_id
        self.img_height = img_height
        self.img_width = img_width
        self.max_episode_length = max_episode_length
        self.fps = fps
        self.video_resolution = video_resolution
        self.device = device

        # Create directories for saving data
        self.video_dir = video_dir
        self.run_data_dir = os.path.join(os.path.dirname(__file__))
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.run_data_dir, exist_ok=True)
        
        self.benchmark = get_benchmark(benchmark_name)()
        self.task_description = self.benchmark.get_task(task_id).language
        self.env = self.create_env()
        
        # Initialize GR00T model
        self.gr00t_model = Gr00tLiberoModel(
            model_path=model_path,
            embodiment_tag=embodiment_tag,
            device=device,
            denoising_steps=denoising_steps,
            mode=mode
        )
        self.gr00t_model.task_description = self.task_description
        self.action_horizon = action_horizon

    def create_env(self):
        """Create the Libero environment for the specific task"""
        env_args = {
            "bddl_file_name": self.benchmark.get_task_bddl_file_path(self.task_id),
            "camera_heights": self.img_height,
            "camera_widths": self.img_width,
            'camera_names': ('agentview', 'robot0_eye_in_hand')
        }
        return OffScreenRenderEnv(**env_args)
    
    def save_video(self, frames, episode_num, success, task_description=None):
        """
        Save frames as a video file
        
        Args:
            frames: List of frames to save
            episode_num: Episode number
            success: Whether the episode was successful
            task_description: Task description for filename
        """
        if not frames:
            return
            
        # Create video filename with task description and episode number
        success_str = "success" if success else "fail"
        
        # Clean task description for filename (remove spaces and special chars)
        if task_description:
            clean_task = task_description.replace(" ", "_").replace(",", "").replace(".", "").replace("'", "").replace('"', "")
            filename = f"{clean_task}_episode_{episode_num}_{success_str}.mp4"
        else:
            filename = f"episode_{episode_num}_{success_str}.mp4"
            
        video_path = os.path.join(self.video_dir, filename)
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
        
        # Write frames
        for frame in frames:
            # IMPORTANT: rotate 180 degrees to match train preprocessing
            frame_rotated = np.ascontiguousarray(frame[::-1, ::-1])
            # Convert from RGB to BGR (OpenCV format)
            frame_bgr = cv2.cvtColor(frame_rotated, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        # Release video writer
        out.release()
        print(f"Saved video to {video_path}")
    
    def save_state_data(self, episode_data, episode_num, success):
        """
        Save state data to a parquet file
        
        Args:
            episode_data: Dictionary containing episode data
            episode_num: Episode number
            success: Whether the episode was successful
        """
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        success_str = "success" if success else "fail"
        
        # Prepare data for DataFrame
        data = []
        for step, (obs, action, reward) in enumerate(zip(
            episode_data['observations'],
            episode_data['actions'],
            episode_data['rewards']
        )):
            # Extract state information
            state_info = {
                'episode': episode_num,
                'step': step,
                'success': success,
                'reward': reward,
                'action': action.tolist(),  # Convert numpy array to list
                'joint_pos': obs['robot0_joint_pos'].tolist(),
                'gripper_qpos': obs['robot0_gripper_qpos'].tolist(),
                'eef_pos': obs['robot0_eef_pos'].tolist(),
                'eef_quat': obs['robot0_eef_quat'].tolist(),
                'timestamp': timestamp
            }
            data.append(state_info)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to parquet file
        filename = f"episode_{episode_num}_{success_str}_{timestamp}.parquet"
        filepath = os.path.join(self.run_data_dir, filename)
        df.to_parquet(filepath)
        print(f"Saved state data to {filepath}")
    
    def run_episode(self, record_video=True):
        """
        Run a single episode for a specific task with the gr00t model
        
        Args:
            record_video: whether to record video frames
            
        Returns:
            success: whether the task was completed successfully
            total_reward: total reward accumulated
            episode_data: dictionary containing episode data
        """
        # Reset environment
        obs = self.env.reset()
        
        # Reset model's frame buffer for new episode (feedback mode only)
        if hasattr(self.gr00t_model, 'reset'):
            self.gr00t_model.reset()

        # Initialize episode data
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'frames': [] if record_video else None
        }
        
        # Initialize action chunk
        self.action_chunk = []
        self.action_chunk_idx = 0
        
        total_reward = 0
        success = False
        # Run episode
        for step in range(self.max_episode_length):
            # Get current observation
            current_obs = obs
            episode_data['observations'].append(current_obs)
            
            if record_video:
                if self.video_resolution is not None:
                    # Render a dedicated high-res agentview frame from the sim.
                    # Inference is unaffected — obs['agentview_image'] stays at native res.
                    frame = self.env.sim.render(
                        camera_name="agentview",
                        width=self.video_resolution,
                        height=self.video_resolution,
                        depth=False,
                    )
                    episode_data['frames'].append(np.ascontiguousarray(frame))
                else:
                    episode_data['frames'].append(obs['agentview_image'])
            
            
            # Get action from GR00T model
            if self.gr00t_model.mode == "feedback":
                # Feedback mode: call every step with the actual step counter.
                # The model returns 1 refined action per call and internally decides
                # when to re-run the full VLM (every 16 steps via time_step % 16 == 0).
                print(f"step: {step} (feedback, time_step={step})")
                action_raw = self.gr00t_model.get_action(current_obs, step, self.task_description)
                if action_raw is None:
                    print(f"[ENV] TIME {step}: no action returned; ending episode early")
                    break
                action = action_raw[0] if action_raw.ndim > 1 else action_raw
            else:
                # Baseline mode: call every action_horizon steps, returns full chunk.
                if step % self.action_horizon == 0 or len(self.action_chunk) == 0 or self.action_chunk_idx >= len(self.action_chunk):
                    inference_idx = step // self.action_horizon
                    print(f"step: {step}, inference_idx: {inference_idx}")
                    action_chunk_raw = self.gr00t_model.get_action(current_obs, inference_idx, self.task_description)
                    if action_chunk_raw is None:
                        self.action_chunk = []
                    else:
                        self.action_chunk = list(action_chunk_raw)
                    self.action_chunk_idx = 0
                    print(f"[ENV] TIME {step}: chunk size {len(self.action_chunk)}")
                    if len(self.action_chunk) == 0:
                        print(f"[ENV] TIME {step}: empty action chunk; ending episode early")
                        break
                action = self.action_chunk[self.action_chunk_idx]
                self.action_chunk_idx += 1
            
            # Execute the action
            step_result = self.env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                truncated = False
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            # Record data
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            total_reward += reward
            
            # Check if task is complete (done = success in Libero)
            if done:
                success = True
                break
        print(success, total_reward)
        
        return success, total_reward, episode_data
    

    def evaluate(self, n_episodes, save_video=True, save_data=False):
        """
        Evaluate the GR00T model over multiple episodes
        
        Args:
            n_episodes: number of episodes to run
            save_video: whether to save videos
            save_data: whether to save episode data
            
        Returns:
            results: dictionary containing evaluation metrics
        """
        successes = []
        rewards = []
        episodes = []
        
        print(f"\nStarting evaluation for task {self.task_id}: {self.task_description}")
        print(f"Running {n_episodes} episodes...")
        
        for episode in tqdm(range(n_episodes), desc="Episodes"):
            success, reward, episode_data = self.run_episode(record_video=save_video)
            successes.append(success)
            rewards.append(reward)
            episodes.append(episode_data)
            
            # Save video for this episode
            if save_video and episode_data['frames']:
                self.save_video(episode_data['frames'], episode, success, self.task_description)
            
            if save_data:
                self.save_state_data(episode_data, episode, success)
            
            # Log current results
            current_success_rate = np.mean(successes)
            print(f"Episode {episode + 1}/{n_episodes}: Success={success}, Reward={reward:.2f}, Current Success Rate={current_success_rate:.2f}")
        
        final_success_rate = np.mean(successes)
        final_avg_reward = np.mean(rewards)
        
        results = {
            'task_id': self.task_id,
            'task_description': self.task_description,
            'success_rate': final_success_rate,
            'average_reward': final_avg_reward,
            'total_episodes': n_episodes,
            'total_successes': sum(successes),
            'episodes': episodes
        }
        
        print(f"\nTask {self.task_id} Final Results:")
        print(f"  Success Rate: {final_success_rate:.2f} ({sum(successes)}/{n_episodes})")
        print(f"  Average Reward: {final_avg_reward:.2f}")
        
        return results

    def run_all_tasks(self, n_runs_per_task=2, save_video=True, save_data=False, num_tasks=10):
        """
        Run all tasks from task_id 0 to num_tasks-1, each task n_runs_per_task times
        
        Args:
            n_runs_per_task: number of runs per task
            save_video: whether to save videos
            save_data: whether to save episode data
            num_tasks: number of tasks to run (default 10)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = {}
        total_episodes = 0
        total_successes = 0
        
        print(f"Starting evaluation of {num_tasks} tasks, {n_runs_per_task} runs each...")
        print(f"Total episodes to run: {num_tasks * n_runs_per_task}")
        
        for task_id in range(num_tasks):
            print(f"\n{'='*60}")
            print(f"Running Task {task_id}")
            print(f"{'='*60}")
            
            # Update task for this iteration
            self.task_id = task_id
            self.env = self.create_env()
            self.task_description = self.benchmark.get_task(task_id).language
            self.gr00t_model.task_description = self.task_description
            
            task_results = self.evaluate(
                n_episodes=n_runs_per_task, 
                save_video=save_video, 
                save_data=save_data
            )
            
            all_results[f'task_{task_id}'] = task_results
            
            # Update overall statistics
            total_episodes += task_results['total_episodes']
            total_successes += task_results['total_successes']
            
            print(f"\nTask {task_id} Summary:")
            print(f"  Success Rate: {task_results['success_rate']:.2f} ({task_results['total_successes']}/{task_results['total_episodes']})")
            print(f"  Average Reward: {task_results['average_reward']:.2f}")
            print(f"  Overall Success Rate So Far: {total_successes}/{total_episodes} ({total_successes/total_episodes:.2f})")
        
        # Calculate final overall statistics
        overall_success_rate = total_successes / total_episodes if total_episodes > 0 else 0
        
        all_results['overall'] = {
            'total_episodes': total_episodes,
            'total_successes': total_successes,
            'overall_success_rate': overall_success_rate,
            'num_tasks': num_tasks,
            'runs_per_task': n_runs_per_task
        }
        
        return all_results


if __name__ == "__main__":
    # =============================================================================
    # CONFIGURATION
    # =============================================================================
    # Checkpoint paths are relative to the project root.
    CHECKPOINT_MAP = {
        "baseline": "checkpoints/cp_libero_baseline/checkpoint-20000",
        "feedback": "checkpoints/cp_libero_obs",
    }

    MODE = "feedback"  # "baseline" or "feedback"
    CHECKPOINT_PATH = CHECKPOINT_MAP[MODE]

    # Benchmark / task selection
    BENCHMARK_NAME = "LIBERO_OBJECT"  # e.g. "LIBERO_GOAL", "LIBERO_OBJECT"
    RUN_ALL_TASKS = True   # False = evaluate a single task; True = run_all_tasks
    TASK_ID = 0            # Only used when RUN_ALL_TASKS = False
    NUM_TASKS = 10         # Only used when RUN_ALL_TASKS = True
    NUM_EPISODES = 1     # Episodes per task

    SAVE_VIDEO = True
    VIDEO_RESOLUTION = 1024  # Agentview video rendered at this size; set to None to use the obs resolution. Inference always sees the native obs resolution.
    SAVE_DATA = False
    MAX_EPISODE_LENGTH = 300
    TIMING = False          # Per-component timing logs from gr00t_model.py
    # =============================================================================

    if TIMING:
        os.environ["GR00T_TIMING"] = "1"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiments_dir = os.path.dirname(__file__)
    video_dir = os.path.join(experiments_dir, "run_data", f"videos_{timestamp}")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    checkpoint_path = os.path.join(project_root, CHECKPOINT_PATH)

    print(f"=" * 70)
    print(f"Running in {MODE.upper()} mode")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"=" * 70)

    runner = RunLibero(
        benchmark_name=BENCHMARK_NAME,
        max_episode_length=MAX_EPISODE_LENGTH,
        task_id=TASK_ID,
        model_path=checkpoint_path,
        embodiment_tag="libero_arm",
        video_dir=video_dir,
        mode=MODE,
        denoising_steps=4,
        action_horizon=16,
        video_resolution=VIDEO_RESOLUTION if SAVE_VIDEO else None,
        device="cuda"
    )

    if RUN_ALL_TASKS:
        print("Starting evaluation of all tasks...")
        all_results = runner.run_all_tasks(n_runs_per_task=NUM_EPISODES, save_video=SAVE_VIDEO, save_data=SAVE_DATA, num_tasks=NUM_TASKS)
    else:
        all_results = runner.evaluate(n_episodes=NUM_EPISODES, save_video=SAVE_VIDEO, save_data=SAVE_DATA)
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL EVALUATION SUMMARY")
    print("="*70)
    
    if 'overall' in all_results:
        overall = all_results['overall']
        print(f"OVERALL RESULTS:")
        print(f"  Total Episodes: {overall['total_episodes']}")
        print(f"  Total Successes: {overall['total_successes']}")
        print(f"  Overall Success Rate: {overall['overall_success_rate']:.2f} ({overall['total_successes']}/{overall['total_episodes']})")
        print(f"  Tasks Evaluated: {overall['num_tasks']}")
        print(f"  Runs per Task: {overall['runs_per_task']}")
        print()
    
    print("PER-TASK RESULTS:")
    for task_key, task_data in all_results.items():
        if task_key != 'overall':
            print(f"  {task_key}: {task_data['task_description']} - Success Rate = {task_data['success_rate']:.2f} ({task_data['total_successes']}/{task_data['total_episodes']}), Avg Reward = {task_data['average_reward']:.2f}")
    
    print(f"\nVideos saved in: {runner.video_dir}")
    print(f"Data saved in: {runner.run_data_dir}")
    print(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

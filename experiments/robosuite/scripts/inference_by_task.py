#!/usr/bin/env python3
"""
Run inference on a single task and report the success rate.

Usage:
    python3 inference_by_task.py
    python3 inference_by_task.py --mode baseline --task toy_red --episodes 20

Configuration:
    Edit the constants below. The command-line flags are optional overrides for
    the settings that change most often; anything not passed keeps its constant.
"""

import argparse
import os
import sys
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_robosuite_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _robosuite_dir)

from tasks.get_task import get_task
from inference.inference_wrapper import InferenceWrapper
from inference.seed_tracker import SeedTracker


# =============================================================================
# CONFIGURATION
# =============================================================================
# Checkpoint paths are relative to the project root.
CHECKPOINT_MAP = {
    "baseline": "checkpoints/cp_sim_baseline",
    "feedback": "checkpoints/cp_sim_obs",
}

MODE = "feedback"  # "baseline" or "feedback"

# See tasks/get_task.py for the full list.
# seen:   "toy_red", "toy_green", "drop_ball"
# unseen: "catch_block", "toy_red_speed", "drop_ball_color", "drop_ball_speed"
TASK = "toy_red"

NUM_EPISODES = 10
SAVE_VIDEO = True
VIDEO_RESOLUTION = 1024  # Wrist video render size. Higher = sharper for paper figures; None matches the obs resolution. Inference is unaffected — the model always sees the task's native resolution.
EARLY_TERMINATION = True  # Stop episode early on success
SUCCESS_CHECK_FREQ = 10  # Check for success every N steps
TIMING = False  # Per-component timing logs from gr00t_model.py

CHECKPOINT_PATH = CHECKPOINT_MAP[MODE]
TASK_CONFIG, TASK_SUCCESS = get_task(TASK)


def run_inference():
    """Run inference for the configured task and calculate success rate."""
    
    print("\n" + "="*70)
    print(f"RUNNING INFERENCE FOR TASK: {TASK_CONFIG.task_name}")
    print("="*70)
    print(f"Description: {TASK_CONFIG.task_description}")
    print(f"Environment: {TASK_CONFIG.env_name}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Mode: {MODE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Save Video: {SAVE_VIDEO}")
    print(f"Early Termination: {EARLY_TERMINATION}")
    print("="*70 + "\n")
    
    if TIMING:
        os.environ["GR00T_TIMING"] = "1"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = os.path.join(_robosuite_dir, "inference", "run_data", f"{TASK_CONFIG.task_name}_{timestamp}")

    os.makedirs(video_dir, exist_ok=True)
    print(f"[INFO] Videos will be saved to: {video_dir}/\n")

    # Initialize seed tracker
    seed_tracker = SeedTracker(save_dir=video_dir, task_name=TASK_CONFIG.task_name)
    
    # =========================================================================
    # MODEL LOADING
    # =========================================================================
    from inference.model_loader import Gr00tRobosuiteModel

    model = Gr00tRobosuiteModel(
        model_path=CHECKPOINT_PATH,
        embodiment_tag="libero_arm",
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu",
        mode=MODE,
        action_horizon=16,
    )
    print(f"[INFO] GR00T model loaded successfully\n")
    # =========================================================================
    
    if SAVE_VIDEO and VIDEO_RESOLUTION is not None:
        print(f"[INFO] Saving wrist video at {VIDEO_RESOLUTION}x{VIDEO_RESOLUTION} (observations stay at {TASK_CONFIG.camera_height}x{TASK_CONFIG.camera_width})")

    # Create inference wrapper with task-specific settings
    wrapper = InferenceWrapper(
        env_name=TASK_CONFIG.env_name,
        robot=TASK_CONFIG.robot,
        controller=TASK_CONFIG.controller,
        control_freq=TASK_CONFIG.control_freq,
        horizon=TASK_CONFIG.max_episode_steps,
        camera_names=TASK_CONFIG.camera_names,
        camera_heights=TASK_CONFIG.camera_height,
        camera_widths=TASK_CONFIG.camera_width,
        video_resolution=VIDEO_RESOLUTION if SAVE_VIDEO else None,
        randomize_init=TASK_CONFIG.randomize_init,
        target_joint_name=TASK_CONFIG.target_joint_name,
        target_speed_y=TASK_CONFIG.target_speed_y,
        target_axis=TASK_CONFIG.target_axis,
        mover_type=TASK_CONFIG.mover_type,
        decision_interval=TASK_CONFIG.decision_interval,
        speeds=TASK_CONFIG.speeds,
        inference_patterns=getattr(TASK_CONFIG, 'inference_patterns', None),
        task_name=TASK_CONFIG.task_name,
        task_success=TASK_SUCCESS,
        task_description=TASK_CONFIG.task_description,  # Pass task description for model
        model=model,  # Pass your model here
        **({'slope_angle': TASK_CONFIG.slope_angle} if hasattr(TASK_CONFIG, 'slope_angle') else {}),
        **({'cone_end': TASK_CONFIG.cone_end} if hasattr(TASK_CONFIG, 'cone_end') else {}),
    )
    
    # Run episodes and track success
    successes = []
    scores = []
    
    for episode_idx in range(NUM_EPISODES):
        print(f"\n[Episode {episode_idx + 1}/{NUM_EPISODES}]")
        TASK_SUCCESS.debug_verbose = False

        # Generate seed for this episode (used for random mover)
        seed = seed_tracker.generate_seed(episode_idx)
        # Pattern ID cycles through available patterns (used for pattern mover)
        pattern_id = episode_idx  # Will wrap around via modulo in PatternMover
        print(f"  Seed: {seed}, Pattern ID: {pattern_id % 10}")

        # Run one episode with early termination on success
        # pattern_id cycles through patterns 0, 1, 2, ... and repeats
        results = wrapper.run_episode(
            save_video=SAVE_VIDEO,
            video_dir=video_dir if SAVE_VIDEO else None,
            episode_idx=episode_idx,
            early_termination=EARLY_TERMINATION,
            success_check_freq=SUCCESS_CHECK_FREQ,
            mover_seed=seed,
            pattern_id=episode_idx,  # Cycles through patterns one by one
        )

        # Check success using the task's success conditions
        env = wrapper.env
        TASK_SUCCESS.debug_verbose = True
        success, score, details = env.check_success(verbose=False)
        TASK_SUCCESS.debug_verbose = False

        successes.append(success)
        scores.append(score)

        # Print episode result with step count
        status = "✓ SUCCESS" if success else "✗ FAILURE"
        steps_taken = results.get("steps", 0)
        print(f"  Result: {status} (score: {int(round(score))}, steps: {steps_taken})")

        # Get speed history from mover
        speed_history = []
        if wrapper.target_mover:
            stats = wrapper.target_mover.get_stats()
            speed_history = stats.get("speed_history", [])
            print(f"  Speed pattern: {speed_history}")

        # Record result with seed and speed history
        seed_tracker.add_episode_result(episode_idx, seed, success, score, steps_taken, speed_history)
    
    # Calculate and print statistics
    success_rate = np.mean(successes)
    avg_score = np.mean(scores)
    
    print("\n" + "="*70)
    print("INFERENCE RESULTS")
    print("="*70)
    print(f"Task: {TASK_CONFIG.task_description}")
    print(f"Mode: {MODE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Success Rate: {success_rate:.2%} ({sum(successes)}/{NUM_EPISODES})")
    print(f"Average Score: {avg_score:.2f}")
    print("-"*70)
    print("Per-episode results:")
    for i, (s, sc) in enumerate(zip(successes, scores)):
        status = "✓" if s else "✗"
        print(f"  Episode {i+1}: {status} (score: {int(round(sc))})")
    print("="*70 + "\n")
    
    if SAVE_VIDEO:
        # Count saved video files
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        print(f"[INFO] Saved {len(video_files)} videos to: {video_dir}/")
        print(f"       Files: episode_0000_wrist_success.mp4, episode_0001_wrist_fail.mp4, ...\n")

    # Save seeds to file
    seed_tracker.save(extra_info={
        "mode": MODE,
        "checkpoint": CHECKPOINT_PATH,
        "task_description": TASK_CONFIG.task_description,
        "success_rate": f"{success_rate:.2%}",
    })

    # Clean up
    wrapper.close()
    
    return {
        "task_name": TASK_CONFIG.task_name,
        "num_episodes": NUM_EPISODES,
        "successes": successes,
        "scores": scores,
        "success_rate": success_rate,
        "avg_score": avg_score,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single task.")
    parser.add_argument("--mode", choices=sorted(CHECKPOINT_MAP), help=f"default: {MODE}")
    parser.add_argument("--task", help=f"default: {TASK}")
    parser.add_argument("--episodes", type=int, help=f"default: {NUM_EPISODES}")
    parser.add_argument("--checkpoint", help="override the checkpoint for the selected mode")
    parser.add_argument("--no-video", action="store_true", help="disable video saving")
    args = parser.parse_args()

    if args.mode:
        MODE = args.mode
        CHECKPOINT_PATH = CHECKPOINT_MAP[MODE]
    if args.checkpoint:
        CHECKPOINT_PATH = args.checkpoint
    if args.task:
        TASK = args.task
        TASK_CONFIG, TASK_SUCCESS = get_task(TASK)
    if args.episodes:
        NUM_EPISODES = args.episodes
    if args.no_video:
        SAVE_VIDEO = False

    run_inference()

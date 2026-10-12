#!/usr/bin/env python3
"""
Run inference across multiple tasks and summarize success rates.

Usage:
    python3 inference_all.py
    python3 inference_all.py --mode baseline feedback --episodes 10

Configuration:
    Edit the constants below. The command-line flags are optional overrides for
    the settings that change most often; anything not passed keeps its constant.
"""

import argparse
import os
import sys
import numpy as np
from datetime import datetime

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

# A list runs each mode in turn, e.g. ["baseline", "feedback"] to compare them.
MODE = ["feedback"]

# See tasks/get_task.py for the full list.
# seen:   "toy_red", "toy_green", "drop_ball"
# unseen: "catch_block", "toy_red_speed", "drop_ball_color", "drop_ball_speed"
INFERENCE_TASKS = [
    "toy_red",
    "toy_green",
    "drop_ball",
    "catch_block",
    "toy_red_speed",
    "drop_ball_color",
    "drop_ball_speed",
]

NUM_EPISODES = 10
SAVE_VIDEO = False
VIDEO_RESOLUTION = 1024  # Agentview video render size. Higher = sharper for paper figures; None matches the obs resolution. Inference is unaffected.
EARLY_TERMINATION = True  # Stop episode early on success
SUCCESS_CHECK_FREQ = 10  # Check for success every N steps
TIMING = False  # Per-component timing logs from gr00t_model.py


def run_task(task_name, model, mode, checkpoint_path, run_dir, num_episodes, save_video, video_resolution, early_termination, success_check_freq):
    """Run all episodes for one task into run_dir. Returns (success_rate, n_success, n_total)."""
    task_config, task_success = get_task(task_name)

    os.makedirs(run_dir, exist_ok=True)

    seed_tracker = SeedTracker(save_dir=run_dir, task_name=task_config.task_name)

    wrapper = InferenceWrapper(
        env_name=task_config.env_name,
        robot=task_config.robot,
        controller=task_config.controller,
        control_freq=task_config.control_freq,
        horizon=task_config.max_episode_steps,
        camera_names=task_config.camera_names,
        camera_heights=task_config.camera_height,
        camera_widths=task_config.camera_width,
        randomize_init=task_config.randomize_init,
        target_joint_name=task_config.target_joint_name,
        target_speed_y=task_config.target_speed_y,
        target_axis=task_config.target_axis,
        mover_type=task_config.mover_type,
        decision_interval=task_config.decision_interval,
        speeds=task_config.speeds,
        inference_patterns=getattr(task_config, 'inference_patterns', None),
        task_name=task_config.task_name,
        task_success=task_success,
        task_description=task_config.task_description,
        model=model,
        video_resolution=video_resolution if save_video else None,
        **({'slope_angle': task_config.slope_angle} if hasattr(task_config, 'slope_angle') else {}),
        **({'cone_end': task_config.cone_end} if hasattr(task_config, 'cone_end') else {}),
    )

    successes = []
    scores = []

    for episode_idx in range(num_episodes):
        seed = seed_tracker.generate_seed(episode_idx)
        task_success.debug_verbose = False

        results = wrapper.run_episode(
            save_video=save_video,
            video_dir=run_dir if save_video else None,
            episode_idx=episode_idx,
            early_termination=early_termination,
            success_check_freq=success_check_freq,
            mover_seed=seed,
            pattern_id=episode_idx,
        )

        success, score, _ = wrapper.env.check_success(verbose=False)
        successes.append(success)
        scores.append(score)

        steps_taken = results.get("steps", 0)
        status = "✓" if success else "✗"
        print(f"  [{task_name}] Ep {episode_idx + 1:2d}/{num_episodes}: {status}  steps={steps_taken}")

        speed_history = []
        if wrapper.target_mover:
            speed_history = wrapper.target_mover.get_stats().get("speed_history", [])

        seed_tracker.add_episode_result(episode_idx, seed, success, score, steps_taken, speed_history)

    success_rate = float(np.mean(successes))
    seed_tracker.save(extra_info={
        "mode": mode,
        "checkpoint": checkpoint_path,
        "task_description": task_config.task_description,
        "success_rate": f"{success_rate:.2%}",
    })
    wrapper.close()

    return success_rate, sum(successes), num_episodes


def _print_summary(mode_results):
    """Print a summary table. mode_results: {mode: {task: (sr, n, total)}}"""
    col_task = 25
    col_mode = 14
    modes = list(mode_results.keys())
    all_tasks = list(next(iter(mode_results.values())).keys())

    header = f"  {'Task':<{col_task}}" + "".join(f" {m:>{col_mode}}" for m in modes)
    print(f"\n{'='*55}")
    print(f"  RESULTS  |  " + "  ".join(modes))
    print(f"{'='*55}")
    print(header)
    print(f"  {'-'*col_task}" + "".join(f" {'-'*col_mode}" for _ in modes))
    for task_name in all_tasks:
        row = f"  {task_name:<{col_task}}"
        for m in modes:
            sr, n_success, n_total = mode_results[m][task_name]
            row += f" {f'{sr:.1%} ({n_success}/{n_total})':>{col_mode}}"
        print(row)
    print(f"{'='*55}\n")


def run_all():
    if TIMING:
        os.environ["GR00T_TIMING"] = "1"

    modes = MODE if isinstance(MODE, list) else [MODE]

    from inference.model_loader import Gr00tRobosuiteModel

    # One integrated folder for this whole run: inference_all_{ts}/{task}/{mode}/
    run_all_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_all_dir = os.path.join(_robosuite_dir, "inference", "run_data", f"inference_all_{run_all_timestamp}")
    os.makedirs(run_all_dir, exist_ok=True)
    print(f"[INFO] inference_all results dir: {run_all_dir}")

    mode_results = {}
    for mode in modes:
        checkpoint_path = CHECKPOINT_MAP[mode]
        model = Gr00tRobosuiteModel(
            model_path=checkpoint_path,
            embodiment_tag="libero_arm",
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu",
            mode=mode,
            action_horizon=16,
        )
        print(f"\n[INFO] Model loaded: {checkpoint_path}  mode={mode}\n")

        results = {}
        for task_name in INFERENCE_TASKS:
            print(f"\n{'='*55}")
            print(f"  TASK: {task_name}  mode={mode}  ({NUM_EPISODES} episodes)")
            print(f"{'='*55}")
            run_dir = os.path.join(run_all_dir, task_name, mode)
            sr, n_success, n_total = run_task(
                task_name, model, mode, checkpoint_path, run_dir,
                NUM_EPISODES, SAVE_VIDEO, VIDEO_RESOLUTION, EARLY_TERMINATION, SUCCESS_CHECK_FREQ,
            )
            results[task_name] = (sr, n_success, n_total)
            print(f"  → {task_name}: {sr:.1%} ({n_success}/{n_total})")

        mode_results[mode] = results

    _print_summary(mode_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference across multiple tasks.")
    parser.add_argument("--mode", nargs="+", choices=sorted(CHECKPOINT_MAP),
                        help=f"one or more modes, run in turn. default: {' '.join(MODE)}")
    parser.add_argument("--tasks", nargs="+", help=f"default: {' '.join(INFERENCE_TASKS)}")
    parser.add_argument("--episodes", type=int, help=f"default: {NUM_EPISODES}")
    parser.add_argument("--no-video", action="store_true", help="disable video saving")
    args = parser.parse_args()

    if args.mode:
        MODE = args.mode
    if args.tasks:
        INFERENCE_TASKS = args.tasks
    if args.episodes:
        NUM_EPISODES = args.episodes
    if args.no_video:
        SAVE_VIDEO = False

    run_all()

#!/usr/bin/env python3
"""
Collect consecutive teleoperation demos with save/discard controls.
Uses EnvRender.collect_demos for consistency.
Usage:
    python3 collect_demos.py 
    (but remember to change the task configuration)
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_robosuite_dir = os.path.dirname(_script_dir)
_project_root = os.path.dirname(_robosuite_dir)
sys.path.insert(0, _robosuite_dir)

from tasks.get_task import get_task
from env_render import EnvRender

# Change this to switch tasks (see tasks/get_task.py for options)
TASK_CONFIG, TASK_SUCCESS = get_task("toy_can") # random_block, random_block_2d, toy_cart, toy_can, toy_red, toy_green, catch_carrot, drop_ball

# Demo data folder (relative to experiments directory)
# Note: _project_root is experiments/, not LoopAct root
DEMO_DATA_DIR = os.path.join(_project_root, "demo_data")


def main():
    parser = argparse.ArgumentParser(description="Collect consecutive teleop demos to HDF5.")
    parser.add_argument("--output-dir", default=None, 
                        help="Output directory (default: demo_data/task_name_YYYYMMDD_HHMMSS)")
    parser.add_argument("--prefix", default="demo", help="Demo filename prefix")
    args = parser.parse_args()

    print("\n" + "="*70)
    print(f"COLLECTING DEMOS FOR TASK: {TASK_CONFIG.task_name}")
    print("="*70)
    print(f"Description: {TASK_CONFIG.task_description}")
    print(f"Environment: {TASK_CONFIG.env_name}")
    print(f"Difficulty: {TASK_CONFIG.difficulty}")
    print("="*70 + "\n")

    # Setup based on save_data config
    if TASK_CONFIG.save_data:
        # Create output directory with timestamp inside demo_data folder
        if args.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = os.path.join(DEMO_DATA_DIR, f"{TASK_CONFIG.task_name}_{timestamp}")
        
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[INFO] Saving demos to: {args.output_dir}/")
        placeholder_path = os.path.join(args.output_dir, f"{args.prefix}_0.hdf5")
    else:
        print("[INFO] Practice mode - data saving disabled")
        placeholder_path = None

    # Use task config for all settings
    env_kwargs = TASK_CONFIG.to_env_kwargs()
    
    runner = EnvRender(
        **env_kwargs,
        task_success=TASK_SUCCESS,  # Pass TaskSuccess instance
        record_path=placeholder_path,
        record_stop_key=None,  # Disabled for collect_demos (uses save/discard keys)
    )
    
    if TASK_CONFIG.save_data:
        # Use fixed keys for save/discard
        runner.collect_demos(
            output_dir=args.output_dir,
            prefix=args.prefix,
            save_key="l",
            discard_key="k",
        )
    else:
        # Practice mode without recording
        runner.run()


if __name__ == "__main__":
    main()

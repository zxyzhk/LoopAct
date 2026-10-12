#!/usr/bin/env python3
"""
Seed Tracker - Save and load random seeds for reproducible inference runs.

Saves seeds to a human-readable file so experiments can be reproduced exactly.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional


class SeedTracker:
    """Track and save random seeds for reproducible experiments."""

    def __init__(self, save_dir: str, task_name: str):
        """
        Initialize seed tracker.

        Args:
            save_dir: Directory to save seed file
            task_name: Name of the task (used in filename)
        """
        self.save_dir = save_dir
        self.task_name = task_name
        self.seeds: List[int] = []
        self.episode_data: List[Dict] = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_seed(self, episode_idx: int) -> int:
        """
        Generate and store a random seed for an episode.

        Args:
            episode_idx: Episode index

        Returns:
            Generated seed
        """
        seed = np.random.randint(0, 2**31 - 1)
        self.seeds.append(seed)
        return seed

    def add_episode_result(
        self,
        episode_idx: int,
        seed: int,
        success: bool,
        score: float,
        steps: int,
        speed_history: Optional[List[float]] = None,
    ):
        """
        Record episode result with its seed and speed pattern.

        Args:
            episode_idx: Episode index
            seed: Seed used for this episode
            success: Whether episode succeeded
            score: Episode score
            steps: Steps taken
            speed_history: List of speeds chosen during episode
        """
        self.episode_data.append({
            "episode": episode_idx,
            "seed": seed,
            "success": success,
            "score": score,
            "steps": steps,
            "speed_history": speed_history or [],
        })

    def save(self, extra_info: Optional[Dict] = None):
        """
        Save seeds and results to a readable file.

        Args:
            extra_info: Additional info to save (mode, checkpoint, etc.)
        """
        os.makedirs(self.save_dir, exist_ok=True)

        filename = f"seeds_{self.task_name}_{self.timestamp}.json"
        filepath = os.path.join(self.save_dir, filename)

        data = {
            "task_name": self.task_name,
            "timestamp": self.timestamp,
            "num_episodes": len(self.episode_data),
            "seeds": self.seeds,
            "episodes": self.episode_data,
        }

        if extra_info:
            data["config"] = extra_info

        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)

        # Also save a simple text version for quick viewing
        txt_filename = f"seeds_{self.task_name}_{self.timestamp}.txt"
        txt_filepath = os.path.join(self.save_dir, txt_filename)

        with open(txt_filepath, "w") as f:
            f.write(f"Task: {self.task_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            if extra_info:
                f.write(f"Mode: {extra_info.get('mode', 'N/A')}\n")
                f.write(f"Checkpoint: {extra_info.get('checkpoint', 'N/A')}\n")
            f.write(f"\n{'='*80}\n")

            for ep in self.episode_data:
                success_str = "SUCCESS" if ep["success"] else "FAIL"
                f.write(f"\nEpisode {ep['episode']}: {success_str} (score: {ep['score']:.1f}, steps: {ep['steps']})\n")
                f.write(f"  Seed: {ep['seed']}\n")
                if ep.get("speed_history"):
                    # Format speed history nicely (handles scalar, list, and ndarray entries)
                    speeds = ep["speed_history"]
                    flat_speeds = []
                    for s in speeds:
                        if hasattr(s, "flat"):
                            flat_speeds.extend(float(v) for v in s.flat)
                        elif isinstance(s, (list, tuple)):
                            flat_speeds.extend(float(v) for v in s)
                        else:
                            flat_speeds.append(float(s))
                    speed_str = ", ".join([f"{v:.3f}" for v in flat_speeds])
                    f.write(f"  Speed pattern: [{speed_str}]\n")

            f.write(f"\n{'='*80}\n")
            success_count = sum(1 for ep in self.episode_data if ep["success"])
            f.write(f"Success Rate: {success_count}/{len(self.episode_data)}\n")

        print(f"[SeedTracker] Saved seeds to: {filepath}")
        print(f"[SeedTracker] Saved readable summary to: {txt_filepath}")

        return filepath, txt_filepath

    @staticmethod
    def load_seeds(filepath: str) -> List[int]:
        """
        Load seeds from a saved file.

        Args:
            filepath: Path to JSON seed file

        Returns:
            List of seeds
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return data["seeds"]

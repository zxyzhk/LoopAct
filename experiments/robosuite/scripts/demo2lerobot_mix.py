"""
demo2lerobot_mix.py - Merge multiple demo data folders into one LeRobot dataset.

Usage:
    1. Set DEMO_FOLDER_NAMES to the list of folder names under demo_data/
    2. Run: python demo2lerobot_mix.py

Each folder is treated as a separate task. All demos are merged into a single
sequential dataset with continuous episode indices.
"""

import os
import h5py
import json
import pandas as pd
import numpy as np
import glob
import cv2
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).parent                # experiments/robosuite/scripts/
_experiments_dir = _script_dir.parent.parent       # experiments/
_demo_data_dir = _experiments_dir / "demo_data"

# List every folder you want to merge (relative to demo_data/)
DEMO_FOLDER_NAMES = [
    "can_10",
    "can_12",
    "can_14",
    "can_16",
]

# Task name used in metadata (set None to auto-derive from folder names)
TASK_NAME = "pick_up_the_can"

# Output dataset name (auto-generated from folder names)
def _normalize_name(name: str) -> str:
    return name.strip().replace(" ", "_")

if TASK_NAME:
    TASK_NAMES = [_normalize_name(TASK_NAME)]
    DATASET_NAME_OUTPUT = f"{_normalize_name(TASK_NAME)}_data"
else:
    TASK_NAMES = [_normalize_name(n) for n in DEMO_FOLDER_NAMES]
    DATASET_NAME_OUTPUT = f"{'_'.join(TASK_NAMES)}_data"
DATA_OUTPUT_LOCATION = str(_demo_data_dir)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------
class Demo2LeRobotMix:
    """Merge multiple demo folders into a single LeRobot dataset."""

    def __init__(self, folder_names=None, output_location=None):
        self.folder_names = folder_names or DEMO_FOLDER_NAMES
        self.output_location = output_location or DATA_OUTPUT_LOCATION
        self.fps = 10

        # Build task list and discover all demo files
        self.task_names = TASK_NAMES
        self.demo_entries = self._discover_demos()

    def _discover_demos(self):
        """Scan each folder and return a flat list of (task_index, task_display, demo_file)."""
        entries = []
        for folder_index, folder_name in enumerate(self.folder_names):
            data_dir = str(_demo_data_dir / folder_name)
            demo_files = sorted(
                glob.glob(os.path.join(data_dir, "demo_*.hdf5")),
                key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]),
            )
            # When TASK_NAME is set, all folders share task_index=0
            task_index = 0 if TASK_NAME else folder_index
            task_display = self.task_names[task_index].replace("_", " ")
            for demo_file in demo_files:
                entries.append((task_index, task_display, demo_file))
        return entries

    # ---- data ----
    def process_data(self, demo_idx, demo_file, task_index):
        parquet_name = f"episode_{demo_idx:06d}.parquet"

        with h5py.File(demo_file, "r") as f:
            actions = f["demo_0/actions"][:]
            ee_pos = f["demo_0/obs/ee_pos"][:]
            ee_ori = f["demo_0/obs/ee_ori"][:]
            ee_states = f["demo_0/obs/ee_states"][:]
            gripper_states = f["demo_0/obs/gripper_states"][:]
            joint_states = f["demo_0/obs/joint_states"][:]

            T = actions.shape[0]
            records = []
            for t in range(T):
                obs_state = np.concatenate([
                    ee_pos[t], ee_ori[t], ee_states[t],
                    gripper_states[t], joint_states[t],
                ])
                records.append({
                    "action": actions[t].tolist(),
                    "observation.state": obs_state.tolist(),
                    "timestamp": t * (1 / self.fps),
                    "episode_index": demo_idx,
                    "task_index": task_index,
                    "annotation.human.action.task_description": task_index,
                })

        chunk_dir = os.path.join(self.output_location, DATASET_NAME_OUTPUT, "data", "chunk-000")
        os.makedirs(chunk_dir, exist_ok=True)
        out_path = os.path.join(chunk_dir, parquet_name)
        pd.DataFrame(records).to_parquet(out_path, index=False)
        print(f"  Wrote {len(records)} timesteps -> {out_path}")

    # ---- videos ----
    def process_videos(self, demo_idx, demo_file):
        video_name = f"episode_{demo_idx:06d}.mp4"

        with h5py.File(demo_file, "r") as f:
            for obs_key, folder_key in [
                ("agentview_rgb",    "observation.images.agentview_rgb"),
                ("eye_in_hand_rgb",  "observation.images.eye_in_hand_rgb"),
            ]:
                video_dir = os.path.join(
                    self.output_location, DATASET_NAME_OUTPUT,
                    "videos", "chunk-000", folder_key,
                )
                os.makedirs(video_dir, exist_ok=True)
                out_path = os.path.join(video_dir, video_name)

                video_data = f[f"demo_0/obs/{obs_key}"][:]
                T, H, W, C = video_data.shape
                writer = cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps, (W, H),
                )
                for t in range(T):
                    writer.write(video_data[t][..., ::-1])  # RGB -> BGR
                writer.release()
                print(f"  Saved {obs_key} video -> {out_path}")

    # ---- metadata ----
    def calculate_obs_indices(self):
        demo_file = self.demo_entries[0][2]
        with h5py.File(demo_file, "r") as f:
            dims = {
                "ee_pos":          f["demo_0/obs/ee_pos"].shape[1],
                "ee_ori":          f["demo_0/obs/ee_ori"].shape[1],
                "ee_states":       f["demo_0/obs/ee_states"].shape[1],
                "gripper_states":  f["demo_0/obs/gripper_states"].shape[1],
                "joint_states":    f["demo_0/obs/joint_states"].shape[1],
                "action":          f["demo_0/actions"].shape[1],
            }

        idx, result = 0, {}
        for key in ["ee_pos", "ee_ori", "ee_states", "gripper_states", "joint_states"]:
            result[f"{key}_start"] = idx
            idx += dims[key]
            result[f"{key}_end"] = idx
        result["action_single_start"] = 0
        result["action_single_end"] = dims["action"]
        return result

    def process_metadata(self, episode_entries):
        meta_dir = os.path.join(self.output_location, DATASET_NAME_OUTPUT, "meta")
        os.makedirs(meta_dir, exist_ok=True)

        # tasks.jsonl
        tasks_path = os.path.join(meta_dir, "tasks.jsonl")
        with open(tasks_path, "w") as f:
            for task_index, task_name in enumerate(self.task_names):
                f.write(json.dumps({
                    "task_index": task_index,
                    "task": task_name.replace("_", " "),
                }) + "\n")
        print(f"Saved tasks        -> {tasks_path}")

        # episodes.jsonl
        episodes_path = os.path.join(meta_dir, "episodes.jsonl")
        with open(episodes_path, "w") as f:
            for entry in episode_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved episodes     -> {episodes_path}")

        # modality.json
        indices = self.calculate_obs_indices()
        modality = {
            "state": {
                "ee_pos":          {"start": indices["ee_pos_start"],          "end": indices["ee_pos_end"]},
                "ee_ori":          {"start": indices["ee_ori_start"],          "end": indices["ee_ori_end"]},
                "ee_states":       {"start": indices["ee_states_start"],       "end": indices["ee_states_end"]},
                "gripper_states":  {"start": indices["gripper_states_start"],  "end": indices["gripper_states_end"]},
                "joint_states":    {"start": indices["joint_states_start"],    "end": indices["joint_states_end"]},
            },
            "action": {
                "all_actions": {"start": indices["action_single_start"], "end": indices["action_single_end"]},
            },
            "video": {
                "agentview_rgb":    {"original_key": "observation.images.agentview_rgb"},
                "eye_in_hand_rgb":  {"original_key": "observation.images.eye_in_hand_rgb"},
            },
            "annotation": {
                "human.action.task_description": {},
            },
        }
        modality_path = os.path.join(meta_dir, "modality.json")
        with open(modality_path, "w") as f:
            json.dump(modality, f, indent=4)
        print(f"Saved modality     -> {modality_path}")

        # info.json
        with h5py.File(self.demo_entries[0][2], "r") as f:
            _, video_height, video_width, _ = f["demo_0/obs/agentview_rgb"].shape

        info = {
            "codebase_version": "v2.0",
            "robot_type": "panda",
            "total_episodes": len(episode_entries),
            "total_frames": sum(e["length"] for e in episode_entries),
            "total_tasks": len(self.task_names),
            "total_videos": len(episode_entries),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {"train": f"0:{len(episode_entries)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.images.agentview_rgb": {
                    "dtype": "video",
                    "shape": [video_height, video_width, 3],
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": self.fps,
                        "video.height": video_height,
                        "video.width": video_width,
                        "video.channels": 3,
                        "video.is_depth": False,
                        "has_audio": False,
                    },
                },
                "observation.images.eye_in_hand_rgb": {
                    "dtype": "video",
                    "shape": [video_height, video_width, 3],
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": self.fps,
                        "video.height": video_height,
                        "video.width": video_width,
                        "video.channels": 3,
                        "video.is_depth": False,
                        "has_audio": False,
                    },
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [indices["joint_states_end"]],
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": [indices["action_single_end"]],
                    "names": ["action"],
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
                "annotation.human.action.task_description": {
                    "dtype": "int64", "shape": [1], "names": None,
                },
            },
        }
        info_path = os.path.join(meta_dir, "info.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print(f"Saved info         -> {info_path}")


def main():
    converter = Demo2LeRobotMix()

    if not converter.demo_entries:
        print(f"ERROR: No demo files found in any of: {converter.folder_names}")
        return

    print(f"Tasks:  {converter.task_names}")
    print(f"Demos:  {len(converter.demo_entries)} total\n")

    episode_entries = []
    for demo_idx, (task_index, task_display, demo_file) in enumerate(converter.demo_entries):
        print(f"[{demo_idx}] {os.path.basename(demo_file)}  (task: {task_display})")
        converter.process_data(demo_idx, demo_file, task_index)
        converter.process_videos(demo_idx, demo_file)

        with h5py.File(demo_file, "r") as f:
            length = f["demo_0/actions"].shape[0]
        episode_entries.append({
            "episode_index": demo_idx,
            "tasks": [task_display],
            "length": int(length),
        })

    print()
    converter.process_metadata(episode_entries)

    total_frames = sum(e["length"] for e in episode_entries)
    out_dir = os.path.join(converter.output_location, DATASET_NAME_OUTPUT)
    print(f"\nDone: {len(episode_entries)} episodes, {total_frames} frames -> {out_dir}")


if __name__ == "__main__":
    main()

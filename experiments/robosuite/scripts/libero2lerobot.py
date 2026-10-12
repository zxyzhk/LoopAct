import os
import h5py
import json
import pandas as pd
import numpy as np
import glob
import cv2
from pathlib import Path

_script_dir = Path(__file__).parent
_experiments_dir = _script_dir.parent.parent
_demo_data_dir = _experiments_dir / "demo_data"

# Libero dataset folder under demo_data/ (contains one _demo.hdf5 per task)
LIBERO_FOLDER_NAME = "libero_goal"
LIBERO_DATA_DIR = str(_demo_data_dir / LIBERO_FOLDER_NAME)
DATA_OUTPUT_LOCATION = str(_demo_data_dir)
DATASET_NAME_OUTPUT = f"{LIBERO_FOLDER_NAME}_data"


def _task_name_from_file(hdf5_path: str) -> str:
    """Extract task name from filename like 'open_the_drawer_demo.hdf5' -> 'open the drawer'."""
    stem = Path(hdf5_path).stem  # e.g. 'open_the_middle_drawer_of_the_cabinet_demo'
    if stem.endswith("_demo"):
        stem = stem[:-5]
    return stem.replace("_", " ")


class Libero2LeRobot:
    """Convert LIBERO dataset (one HDF5 per task) to LeRobot format.

    Each HDF5 file in the libero folder corresponds to one task and contains
    demos under 'data/demo_0', 'data/demo_1', etc.
    """

    def __init__(self, libero_data_dir=LIBERO_DATA_DIR, data_output_location=DATA_OUTPUT_LOCATION):
        self.libero_data_dir = libero_data_dir
        self.data_output_location = data_output_location
        self.fps = 10
        self.task_files, self.task_names = self._discover_tasks()
        self.demo_entries = self._build_demo_entries()

    def _discover_tasks(self):
        """Return sorted list of (hdf5_path, task_name) pairs."""
        hdf5_files = sorted(glob.glob(os.path.join(self.libero_data_dir, "*_demo.hdf5")))
        if not hdf5_files:
            raise FileNotFoundError(f"No *_demo.hdf5 files found in {self.libero_data_dir}")
        task_files = hdf5_files
        task_names = [_task_name_from_file(f) for f in hdf5_files]
        return task_files, task_names

    def _build_demo_entries(self):
        """Build flat list of (global_episode_idx, task_index, task_name, hdf5_path, demo_key)."""
        entries = []
        global_idx = 0
        for task_index, (hdf5_path, task_name) in enumerate(zip(self.task_files, self.task_names)):
            with h5py.File(hdf5_path, "r") as f:
                demo_keys = sorted(f["data"].keys(),
                                   key=lambda k: int(k.replace("demo_", "")))
                for demo_key in demo_keys:
                    entries.append((global_idx, task_index, task_name, hdf5_path, demo_key))
                    global_idx += 1
        return entries

    def process_data(self, global_idx, task_index, hdf5_path, demo_key):
        """Write one episode parquet."""
        parquet_name = f"episode_{global_idx:06d}.parquet"

        with h5py.File(hdf5_path, "r") as f:
            base = f[f"data/{demo_key}"]
            actions = base["actions"][:]                  # (T, 7)
            ee_pos = base["obs/ee_pos"][:]                # (T, 3)
            ee_ori = base["obs/ee_ori"][:]                # (T, 3)
            ee_states = base["obs/ee_states"][:]          # (T, 6)
            gripper_states = base["obs/gripper_states"][:] # (T, 2)
            joint_states = base["obs/joint_states"][:]    # (T, 7)

            T = actions.shape[0]
            records = []
            for t in range(T):
                obs_state = np.concatenate([
                    ee_pos[t], ee_ori[t], ee_states[t],
                    gripper_states[t], joint_states[t]
                ])
                records.append({
                    "action": actions[t].tolist(),
                    "observation.state": obs_state.tolist(),
                    "timestamp": t * (1.0 / self.fps),
                    "episode_index": global_idx,
                    "task_index": task_index,
                    "annotation.human.action.task_description": task_index,
                })

        chunk_dir = os.path.join(self.data_output_location, DATASET_NAME_OUTPUT, "data", "chunk-000")
        os.makedirs(chunk_dir, exist_ok=True)
        out_path = os.path.join(chunk_dir, parquet_name)
        pd.DataFrame(records).to_parquet(out_path, index=False)
        print(f"  Wrote {T} timesteps -> {out_path}")

    def process_videos(self, global_idx, hdf5_path, demo_key):
        """Write agentview and eye_in_hand MP4 videos for one episode."""
        video_name = f"episode_{global_idx:06d}.mp4"

        views = {
            "observation.images.agentview_rgb": "obs/agentview_rgb",
            "observation.images.eye_in_hand_rgb": "obs/eye_in_hand_rgb",
        }

        with h5py.File(hdf5_path, "r") as f:
            for view_key, hdf5_key in views.items():
                video_dir = os.path.join(
                    self.data_output_location, DATASET_NAME_OUTPUT,
                    "videos", "chunk-000", view_key
                )
                os.makedirs(video_dir, exist_ok=True)
                out_path = os.path.join(video_dir, video_name)

                frames = f[f"data/{demo_key}/{hdf5_key}"][:]  # (T, H, W, 3)
                T, H, W, _ = frames.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, self.fps, (W, H))
                for t in range(T):
                    writer.write(frames[t, ..., ::-1])  # RGB -> BGR
                writer.release()
                print(f"  Saved {view_key} -> {out_path}")

    def _obs_dims(self):
        """Return observation component dimensions from the first demo."""
        hdf5_path = self.demo_entries[0][3]
        demo_key = self.demo_entries[0][4]
        with h5py.File(hdf5_path, "r") as f:
            base = f[f"data/{demo_key}"]
            dims = {
                "ee_pos": base["obs/ee_pos"].shape[1],
                "ee_ori": base["obs/ee_ori"].shape[1],
                "ee_states": base["obs/ee_states"].shape[1],
                "gripper_states": base["obs/gripper_states"].shape[1],
                "joint_states": base["obs/joint_states"].shape[1],
                "action": base["actions"].shape[1],
                "video_height": base["obs/agentview_rgb"].shape[1],
                "video_width": base["obs/agentview_rgb"].shape[2],
            }
        # Build index ranges
        idx = {}
        cursor = 0
        for key in ("ee_pos", "ee_ori", "ee_states", "gripper_states", "joint_states"):
            idx[f"{key}_start"] = cursor
            cursor += dims[key]
            idx[f"{key}_end"] = cursor
        idx["state_dim"] = cursor
        idx["action_dim"] = dims["action"]
        idx["video_height"] = dims["video_height"]
        idx["video_width"] = dims["video_width"]
        return idx

    def process_metadata(self, episode_lengths):
        """Write tasks.jsonl, episodes.jsonl, modality.json, info.json."""
        meta_dir = os.path.join(self.data_output_location, DATASET_NAME_OUTPUT, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        idx = self._obs_dims()

        # tasks.jsonl
        tasks_path = os.path.join(meta_dir, "tasks.jsonl")
        with open(tasks_path, "w") as f:
            for task_index, task_name in enumerate(self.task_names):
                f.write(json.dumps({"task_index": task_index, "task": task_name}) + "\n")
        print(f"Saved tasks.jsonl ({len(self.task_names)} tasks)")

        # episodes.jsonl
        episodes_path = os.path.join(meta_dir, "episodes.jsonl")
        with open(episodes_path, "w") as f:
            for global_idx, task_index, _, _, _ in self.demo_entries:
                entry = {
                    "episode_index": global_idx,
                    "tasks": [task_name],
                    "length": episode_lengths[global_idx],
                }
                f.write(json.dumps(entry) + "\n")
        print(f"Saved episodes.jsonl ({len(self.demo_entries)} episodes)")

        # modality.json
        modality = {
            "state": {
                "ee_pos":          {"start": idx["ee_pos_start"],          "end": idx["ee_pos_end"]},
                "ee_ori":          {"start": idx["ee_ori_start"],          "end": idx["ee_ori_end"]},
                "ee_states":       {"start": idx["ee_states_start"],       "end": idx["ee_states_end"]},
                "gripper_states":  {"start": idx["gripper_states_start"],  "end": idx["gripper_states_end"]},
                "joint_states":    {"start": idx["joint_states_start"],    "end": idx["joint_states_end"]},
            },
            "action": {
                "all_actions": {"start": 0, "end": idx["action_dim"]},
            },
            "video": {
                "agentview_rgb":    {"original_key": "observation.images.agentview_rgb"},
                "eye_in_hand_rgb":  {"original_key": "observation.images.eye_in_hand_rgb"},
            },
            "annotation": {
                "human.action.task_description": {}
            },
        }
        modality_path = os.path.join(meta_dir, "modality.json")
        with open(modality_path, "w") as f:
            json.dump(modality, f, indent=4)
        print(f"Saved modality.json")

        # info.json
        H, W = idx["video_height"], idx["video_width"]
        total_frames = sum(episode_lengths.values())
        info = {
            "codebase_version": "v2.0",
            "robot_type": "panda",
            "total_episodes": len(self.demo_entries),
            "total_frames": total_frames,
            "total_tasks": len(self.task_names),
            "total_videos": len(self.demo_entries),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self.demo_entries)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.images.agentview_rgb": {
                    "dtype": "video",
                    "shape": [H, W, 3],
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": self.fps, "video.height": H, "video.width": W,
                        "video.channels": 3, "video.is_depth": False, "has_audio": False,
                    },
                },
                "observation.images.eye_in_hand_rgb": {
                    "dtype": "video",
                    "shape": [H, W, 3],
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": self.fps, "video.height": H, "video.width": W,
                        "video.channels": 3, "video.is_depth": False, "has_audio": False,
                    },
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [idx["state_dim"]],
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": [idx["action_dim"]],
                    "names": ["action"],
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
                "annotation.human.action.task_description": {"dtype": "int64", "shape": [1], "names": None},
            },
        }
        info_path = os.path.join(meta_dir, "info.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print(f"Saved info.json")


def main():
    converter = Libero2LeRobot()

    print(f"Found {len(converter.task_files)} tasks, {len(converter.demo_entries)} total episodes")
    for i, name in enumerate(converter.task_names):
        print(f"  [{i}] {name}")

    episode_lengths = {}
    for global_idx, task_index, task_name, hdf5_path, demo_key in converter.demo_entries:
        print(f"\nEpisode {global_idx:04d} | task {task_index} | {demo_key} | {Path(hdf5_path).stem}")
        converter.process_data(global_idx, task_index, hdf5_path, demo_key)
        converter.process_videos(global_idx, hdf5_path, demo_key)

        # Record length for metadata
        with h5py.File(hdf5_path, "r") as f:
            episode_lengths[global_idx] = int(f[f"data/{demo_key}/actions"].shape[0])

    converter.process_metadata(episode_lengths)

    print(f"\nDone. Output: {os.path.join(DATA_OUTPUT_LOCATION, DATASET_NAME_OUTPUT)}")
    print(f"Total episodes: {len(converter.demo_entries)}")


if __name__ == "__main__":
    main()

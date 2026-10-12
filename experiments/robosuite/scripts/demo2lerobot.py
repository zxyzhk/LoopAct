import os
import h5py
import json
import pandas as pd
import numpy as np
import glob
import cv2
from pathlib import Path

### "annotation.human.action.task_description": 0  HARDCODE FOR NOW ONLY ONE TASK

# Go up 2 levels to experiments/, then into demo_data/
_script_dir = Path(__file__).parent  # experiments/robosuite/scripts/
_experiments_dir = _script_dir.parent.parent  # Go up 2 levels to experiments/
_demo_data_dir = _experiments_dir / "demo_data"

# Specify the demo folder name(s) here (relative to demo_data/)
# DEMO_FOLDER_NAME = "pick_up_ladle_move"
DEMO_FOLDER_NAME = ["libero_goal"]

def _normalize_name(name: str) -> str:
    return name.strip().replace(" ", "_")

if isinstance(DEMO_FOLDER_NAME, (list, tuple)):
    DATA_LOCATIONS = [str(_demo_data_dir / name) for name in DEMO_FOLDER_NAME]
    TASK_NAMES = [_normalize_name(name) for name in DEMO_FOLDER_NAME]
else:
    DATA_LOCATIONS = [str(_demo_data_dir / DEMO_FOLDER_NAME)]
    TASK_NAMES = [_normalize_name(DEMO_FOLDER_NAME)]

DATA_LOCATION = DATA_LOCATIONS[0]
DATA_OUTPUT_LOCATION = str(_demo_data_dir)
# Task name is automatically extracted from the folder name
# TASK_NAME: with underscores for dataset naming
# TASK_NAME_DISPLAY: with spaces for metadata
TASK_NAME = _normalize_name(os.path.basename(DATA_LOCATION.rstrip('/')))
TASK_NAME_DISPLAY = TASK_NAME.replace('_', ' ')
DATASET_NAME_OUTPUT = f"{'_'.join(TASK_NAMES)}_data"

class Demo2LeRobot:
    """Convert collected demo dataset to LeRobot format.
   
    This class handles the conversion of robosuite demo format to LeRobot format,
    creating necessary data and video folders with their contents.
    """
   
    def __init__(self, data_location=DATA_LOCATION, data_output_location=DATA_OUTPUT_LOCATION):
        """Initialize the converter.
       
        Args:
            data_location (str): Path to input data directory containing demo_X.hdf5 files
            data_output_location (str): Path to output data directory
        """
        self.data_location = data_location
        self.data_output_location = data_output_location
        self.fps = 10
        self.demo_entries = self.get_demo_entries()

    def get_demo_entries(self):
        """Get list of demo files and their task indices."""
        entries = []
        for task_index, data_location in enumerate(DATA_LOCATIONS):
            demo_files = glob.glob(os.path.join(data_location, 'demo_*.hdf5'))
            demo_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            task_name = TASK_NAMES[task_index]
            task_display = task_name.replace('_', ' ')
            for demo_file in demo_files:
                entries.append((task_index, task_display, demo_file))
        return entries

    def process_data(self, demo_idx, demo_file, task_index):
        """Process data for a single demo and save to parquet format.
       
        Args:
            demo_idx (int): Index of the demo to process
            demo_file (str): Path to the demo HDF5 file
        """
        parquet_name = f"episode_{demo_idx:06d}.parquet"
       
        with h5py.File(demo_file, "r") as f:
            # Read data from the demo_0 group
            actions = f["demo_0/actions"][:]           # (T, 7)
            ee_pos = f["demo_0/obs/ee_pos"][:]         # (T, 3)
            ee_ori = f["demo_0/obs/ee_ori"][:]         # (T, 3)
            ee_states = f["demo_0/obs/ee_states"][:]   # (T, 6)
            gripper_states = f["demo_0/obs/gripper_states"][:] # (T, 2)
            joint_states = f["demo_0/obs/joint_states"][:]     # (T, 7)

            T = actions.shape[0]
            records = []
            for time_step in range(T):
                current_action = actions[time_step]
                cur_ee_pos = ee_pos[time_step]
                cur_ee_ori = ee_ori[time_step]
                cur_ee_states = ee_states[time_step]
                cur_gripper_states = gripper_states[time_step]
                cur_joint_states = joint_states[time_step]
                timestamp = time_step * (1/self.fps)
               
                obs_state = np.concatenate([
                    cur_ee_pos, cur_ee_ori, cur_ee_states,
                    cur_gripper_states, cur_joint_states
                ])
               
                record = {
                    "action": current_action.tolist(),
                    "observation.state": obs_state.tolist(),
                    "timestamp": timestamp,
                    "episode_index": demo_idx,
                    "task_index": task_index,
                    "annotation.human.action.task_description": task_index
                }
                records.append(record)

        chunk_dir = os.path.join(self.data_output_location, DATASET_NAME_OUTPUT, "data", "chunk-000")
        os.makedirs(chunk_dir, exist_ok=True)
        out_parquet = os.path.join(chunk_dir, parquet_name)
        df = pd.DataFrame(records)
        df.to_parquet(out_parquet, index=False)
        print(f"Wrote {len(records)} timesteps to {out_parquet}")

    def process_videos(self, demo_idx, demo_file):
        """Process videos for a single demo and save in MP4 format.
       
        Args:
            demo_idx (int): Index of the demo to process
            demo_file (str): Path to the demo HDF5 file
        """
        video_name = f"episode_{demo_idx:06d}.mp4"
       
        # Process agent view video
        video_dir = os.path.join(self.data_output_location, DATASET_NAME_OUTPUT, "videos",
                                "chunk-000", "observation.images.agentview_rgb")
        os.makedirs(video_dir, exist_ok=True)
        out_video_path = os.path.join(video_dir, video_name)
       
        with h5py.File(demo_file, "r") as f:
            video_data = f["demo_0/obs/agentview_rgb"][:]  # (T, H, W, 3)
            T, H, W, C = video_data.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_video_path, fourcc, self.fps, (W, H))
           
            for t in range(T):
                frame = video_data[t]
                frame_bgr = frame[..., ::-1]  # RGB to BGR
                writer.write(frame_bgr)
            writer.release()
        print(f"Saved agentview video to {out_video_path}")

        # Process eye-in-hand video
        video2_dir = os.path.join(self.data_output_location, DATASET_NAME_OUTPUT, "videos",
                                 "chunk-000", "observation.images.eye_in_hand_rgb")
        os.makedirs(video2_dir, exist_ok=True)
        out_video2_path = os.path.join(video2_dir, video_name)
       
        with h5py.File(demo_file, "r") as f:
            video_data2 = f["demo_0/obs/eye_in_hand_rgb"][:]  # (T, H, W, 3)
            T, H, W, C = video_data2.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_video2_path, fourcc, self.fps, (W, H))
           
            for t in range(T):
                frame = video_data2[t]
                frame_bgr = frame[..., ::-1]  # RGB to BGR
                writer.write(frame_bgr)
            writer.release()
        print(f"Saved eye_in_hand video to {out_video2_path}")

    def calculate_obs_indices(self):
        """Calculate observation indices for different state components."""
        demo_file = self.demo_entries[0][2]  # Use first demo to get dimensions
       
        with h5py.File(demo_file, "r") as f:
            # Get dimensions from the data
            ee_pos = f["demo_0/obs/ee_pos"]         # (T, 3)
            ee_ori = f["demo_0/obs/ee_ori"]         # (T, 3)
            ee_states = f["demo_0/obs/ee_states"]   # (T, 6)
            gripper_states = f["demo_0/obs/gripper_states"] # (T, 2)
            joint_states = f["demo_0/obs/joint_states"]     # (T, 7)
            actions = f["demo_0/actions"]           # (T, 7)
           
            # Calculate start and end indices
            ee_pos_start = 0
            ee_pos_end = ee_pos.shape[1]  # 3
           
            ee_ori_start = ee_pos_end
            ee_ori_end = ee_ori_start + ee_ori.shape[1]  # 3
           
            ee_states_start = ee_ori_end
            ee_states_end = ee_states_start + ee_states.shape[1]  # 6
           
            gripper_states_start = ee_states_end
            gripper_states_end = gripper_states_start + gripper_states.shape[1]  # 2
           
            joint_states_start = gripper_states_end
            joint_states_end = joint_states_start + joint_states.shape[1]  # 7
           
            action_single_start = 0
            action_single_end = actions.shape[1]  # 7
           
            return {
                "ee_pos_start": ee_pos_start,
                "ee_pos_end": ee_pos_end,
                "ee_ori_start": ee_ori_start,
                "ee_ori_end": ee_ori_end,
                "ee_states_start": ee_states_start,
                "ee_states_end": ee_states_end,
                "gripper_states_start": gripper_states_start,
                "gripper_states_end": gripper_states_end,
                "joint_states_start": joint_states_start,
                "joint_states_end": joint_states_end,
                "action_single_start": action_single_start,
                "action_single_end": action_single_end
            }

    def process_metadata(self):
        """Process and create metadata files for the dataset.
       
        Creates:
        - tasks.jsonl: Task descriptions
        - episodes.jsonl: Episode information
        - modality.json: Data modality specifications
        - info.json: Dataset information
        """
        # Create meta directory
        meta_dir = os.path.join(self.data_output_location, DATASET_NAME_OUTPUT, "meta")
        os.makedirs(meta_dir, exist_ok=True)
       
        # Process tasks.jsonl (multiple tasks supported)
        meta_path = os.path.join(meta_dir, "tasks.jsonl")
        with open(meta_path, "w") as f:
            for task_index, task_name in enumerate(TASK_NAMES):
                entry = {"task_index": task_index, "task": task_name.replace("_", " ")}
                f.write(json.dumps(entry) + "\n")
        print(f"Saved task metadata to {meta_path}")

        # Process episodes.jsonl
        episode_path = os.path.join(meta_dir, "episodes.jsonl")
        episode_entries = []
       
        for demo_idx, (_, task_display, demo_file) in enumerate(self.demo_entries):
            try:
                with h5py.File(demo_file, "r") as f:
                    actions = f["demo_0/actions"]
                    length = actions.shape[0]
                    entry = {
                        "episode_index": demo_idx,
                        "tasks": [task_display],
                        "length": int(length)
                    }
                    episode_entries.append(entry)
            except Exception as e:
                print(f"Error reading {demo_file}: {e}")
               
        with open(episode_path, "w") as f:
            for entry in episode_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved episode metadata to {episode_path}")

        # Calculate indices for modality.json
        indices = self.calculate_obs_indices()
       
        # Create modality.json
        metadata = {
            "state": {
                "ee_pos": {"start": indices["ee_pos_start"], "end": indices["ee_pos_end"]},
                "ee_ori": {"start": indices["ee_ori_start"], "end": indices["ee_ori_end"]},
                "ee_states": {"start": indices["ee_states_start"], "end": indices["ee_states_end"]},
                "gripper_states": {"start": indices["gripper_states_start"], "end": indices["gripper_states_end"]},
                "joint_states": {"start": indices["joint_states_start"], "end": indices["joint_states_end"]},
            },
            "action": {
                "all_actions": {"start": indices["action_single_start"], "end": indices["action_single_end"]},
            },
            "video": {
                "agentview_rgb": {"original_key": "observation.images.agentview_rgb"},
                "eye_in_hand_rgb": {"original_key": "observation.images.eye_in_hand_rgb"}
            },
            "annotation": {
                "human.action.task_description": {}
            }
        }

        modality_path = os.path.join(meta_dir, "modality.json")
        with open(modality_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved modality metadata to {modality_path}")

        # Get video dimensions from first demo
        with h5py.File(self.demo_entries[0][2], "r") as f:
            video_data = f["demo_0/obs/agentview_rgb"]
            _, video_height, video_width, _ = video_data.shape

        # Create info.json
        info = {
            "codebase_version": "v2.0",
            "robot_type": "panda",
            "total_episodes": len(episode_entries),
            "total_frames": sum(entry["length"] for entry in episode_entries),
            "total_tasks": len(TASK_NAMES),
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
                        "has_audio": False
                    }
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
                        "has_audio": False
                    }
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [indices["joint_states_end"]],
                    "names": ["state"]
                },
                "action": {
                    "dtype": "float32",
                    "shape": [indices["action_single_end"]],
                    "names": ["action"]
                },
                "timestamp": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": None
                },
                "episode_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "task_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "annotation.human.action.task_description": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                }
            }
        }
       
        info_path = os.path.join(meta_dir, "info.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print(f"Saved info metadata to {info_path}")


def main():
    """Main function to process all demos in the dataset."""
    converter = Demo2LeRobot(data_location=DATA_LOCATION, data_output_location=DATA_OUTPUT_LOCATION)
    
    if not converter.demo_entries:
        print(f"ERROR: No demo files found in {DATA_LOCATIONS}")
        print(f"Please ensure the directories contain demo_*.hdf5 files")
        return
    
    print(f"Found {len(converter.demo_entries)} demo file(s) in {DATA_LOCATIONS}")
    
    # Process each demo file
    for demo_idx, (task_index, _, demo_file) in enumerate(converter.demo_entries):
        demo_filename = os.path.basename(demo_file)
        print(f"Processing demo {demo_idx}: {demo_filename}")
        
        converter.process_data(demo_idx, demo_file, task_index)
        converter.process_videos(demo_idx, demo_file)

    # Process metadata
    converter.process_metadata()
    print(f"\nTotal demos processed: {len(converter.demo_entries)}")
    print(f"Output location: {os.path.join(DATA_OUTPUT_LOCATION, DATASET_NAME_OUTPUT)}")


if __name__ == "__main__":
    main()

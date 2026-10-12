"""
lerobot_mix.py - Merge multiple LeRobot-format datasets into one.

Source datasets may have one or multiple tasks. The merged dataset gets:
- Sequential episode indices across all sources
- Global task indices (remapped from per-source indices)
- Parquet files with updated episode_index / task_index columns
- Videos copied and renamed to match new episode indices

Usage:
    python lerobot_mix.py
"""

import json
import shutil
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).parent                # experiments/robosuite/scripts/
_experiments_dir = _script_dir.parent.parent       # experiments/
_demo_data_dir = _experiments_dir / "demo_data"

# SOURCE_DATASETS = [
#     "pick_up_the_red_toy_data",
#     "pick_up_the_green_toy_data",
#     "pick_up_the_can_data",
# ]

SOURCE_DATASETS = [
    "pick_up_the_red_toy_data",
    "pick_up_the_green_toy_data",
    "drop_the_ball_data_imp",
    "pick_up_the_can_data"
]

# Output dataset name
OUTPUT_NAME = "4_tasks_data_n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_jsonl(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def write_jsonl(path, entries):
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    output_dir = _demo_data_dir / OUTPUT_NAME
    if output_dir.exists():
        print(f"Output directory already exists: {output_dir}")
        print("Remove it first if you want to regenerate.")
        return

    # 1. Load source metadata
    sources = []
    for ds_name in SOURCE_DATASETS:
        ds_dir = _demo_data_dir / ds_name
        info = json.loads((ds_dir / "meta" / "info.json").read_text())
        tasks = read_jsonl(ds_dir / "meta" / "tasks.jsonl")
        episodes = read_jsonl(ds_dir / "meta" / "episodes.jsonl")
        modality = json.loads((ds_dir / "meta" / "modality.json").read_text())
        sources.append({
            "name": ds_name,
            "dir": ds_dir,
            "info": info,
            "tasks": tasks,
            "episodes": episodes,
            "modality": modality,
        })

    # 2. Build merged task list (supports multi-task sources)
    # For each source, map old task indices → new global task indices.
    merged_tasks = []
    for src in sources:
        task_idx_map = {}
        for task_entry in src["tasks"]:
            old_idx = task_entry["task_index"]
            new_idx = len(merged_tasks)
            merged_tasks.append({"task_index": new_idx, "task": task_entry["task"]})
            task_idx_map[old_idx] = new_idx
        src["task_idx_map"] = task_idx_map
    print(f"Total tasks: {len(merged_tasks)}")
    for t in merged_tasks:
        print(f"  [{t['task_index']}] {t['task']}")

    chunk_size = sources[0]["info"].get("chunks_size", 1000)

    # 3. Copy data + videos with renumbered episodes
    merged_episodes = []
    total_frames = 0
    global_ep_idx = 0

    for src in sources:
        src_dir = src["dir"]
        task_idx_map = src["task_idx_map"]
        src_chunk_size = src["info"].get("chunks_size", 1000)

        for ep in src["episodes"]:
            old_ep_idx = ep["episode_index"]
            new_ep_idx = global_ep_idx

            old_chunk = f"chunk-{old_ep_idx // src_chunk_size:03d}"
            new_chunk = f"chunk-{new_ep_idx // chunk_size:03d}"
            old_name = f"episode_{old_ep_idx:06d}"
            new_name = f"episode_{new_ep_idx:06d}"

            # --- Parquet ---
            old_parquet = src_dir / "data" / old_chunk / f"{old_name}.parquet"
            new_parquet = output_dir / "data" / new_chunk / f"{new_name}.parquet"
            new_parquet.parent.mkdir(parents=True, exist_ok=True)

            df = pd.read_parquet(old_parquet)
            df["episode_index"] = new_ep_idx
            df["task_index"] = df["task_index"].map(task_idx_map)
            df["annotation.human.action.task_description"] = df["task_index"]
            df.to_parquet(new_parquet, index=False)

            # --- Videos ---
            for video_key in [
                "observation.images.agentview_rgb",
                "observation.images.eye_in_hand_rgb",
            ]:
                old_video = src_dir / "videos" / old_chunk / video_key / f"{old_name}.mp4"
                new_video = output_dir / "videos" / new_chunk / video_key / f"{new_name}.mp4"
                new_video.parent.mkdir(parents=True, exist_ok=True)
                if old_video.exists():
                    shutil.copy2(old_video, new_video)

            merged_episodes.append({
                "episode_index": new_ep_idx,
                "tasks": ep["tasks"],  # task text unchanged; task_index remapped in parquet
                "length": ep["length"],
            })
            total_frames += ep["length"]
            global_ep_idx += 1

        print(f"  {src['name']}: {len(src['episodes'])} episodes, {len(src['tasks'])} task(s)")

    total_episodes = global_ep_idx
    total_chunks = (total_episodes + chunk_size - 1) // chunk_size
    print(f"\nTotal: {total_episodes} episodes, {total_frames} frames, {total_chunks} chunk(s)")

    # 4. Write metadata
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # tasks.jsonl
    write_jsonl(meta_dir / "tasks.jsonl", merged_tasks)

    # episodes.jsonl
    write_jsonl(meta_dir / "episodes.jsonl", merged_episodes)

    # modality.json (identical across sources)
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(sources[0]["modality"], f, indent=4)

    # info.json (based on first source, update totals)
    info = sources[0]["info"].copy()
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = len(merged_tasks)
    info["total_videos"] = total_episodes
    info["total_chunks"] = total_chunks
    info["splits"] = {"train": f"0:{total_episodes}"}
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Validate and visualize HDF5 demo data.
Set the 'path' variable below to your HDF5 file or folder.
"""

import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

# ============ CONFIGURATION ============
# Use relative path from this script's location (robosuite/scripts/ folder)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_robosuite_dir = os.path.dirname(_script_dir)
_project_root = os.path.dirname(_robosuite_dir)

# Demo data folder
DEMO_DATA_DIR = os.path.join(_project_root, "demo_data")

path = os.path.join(DEMO_DATA_DIR, "pick_up_ladle_data")  # Folder or file
fps = 10   # Video playback FPS
no_video = False  # Set True to skip video playback
# =======================================


def load_demo(filepath):
    """Load demo data from HDF5 file."""
    with h5py.File(filepath, "r") as f:
        # Find demo group (at root or under 'data')
        if "data" in f:
            demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
            parent = f["data"]
        else:
            demo_keys = [k for k in f.keys() if k.startswith("demo_")]
            parent = f

        if not demo_keys:
            raise ValueError(f"No demo found in {filepath}")

        demo = parent[demo_keys[0]]
        data = {
            "actions": demo["actions"][:],
            "joint_states": demo["obs/joint_states"][:],
            "gripper_states": demo["obs/gripper_states"][:],
            "robot_states": demo["robot_states"][:],
            "agentview_rgb": demo["obs/agentview_rgb"][:],
            "eye_in_hand_rgb": demo["obs/eye_in_hand_rgb"][:],
        }
    return data


def plot_robot_states_9(data, title_prefix=""):
    """Plot 7 joint states + 2 gripper states as 9 vertical subplots."""
    joint_states = data["joint_states"]  # (N, 7)
    gripper_states = data["gripper_states"]  # (N, 2)
    
    # Combine into 9 columns
    all_states = np.concatenate([joint_states, gripper_states], axis=1)  # (N, 9)
    labels = [f"Joint {i+1}" for i in range(7)] + ["Gripper L", "Gripper R"]
    
    fig, axes = plt.subplots(9, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"{title_prefix}Robot States (7 joints + 2 gripper)", fontsize=12)
    
    for i, ax in enumerate(axes):
        ax.plot(all_states[:, i], linewidth=0.8)
        ax.set_ylabel(labels[i], fontsize=9)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Step", fontsize=10)
    plt.tight_layout()
    return fig


def plot_actions_7(data, title_prefix=""):
    """Plot 7 actions as 7 vertical subplots."""
    actions = data["actions"]  # (N, 7)
    labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    
    fig, axes = plt.subplots(7, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(f"{title_prefix}Actions (7 dims)", fontsize=12)
    
    for i, ax in enumerate(axes):
        ax.plot(actions[:, i], linewidth=0.8)
        ax.set_ylabel(labels[i], fontsize=9)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Step", fontsize=10)
    plt.tight_layout()
    return fig


def play_dual_video(agentview, eye_in_hand, fps=20):
    """Play both camera views side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    num_frames = min(len(agentview), len(eye_in_hand))
    print(f"Playing {num_frames} frames at {fps} fps (side by side)...")
    
    for i in range(num_frames):
        axes[0].clear()
        axes[0].imshow(agentview[i])
        axes[0].set_title(f"Agentview - Frame {i+1}/{num_frames}")
        axes[0].axis("off")
        
        axes[1].clear()
        axes[1].imshow(eye_in_hand[i])
        axes[1].set_title(f"Eye-in-Hand - Frame {i+1}/{num_frames}")
        axes[1].axis("off")
        
        plt.pause(1.0 / fps)
    
    plt.close(fig)


def validate_robot_states(data):
    """Verify robot_states = joint_states + gripper_states."""
    robot_states = data["robot_states"]
    joint_states = data["joint_states"]
    gripper_states = data["gripper_states"]
    
    expected = np.concatenate([joint_states, gripper_states], axis=1)
    match = np.allclose(robot_states, expected)
    
    print(f"\n=== Robot States Validation ===")
    print(f"robot_states shape: {robot_states.shape}")
    print(f"joint_states shape: {joint_states.shape}")
    print(f"gripper_states shape: {gripper_states.shape}")
    print(f"robot_states == [joint_states, gripper_states]: {match}")
    
    return match


def main():
    # Find HDF5 files
    if not path:
        print("Error: Set 'path' variable at the top of this script!")
        return
    
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.hdf5")))
        if not files:
            print(f"No HDF5 files found in {path}")
            return
    else:
        print(f"Path not found: {path}")
        return

    print(f"Found {len(files)} HDF5 file(s)")

    for filepath in files:
        print(f"\n{'='*60}")
        print(f"Processing: {filepath}")
        print(f"{'='*60}")

        try:
            data = load_demo(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

        filename = os.path.basename(filepath)
        
        # Validate robot_states
        validate_robot_states(data)

        # Print shapes
        print(f"\n=== Data Shapes ===")
        print(f"actions: {data['actions'].shape}")
        print(f"joint_states: {data['joint_states'].shape}")
        print(f"gripper_states: {data['gripper_states'].shape}")
        print(f"agentview_rgb: {data['agentview_rgb'].shape}")
        print(f"eye_in_hand_rgb: {data['eye_in_hand_rgb'].shape}")

        # Plot 9 robot states (7 joints + 2 gripper) horizontally
        plot_robot_states_9(data, title_prefix=f"{filename} - ")
        plt.show()

        # Plot 7 actions horizontally
        plot_actions_7(data, title_prefix=f"{filename} - ")
        plt.show()

        # Play video
        if not no_video:
            print("\n=== Playing Camera Views ===")
            play_dual_video(data["agentview_rgb"], data["eye_in_hand_rgb"], fps=fps)

    print("\n[OK] Validation complete!")


if __name__ == "__main__":
    main()

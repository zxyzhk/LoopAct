#!/usr/bin/env python3
"""
Hdf5Recorder - Collects teleop trajectories and writes to HDF5.
"""

import math
import numpy as np


def _first_obs_key(obs, keys, label):
    for key in keys:
        if key in obs:
            return obs[key]
    raise KeyError(f"Missing {label}; tried keys: {keys}")


def _quat_to_euler_xyz(quat_xyzw):
    x, y, z, w = quat_xyzw
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _get_sim_state_vector(env):
    sim_state = env.sim.get_state()
    pieces = [np.asarray(sim_state.qpos), np.asarray(sim_state.qvel)]
    if getattr(sim_state, "act", None) is not None:
        pieces.append(np.asarray(sim_state.act))
    if getattr(sim_state, "udd_state", None) is not None:
        pieces.append(np.asarray(sim_state.udd_state))
    return np.concatenate([p.ravel() for p in pieces]).astype(np.float64)


class Hdf5Recorder:
    def __init__(self, record_path, task_name=None, task_description=None):
        self.record_path = record_path
        self.task_name = task_name or "unknown_task"
        self.task_description = task_description or ""
        self.task_success = False
        self.task_score = 0.0
        self.reset()

    def reset(self):
        self.actions = []
        self.rewards = []
        self.dones = []
        self.agentview_rgb = []
        self.eye_in_hand_rgb = []
        self.ee_pos = []
        self.ee_ori = []
        self.ee_states = []
        self.gripper_states = []
        self.joint_states = []
        self.robot_states = []
        self.states = []
        self.task_success = False
        self.task_score = 0.0
    
    def set_task_success(self, success: bool, score: float = 0.0):
        """Set the task success status and score."""
        self.task_success = success
        self.task_score = score

    def record_step(self, env, obs, action, reward, done):
        agentview = _first_obs_key(
            obs,
            ["agentview_image", "agentview_rgb"],
            "agentview_rgb",
        )
        eye_in_hand = _first_obs_key(
            obs,
            ["robot0_eye_in_hand_image", "eye_in_hand_image", "eye_in_hand_rgb"],
            "eye_in_hand_rgb",
        )
        ee_pos = _first_obs_key(
            obs,
            ["robot0_eef_pos", "eef_pos", "ee_pos"],
            "ee_pos",
        )
        ee_quat = _first_obs_key(
            obs,
            ["robot0_eef_quat", "eef_quat", "ee_quat"],
            "ee_ori",
        )
        ee_ori = _quat_to_euler_xyz(np.asarray(ee_quat, dtype=np.float64))
        gripper = _first_obs_key(
            obs,
            ["robot0_gripper_qpos", "gripper_states", "gripper_qpos"],
            "gripper_states",
        )
        joints = _first_obs_key(
            obs,
            ["robot0_joint_pos", "joint_states", "joint_pos"],
            "joint_states",
        )

        ee_pos = np.asarray(ee_pos, dtype=np.float64).reshape(-1)
        ee_ori = np.asarray(ee_ori, dtype=np.float64).reshape(-1)
        gripper = np.asarray(gripper, dtype=np.float64).reshape(-1)
        joints = np.asarray(joints, dtype=np.float64).reshape(-1)

        ee_state = np.concatenate([ee_pos, ee_ori]).astype(np.float64)
        robot_state = np.concatenate([joints, gripper]).astype(np.float64)
        sim_state = _get_sim_state_vector(env)

        self.actions.append(np.asarray(action, dtype=np.float64))
        self.rewards.append(reward)
        self.dones.append(done)
        self.agentview_rgb.append(np.asarray(agentview, dtype=np.uint8))
        self.eye_in_hand_rgb.append(np.asarray(eye_in_hand, dtype=np.uint8))
        self.ee_pos.append(ee_pos)
        self.ee_ori.append(ee_ori)
        self.ee_states.append(ee_state)
        self.gripper_states.append(gripper)
        self.joint_states.append(joints)
        self.robot_states.append(robot_state)
        self.states.append(sim_state)

    def save(self, record_path=None):
        if not self.actions:
            raise RuntimeError("No steps recorded; nothing to save.")

        try:
            import h5py
        except Exception as exc:
            raise RuntimeError(f"h5py is required for recording: {exc}") from exc

        actions = np.asarray(self.actions, dtype=np.float64)
        rewards = np.asarray(self.rewards, dtype=np.uint8)
        dones = np.asarray(self.dones, dtype=np.uint8)
        agentview_rgb = np.asarray(self.agentview_rgb, dtype=np.uint8)
        eye_in_hand_rgb = np.asarray(self.eye_in_hand_rgb, dtype=np.uint8)
        ee_pos = np.asarray(self.ee_pos, dtype=np.float64)
        ee_ori = np.asarray(self.ee_ori, dtype=np.float64)
        ee_states = np.asarray(self.ee_states, dtype=np.float64)
        gripper_states = np.asarray(self.gripper_states, dtype=np.float64)
        joint_states = np.asarray(self.joint_states, dtype=np.float64)
        robot_states = np.asarray(self.robot_states, dtype=np.float64)
        states = np.asarray(self.states, dtype=np.float64)

        output_path = record_path or self.record_path
        with h5py.File(output_path, "w") as h5f:
            demo_group = h5f.create_group("demo_0")
            demo_group.create_dataset("actions", data=actions)
            demo_group.create_dataset("dones", data=dones)
            obs_group = demo_group.create_group("obs")
            obs_group.create_dataset("agentview_rgb", data=agentview_rgb)
            obs_group.create_dataset("ee_ori", data=ee_ori)
            obs_group.create_dataset("ee_pos", data=ee_pos)
            obs_group.create_dataset("ee_states", data=ee_states)
            obs_group.create_dataset("eye_in_hand_rgb", data=eye_in_hand_rgb)
            obs_group.create_dataset("gripper_states", data=gripper_states)
            obs_group.create_dataset("joint_states", data=joint_states)
            demo_group.create_dataset("rewards", data=rewards)
            demo_group.create_dataset("robot_states", data=robot_states)
            demo_group.create_dataset("states", data=states)
            
            # Save task metadata
            demo_group.attrs["task_name"] = self.task_name
            demo_group.attrs["task_description"] = self.task_description
            demo_group.attrs["task_success"] = int(self.task_success)
            demo_group.attrs["task_score"] = float(self.task_score)
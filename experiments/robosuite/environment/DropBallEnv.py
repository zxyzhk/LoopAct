#!/usr/bin/env python3
"""
DropBall Environment — same table/robot/camera geometry as ToyRedEnv,
with only two objects on the table:
  • cart  (moving target — replaces toybot's role in toy_red)
  • ball  (the object the user manipulates)
"""

import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_robosuite_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _robosuite_dir)

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import Task
from robosuite.models.objects.primitive import BallObject

from robosuite.environments.base import register_env
from robosuite.utils.mjcf_utils import find_elements, new_element
from typing import Optional
from tasks.base import TaskSuccess
from xml_objects import ToyCartObject, CupObject


class DropBall(ManipulationEnv):
    """Table environment with a ball and a moving cart.

    The ball starts snap-locked to the gripper site. A one-way latch releases
    it the first time the user actually presses the open-gripper key (',') —
    not when action[-1] is negative, because collect_demos's reset_gripper()
    sets that to -1.0 every reset and would otherwise auto-drop. While the
    latch is True we also clamp the gripper command to "close" so neither the
    teleop's startup _open_gripper_steps nor the persistent gripper_cmd can
    leak through. On release we snap the fingers wide so the ball drops
    instantly instead of waiting for the actuator to ramp open.
    """

    # Original base pose
    # INIT_QPOS = np.array([0.0, -0.3, 0.0, -1.9, 0.0, 1.8, 0.785])
    # OBJECT_POS_NOISE_XY = 0.05
    # ROBOT_QPOS_NOISE = 0.05

    # Fixed snapshot pose (episode 9/10, seed 1242416922)
    INIT_QPOS = np.array([0.036, -0.337, -0.032, -1.882, -0.025, 1.805, 0.822])
    OBJECT_POS_NOISE_XY = 0.03
    ROBOT_QPOS_NOISE = 0.03

    # Initial finger qpos when wrapped around the ball (purely cosmetic — the
    # ball is held physically by the snap-to-gripper-site sync, not by friction).
    GRIPPER_GRIP_QPOS = 0.025
    # Per-finger qpos used to snap the gripper fully open on release.
    GRIPPER_RELEASE_QPOS = 0.04
    # First env.steps after reset where any release-key press is ignored. The
    # teleop's _open_gripper_steps fires action[-1]=-1.0 here; if the user is
    # holding ',' through it we still don't release until the buffer expires.
    SETTLE_STEPS = 30
    # Key that releases the ball (matches mouse_teleoperate's "Open" key).
    RELEASE_KEY = ","

    GRIPPER_SITE = "gripper0_right_grip_site"
    GRIPPER_FINGER_JOINTS = (
        "gripper0_right_finger_joint1",
        "gripper0_right_finger_joint2",
    )

    # Cup sits on the cart bed; cart bed top (cart-local) is at z = 0.016.
    # We sync the cup's free-joint qpos to the cart's qpos every step so the
    # cup rides the cart without depending on friction.
    CUP_LOCAL_OFFSET_Z = 0.016

    def __init__(
        self,
        randomize_init: bool = True,
        task_name: Optional[str] = None,
        task_success: Optional[TaskSuccess] = None,
        **kwargs,
    ):
        self.randomize_init = randomize_init
        self.object_base_positions = {}
        self.object_base_rotations = {}
        self.task_name = task_name or "drop_ball"
        self.task_success = task_success
        self.current_success = False
        self.success_checked = False
        # One-way latch: True until user opens gripper past RELEASE_QPOS
        self._ball_in_hand = True
        self._steps_since_reset = 0
        super().__init__(**kwargs)
        if self.task_success:
            print(f"\n[TASK] Loaded: {self.task_success.task_description}")

    def _reset_internal(self):
        super()._reset_internal()
        if not hasattr(self, "object_initial_heights"):
            self.object_initial_heights = {}

        # Override agentview camera (matches ToyRedEnv)
        if hasattr(self.sim.model, "camera_names"):
            for cam_name in self.sim.model.camera_names:
                if cam_name == "agentview":
                    cam_id = self.sim.model.camera_name2id(cam_name)
                    self.sim.model.cam_pos[cam_id] = np.array([1.1536, -0.0076, 1.7745])
                    self.sim.model.cam_quat[cam_id] = np.array([0.6260, 0.3337, 0.3316, 0.6219])
                    self.sim.model.cam_fovy[cam_id] = 50.0
                    break

        # Robot init
        robot = self.robots[0]
        robot_qpos = []
        if hasattr(robot, "init_qpos") and self.INIT_QPOS is not None:
            for i, idx in enumerate(robot._ref_joint_pos_indexes):
                if i < len(self.INIT_QPOS):
                    noise = (
                        np.random.uniform(-self.ROBOT_QPOS_NOISE, self.ROBOT_QPOS_NOISE)
                        if self.randomize_init else 0.0
                    )
                    qpos = self.INIT_QPOS[i] + noise
                    self.sim.data.qpos[idx] = qpos
                    robot_qpos.append(qpos)
        if robot_qpos:
            print(f"[INIT] Robot joints: {np.array2string(np.array(robot_qpos), precision=3, suppress_small=True)}")

        # Randomize object positions
        object_positions = {}
        if self.randomize_init:
            for obj_name, base_pos in self.object_base_positions.items():
                joint_name = self.object_joints.get(obj_name)
                if joint_name:
                    try:
                        new_pos = base_pos.copy()
                        new_pos[0] += np.random.uniform(-self.OBJECT_POS_NOISE_XY, self.OBJECT_POS_NOISE_XY)
                        new_pos[1] += np.random.uniform(-self.OBJECT_POS_NOISE_XY, self.OBJECT_POS_NOISE_XY)
                        self.sim.data.set_joint_qpos(joint_name, new_pos)
                        object_positions[obj_name] = new_pos
                    except Exception as e:
                        print(f"[DEBUG] Could not set joint position for {obj_name}: {e}")
                        object_positions[obj_name] = base_pos
                else:
                    object_positions[obj_name] = base_pos
        else:
            object_positions = dict(self.object_base_positions)

        if object_positions:
            print("[INIT] Object positions:")
            for obj_name, pos in object_positions.items():
                print(f"  {obj_name:14s}: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f}]")

        self.sim.forward()

        self.object_initial_heights = {}
        try:
            self.object_initial_heights["ball_main"] = self.sim.data.get_body_xpos("ball_main")[2]
            print(f"[INIT] Initial ball height: {self.object_initial_heights['ball_main']:.3f}m")
        except Exception as e:
            print(f"[WARN] Could not store initial heights: {e}")

        # Reset ball-in-hand state and place ball at the gripper
        self._ball_in_hand = True
        self._steps_since_reset = 0
        self._place_ball_in_gripper()

    def _place_ball_in_gripper(self):
        """Move the ball to the gripper grip site and close the fingers around it."""
        # First sim.forward so the gripper site reflects the new robot pose
        self.sim.forward()

        try:
            gripper_pos = self.sim.data.get_site_xpos(self.GRIPPER_SITE).copy()
        except Exception as e:
            print(f"[WARN] Could not query {self.GRIPPER_SITE}: {e}; ball not placed in hand")
            return

        ball_joint = self.object_joints.get("ball")
        if ball_joint is not None:
            ball_qpos = np.concatenate([gripper_pos, np.array([1.0, 0.0, 0.0, 0.0])])
            try:
                self.sim.data.set_joint_qpos(ball_joint, ball_qpos)
            except Exception as e:
                print(f"[WARN] Could not set ball qpos: {e}")
                return

        # Close the fingers around the ball (best-effort visual; sync handles physics)
        for j, q in zip(self.GRIPPER_FINGER_JOINTS,
                        (self.GRIPPER_GRIP_QPOS, -self.GRIPPER_GRIP_QPOS)):
            try:
                self.sim.data.set_joint_qpos(j, q)
                self.sim.data.set_joint_qvel(j, 0.0)
            except Exception:
                pass

        self.sim.forward()
        print(f"[INIT] Ball placed in gripper at [{gripper_pos[0]:+.3f}, {gripper_pos[1]:+.3f}, {gripper_pos[2]:.3f}]")

    def _pre_action(self, action, policy_step=False):
        # Release the ball when:
        #   (a) user presses the release key during teleoperation, OR
        #   (b) model signals open-gripper (action[-1] < 0) and no teleop is
        #       attached — this lets the inference model trigger release.
        # In both cases the SETTLE_STEPS guard still applies.
        if self._ball_in_hand and self._steps_since_reset >= self.SETTLE_STEPS:
            teleop_release = self._user_pressed_release_key()
            model_release = (
                getattr(self, "_teleop_ref", None) is None
                and action is not None
                and len(action) >= 1
                and float(action[-1]) < 0
            )
            if teleop_release or model_release:
                self._release_ball()

        # While we're still holding, force the gripper command to close so
        # nothing else can drift the gripper open behind our back.
        if (self._ball_in_hand
                and action is not None
                and len(action) >= 1):
            try:
                action[-1] = 1.0
            except Exception:
                pass
        if policy_step:
            self._steps_since_reset += 1

        super()._pre_action(action, policy_step)
        self._sync_cup_to_cart()
        self._update_ball_in_hand()

    def _user_pressed_release_key(self) -> bool:
        teleop = getattr(self, "_teleop_ref", None)
        if teleop is None:
            return False
        return self.RELEASE_KEY in getattr(teleop, "pressed_keys", set())

    def _release_ball(self):
        """Flip the latch, snap the gripper wide, and nudge the ball down so it drops freely."""
        self._ball_in_hand = False
        print("[INFO] Ball released from gripper")
        # Snap fingers wide (visual). The controller may close them again on
        # subsequent frames, but the ball has already cleared their grasp by
        # the time that happens.
        for j, q in zip(self.GRIPPER_FINGER_JOINTS,
                        (self.GRIPPER_RELEASE_QPOS, -self.GRIPPER_RELEASE_QPOS)):
            try:
                self.sim.data.set_joint_qpos(j, q)
                self.sim.data.set_joint_qvel(j, 0.0)
            except Exception:
                pass
        # Move the ball ~5 cm below the gripper site so it's free of the fingers.
        ball_joint = self.object_joints.get("ball")
        if ball_joint is not None:
            try:
                gp = self.sim.data.get_site_xpos(self.GRIPPER_SITE).copy()
                gp[2] -= 0.05
                self.sim.data.set_joint_qpos(
                    ball_joint, np.concatenate([gp, np.array([1.0, 0.0, 0.0, 0.0])])
                )
                qvel_addr = self.sim.model.get_joint_qvel_addr(ball_joint)
                if isinstance(qvel_addr, tuple):
                    start, end = qvel_addr
                    self.sim.data.qvel[start:end] = 0.0
                else:
                    self.sim.data.qvel[qvel_addr:qvel_addr + 6] = 0.0
            except Exception:
                pass

    def _sync_cup_to_cart(self):
        """Lock the cup's pose to the cart's pose (with a small +Z offset)."""
        cart_joint = self.object_joints.get("cart")
        cup_joint = self.object_joints.get("cup")
        if cart_joint is None or cup_joint is None:
            return
        try:
            cart_qpos = self.sim.data.get_joint_qpos(cart_joint).copy()
            cup_qpos = cart_qpos.copy()
            cup_qpos[2] = cart_qpos[2] + self.CUP_LOCAL_OFFSET_Z
            self.sim.data.set_joint_qpos(cup_joint, cup_qpos)
            qvel_addr = self.sim.model.get_joint_qvel_addr(cup_joint)
            if isinstance(qvel_addr, tuple):
                start, end = qvel_addr
                self.sim.data.qvel[start:end] = 0.0
            else:
                self.sim.data.qvel[qvel_addr:qvel_addr + 6] = 0.0
        except Exception:
            pass

    def _update_ball_in_hand(self):
        """While the latch is True, snap the ball to the gripper site and zero its velocity."""
        if not self._ball_in_hand:
            return
        ball_joint = self.object_joints.get("ball")
        if ball_joint is None:
            return
        try:
            gripper_pos = self.sim.data.get_site_xpos(self.GRIPPER_SITE)
            ball_qpos = np.concatenate([gripper_pos, np.array([1.0, 0.0, 0.0, 0.0])])
            self.sim.data.set_joint_qpos(ball_joint, ball_qpos)
            qvel_addr = self.sim.model.get_joint_qvel_addr(ball_joint)
            if isinstance(qvel_addr, tuple):
                start, end = qvel_addr
                self.sim.data.qvel[start:end] = 0.0
            else:
                self.sim.data.qvel[qvel_addr:qvel_addr + 6] = 0.0
        except Exception:
            pass

    def _load_model(self):
        super()._load_model()

        # Same table/robot geometry as ToyRedEnv
        self.table_offset = np.array((0.0, 0, 0.757))
        self.robot_base_z = 0.875
        self.table_size = np.array((1.3, 2.0, 0.05))
        arena = TableArena(
            table_full_size=self.table_size,
            table_offset=self.table_offset,
            has_legs=True,
        )
        arena.set_origin([0.0, 0.0, 0.0])
        arena.worldbody.append(
            new_element(
                "camera",
                name="custom_view",
                mode="fixed",
                pos="-1.67298636 0.79797356 1.75123554",
                quat="-0.44254917 -0.29815212 0.47254251 0.70139798",
                fovy="50",
            )
        )

        table_surface_z = self.table_offset[2]  # 0.757 — actual table surface

        # Cart: traverses the table along Y (replaces toybot's moving role)
        cart_obj = ToyCartObject(name="cart")
        # Cup: sits on the cart bed and rides along (synced in _pre_action)
        cup_obj = CupObject(name="cup")
        # Ball: small, the object to be dropped into the cup
        ball_obj = BallObject(
            name="ball",
            size=[0.025],
            rgba=[0.95, 0.50, 0.05, 1.0],
            density=500,
            friction=(1.5, 0.01, 0.001),
        )

        # No yaw on the cart so its long axis aligns with the Y traversal direction
        _object_quats = {}

        cart_x, cart_y = -0.20, 0.2
        objects = [
            # Cart starts at +Y end of the table; pattern mover sweeps it toward -Y
            ("cart", cart_obj, f"{cart_x:.4f}  {cart_y:.4f} {table_surface_z:.4f}"),
            # Cup spawns on the cart bed (cart_top + 0); sync keeps it there as cart moves
            ("cup",  cup_obj,  f"{cart_x:.4f}  {cart_y:.4f} {table_surface_z + self.CUP_LOCAL_OFFSET_Z:.4f}"),
            # Ball off the cart's path (different X), resting on the table
            ("ball", ball_obj, f"-0.40  0.40 {table_surface_z + 0.025:.4f}"),
        ]

        self.object_joints = {}
        for obj_name, obj, position in objects:
            obj_xml = obj.get_obj()
            obj_xml.set("pos", position)
            if obj_name in _object_quats:
                obj_xml.set("quat", _object_quats[obj_name])

            pos_array = np.array([float(x) for x in position.split()])
            self.object_base_positions[obj_name] = pos_array

            joint_name = None
            joints = getattr(obj, "joints", None)
            if joints:
                if isinstance(joints[0], str):
                    joint_name = joints[0]
                else:
                    joint_names = getattr(obj, "joint_names", None)
                    if joint_names:
                        joint_name = joint_names[0]
            setattr(self, f"{obj_name}_joint", joint_name)
            self.object_joints[obj_name] = joint_name

        # Robot base position (matches ToyRedEnv)
        robot_model = self.robots[0].robot_model
        root_body_name = robot_model.root_body
        if root_body_name:
            body_el = find_elements(
                root=robot_model.root, tags="body",
                attribs={"name": root_body_name}, return_first=True,
            )
            if body_el is not None:
                body_el.set("pos", f"-0.78 0.0 {self.robot_base_z}")

        mujoco_objects = [obj for _, obj, _ in objects]
        self.model = Task(
            mujoco_arena=arena,
            mujoco_robots=self.robots[0].robot_model,
            mujoco_objects=mujoco_objects,
        )

    def reward(self, action):
        return 0.0

    def check_success(self, verbose: bool = False):
        if self.task_success is None:
            print(f"[WARNING] No task success checker loaded for '{self.task_name}'")
            return False, 0.0, {}
        success, score, details = self.task_success.check_success(self, verbose=verbose)
        self.current_success = success
        self.success_checked = True
        return success, score, details

    def get_task_language(self) -> str:
        return self.task_success.task_description if self.task_success else self.task_name

    def get_task_id(self) -> int:
        return 11


register_env(DropBall)

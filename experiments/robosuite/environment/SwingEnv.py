#!/usr/bin/env python3
"""
Swing Pendulum Environment.

A fixed support post is mounted on the table. A rigid stick hangs from a hinge
at the top of the post, with a ball at its tip. The pendulum is released from a
set angle and swings freely under gravity — no mover needed.

Task: catch the swinging ball with the gripper.
"""

import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_robosuite_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _robosuite_dir)

import numpy as np
from robosuite.environments.base import register_env
from robosuite.models.tasks import Task
from robosuite.utils.mjcf_utils import find_elements, new_element
from environment.ToyRobotEnv import ToyRobot


# Pendulum geometry constants
_POST_HEIGHT   = 0.35   # metres — support post height
_CROSSBAR_X    = 0.12   # metres — crossbar reaches this far from post centre
_ARM_LENGTH    = 0.28   # metres — pivot-to-ball-centre length
_BALL_RADIUS   = 0.025  # metres — small ball


class Swing(ToyRobot):
    """
    Table + Panda arm + physics-driven pendulum swing.

    The pendulum support is embedded directly in the arena worldbody (no free
    joint) so it stays fixed.  The hinge joint "swing_hinge" lets the arm
    swing in the X-Z plane.  _reset_internal sets the release angle each
    episode; physics does the rest.
    """

    def _load_model(self):
        from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
        from robosuite.models.arenas import TableArena

        ManipulationEnv._load_model(self)

        # ── Table ──────────────────────────────────────────────────────────
        self.table_offset  = np.array((0.0, 0.0, 0.757))
        self.robot_base_z  = 0.875
        self.table_size    = np.array((1.3, 2.0, 0.05))

        arena = TableArena(
            table_full_size=self.table_size,
            table_offset=self.table_offset,
            has_legs=True,
        )
        arena.set_origin([0.0, 0.0, 0.0])

        # Custom overhead camera (same pose as ToyRed)
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

        # ── Pendulum structure (embedded in arena worldbody) ───────────────
        table_surface_z = self.table_offset[2]   # 0.757 m

        # Root support body — placed on table surface, no joint → fixed
        support = new_element(
            "body",
            name="pendulum_support",
            pos=f"0.0 0.15 {table_surface_z:.4f}",
        )

        # Post — collision (invisible)
        support.append(new_element(
            "geom",
            name="pendulum_post_col",
            pos=f"0 0 {_POST_HEIGHT / 2:.4f}",
            size=f"0.025 0.025 {_POST_HEIGHT / 2:.4f}",
            type="box",
            group="0",
            density="2000",
            friction="1.0 0.01 0.001",
            solref="0.02 1.0",
            solimp="0.9 0.95 0.001",
            rgba="0 0 0 0",
        ))

        # Post — visual (warm wood-brown)
        support.append(new_element(
            "geom",
            name="pendulum_post_vis",
            pos=f"0 0 {_POST_HEIGHT / 2:.4f}",
            size=f"0.023 0.023 {_POST_HEIGHT / 2:.4f}",
            type="box",
            group="1",
            contype="0",
            conaffinity="0",
            density="0",
            rgba="0.60 0.38 0.18 1",
        ))

        # Crossbar — horizontal beam from post top out to pivot (visual)
        support.append(new_element(
            "geom",
            name="pendulum_crossbar_vis",
            fromto=f"0 0 {_POST_HEIGHT:.4f} {_CROSSBAR_X:.4f} 0 {_POST_HEIGHT:.4f}",
            size="0.012",
            type="capsule",
            group="1",
            contype="0",
            conaffinity="0",
            density="0",
            rgba="0.60 0.38 0.18 1",
        ))

        # Axle — small cylinder at pivot end of crossbar (along Y, visual)
        support.append(new_element(
            "geom",
            name="pendulum_axle_vis",
            fromto=f"{_CROSSBAR_X:.4f} -0.030 {_POST_HEIGHT:.4f} {_CROSSBAR_X:.4f} 0.030 {_POST_HEIGHT:.4f}",
            size="0.010",
            type="cylinder",
            group="1",
            contype="0",
            conaffinity="0",
            density="0",
            rgba="0.75 0.75 0.78 1",
        ))

        # ── Pendulum arm body (pivot at end of crossbar, away from post) ───
        arm = new_element(
            "body",
            name="pendulum_arm",
            pos=f"{_CROSSBAR_X:.4f} 0 {_POST_HEIGHT:.4f}",
        )

        # Hinge joint — rotates around X, so the arm swings in Y-Z plane
        # (perpendicular to the crossbar, away from the post — no collision)
        arm.append(new_element(
            "joint",
            name="swing_hinge",
            type="hinge",
            axis="1 0 0",
            pos="0 0 0",
            damping="0.008",
            stiffness="0",
            frictionloss="0.002",
            limited="false",
        ))

        # Stick — collision (invisible, carries mass/friction for physics)
        arm.append(new_element(
            "geom",
            name="pendulum_stick_col",
            fromto=f"0 0 0 0 0 {-_ARM_LENGTH:.4f}",
            size="0.009",
            type="capsule",
            group="0",
            density="80",
            friction="0.8 0.005 0.001",
            rgba="0 0 0 0",
        ))
        # Stick — visual (colored, no collision/mass)
        arm.append(new_element(
            "geom",
            name="pendulum_stick_vis",
            fromto=f"0 0 0 0 0 {-_ARM_LENGTH:.4f}",
            size="0.009",
            type="capsule",
            group="1",
            contype="0",
            conaffinity="0",
            density="0",
            rgba="0.55 0.35 0.18 1",
        ))

        # ── Ball body at tip of stick ──────────────────────────────────────
        ball = new_element(
            "body",
            name="swing_ball_body",
            pos=f"0 0 {-_ARM_LENGTH:.4f}",
        )
        # Ball — collision (invisible)
        ball.append(new_element(
            "geom",
            name="swing_ball_col",
            pos="0 0 0",
            size=f"{_BALL_RADIUS:.4f}",
            type="sphere",
            group="0",
            density="350",
            friction="1.2 0.01 0.001",
            rgba="0 0 0 0",
        ))
        # Ball — visual (bright blue, no collision/mass)
        ball.append(new_element(
            "geom",
            name="swing_ball_vis",
            pos="0 0 0",
            size=f"{_BALL_RADIUS:.4f}",
            type="sphere",
            group="1",
            contype="0",
            conaffinity="0",
            density="0",
            rgba="0.15 0.45 0.92 1",
        ))
        arm.append(ball)
        support.append(arm)
        arena.worldbody.append(support)

        # ── Robot base position (same as ToyRed) ───────────────────────────
        robot_model    = self.robots[0].robot_model
        root_body_name = robot_model.root_body
        if root_body_name:
            body_elem = find_elements(
                root=robot_model.root,
                tags="body",
                attribs={"name": root_body_name},
                return_first=True,
            )
            if body_elem is not None:
                body_elem.set("pos", f"-0.78 0.0 {self.robot_base_z}")

        # No robosuite mujoco_objects — pendulum lives in the arena
        self.object_joints         = {}
        self.object_base_positions = {}

        self.model = Task(
            mujoco_arena=arena,
            mujoco_robots=self.robots[0].robot_model,
            mujoco_objects=[],
        )

    def _reset_internal(self):
        """Reset robot joints then release the pendulum from a random angle."""
        super()._reset_internal()   # handles robot joints, camera, sim.forward()

        # Set initial pendulum angle (degrees 30–72, randomised each episode)
        try:
            joint_id  = self.sim.model.joint_name2id("swing_hinge")
            qpos_addr = self.sim.model.jnt_qposadr[joint_id]
            dof_addr  = self.sim.model.jnt_dofadr[joint_id]

            angle = (
                np.random.uniform(np.pi / 6, np.pi / 2.5)
                if self.randomize_init
                else np.pi / 4
            )
            self.sim.data.qpos[qpos_addr] = angle
            self.sim.data.qvel[dof_addr]  = 0.0   # start from rest
            self.sim.forward()
            print(f"[INIT] Pendulum release angle: {np.degrees(angle):.1f}°")
        except Exception as e:
            print(f"[WARN] Could not set pendulum angle: {e}")

        # Store ball world position for reference
        try:
            ball_pos = self.sim.data.get_body_xpos("swing_ball_body")
            self.object_initial_heights["swing_ball_body"] = float(ball_pos[2])
            print(
                f"[INIT] Ball position: "
                f"[{ball_pos[0]:+.3f}, {ball_pos[1]:+.3f}, {ball_pos[2]:.3f}]"
            )
        except Exception as e:
            print(f"[WARN] Could not store ball position: {e}")


register_env(Swing)


if __name__ == "__main__":
    from tasks.swing import SWING_CONFIG, SWING_SUCCESS
    from env_render import EnvRender

    cfg = SWING_CONFIG

    runner = EnvRender(
        env_name=cfg.env_name,
        robot=cfg.robot,
        controller=cfg.controller,
        control_freq=cfg.control_freq,
        teleoperate_method=cfg.teleoperate_method,
        render_camera=cfg.render_camera,
        camera_names=cfg.camera_names,
        show_camera_obs=cfg.show_camera_obs,
        free_view_mode=cfg.free_view_mode,
        randomize_init=cfg.randomize_init,
        task_name=cfg.task_name,
        task_success=SWING_SUCCESS,
        # No mover — physics drives the pendulum
    )

    runner.run()

#!/usr/bin/env python3
"""
RandomMover - Move objects with random speed changes

The object moves along an axis, but every N seconds it randomly
changes to a different speed from a predefined list.
"""

import numpy as np
from typing import List, Optional


class RandomMover:
    """
    Move a named joint/object with random speed changes.

    Every `decision_interval` seconds, the object randomly switches
    to one of the predefined speeds.

    This creates unpredictable movement that requires visual feedback to track.
    """

    def __init__(
        self,
        env,
        joint_name,
        control_freq,
        speeds: Optional[List] = None,
        axis: str = 'y',
        decision_interval: float = 1.0,
        seed: int = None,
        stop_when_lifted: bool = True,
        lift_threshold: float = 0.02,  # 2cm lift to trigger stop
        # Legacy parameters (for backward compatibility)
        base_speed: float = None,
        move_probability: float = None,
    ):
        """
        Initialize RandomMover.

        Args:
            env: Robosuite environment object
            joint_name: Name of the joint to control (can be None if not available)
            control_freq: Control frequency in Hz
            speeds: List of possible speeds. Two formats:
                    - 1D (single axis): [0, -0.03, 0.03] — applied along `axis`
                    - 2D (multi-axis):  [[0, -0.03], [0.1, 0.04]] — [vx, vy] pairs
                    If None, defaults to [-0.03, 0, 0.03] (1D)
            axis: Axis to move along ('x', 'y', or 'z'). Used for 1D speeds only.
            decision_interval: How often to change speed (seconds)
            seed: Random seed for reproducibility (None for random)
            stop_when_lifted: If True, stop moving when object is lifted
            lift_threshold: Height change (m) to consider object lifted
            base_speed: [LEGACY] If provided and speeds is None, uses [0, base_speed]
            move_probability: [LEGACY] Ignored, kept for backward compatibility
        """
        self.env = env
        self.joint_name = joint_name
        self.control_freq = control_freq
        self.axis = axis.lower()
        self.decision_interval = decision_interval

        # Lift detection
        self.stop_when_lifted = stop_when_lifted
        self.lift_threshold = lift_threshold
        self.initial_z = None
        self.is_lifted = False

        # Handle speeds parameter
        if speeds is not None:
            self.speeds = list(speeds)
        elif base_speed is not None:
            # Legacy: convert base_speed to speeds list
            self.speeds = [0, base_speed]
        else:
            # Default speeds
            self.speeds = [-0.03, 0, 0.03]

        # Detect 2D mode: speeds contains lists/tuples instead of scalars
        self.is_2d = len(self.speeds) > 0 and isinstance(self.speeds[0], (list, tuple, np.ndarray))

        if self.is_2d:
            # Convert to numpy arrays for convenience
            self.speeds = [np.array(s, dtype=float) for s in self.speeds]

        # Random state
        self.rng = np.random.RandomState(seed)

        # Map axis to index (used for 1D mode only)
        self.axis_map = {'x': 0, 'y': 1, 'z': 2}
        self.axis_idx = self.axis_map.get(self.axis, 0)

        # State tracking
        self.step_count = 0
        self.steps_per_decision = int(decision_interval * control_freq)
        self.current_speed = self._random_speed()  # Start with random speed

        # Statistics
        self.total_decisions = 0
        self.speed_history = []

    def step(self):
        """Update target position - call this every simulation step."""
        if not self.joint_name:
            return

        self.step_count += 1

        # Check if object is lifted (stop moving if so)
        if self.stop_when_lifted and not self.is_lifted:
            try:
                qpos = self.env.sim.data.get_joint_qpos(self.joint_name)
                if qpos is not None and len(qpos) >= 3:
                    current_z = qpos[2]

                    # Store initial z on first step
                    if self.initial_z is None:
                        self.initial_z = current_z

                    # Check if lifted
                    if current_z - self.initial_z > self.lift_threshold:
                        self.is_lifted = True
            except Exception:
                pass

        # Don't move if object is lifted
        if self.is_lifted:
            return

        # Make decision every N steps
        if self.step_count % self.steps_per_decision == 0:
            self._make_decision()

        # Skip if speed is zero
        if self.is_2d:
            if np.allclose(self.current_speed, 0):
                return
        else:
            if abs(self.current_speed) < 1e-6:
                return

        try:
            qpos = self.env.sim.data.get_joint_qpos(self.joint_name)
            if qpos is None or len(qpos) < 3:
                return
            qpos = np.array(qpos, dtype=float)

            if self.is_2d:
                # 2D mode: apply speed pair to x and y axes
                qpos[0] += self.current_speed[0] / self.control_freq
                qpos[1] += self.current_speed[1] / self.control_freq
            else:
                # 1D mode: apply scalar speed along configured axis
                qpos[self.axis_idx] += self.current_speed / self.control_freq

            self.env.sim.data.set_joint_qpos(self.joint_name, qpos)
            self.env.sim.forward()
        except Exception:
            # Silently fail if joint doesn't exist or can't be moved
            pass

    def _random_speed(self):
        """Pick a random speed from the speeds list."""
        idx = self.rng.randint(len(self.speeds))
        return self.speeds[idx]

    def _make_decision(self):
        """Randomly choose a new speed from the speeds list."""
        self.total_decisions += 1
        self.current_speed = self._random_speed()
        self.speed_history.append(self.current_speed)

    def set_speeds(self, speeds: List[float]):
        """Change the list of possible speeds."""
        self.speeds = list(speeds)

    def set_axis(self, axis: str):
        """Change the movement axis."""
        self.axis = axis.lower()
        self.axis_idx = self.axis_map.get(self.axis, 0)

    def set_decision_interval(self, interval: float):
        """Change how often decisions are made (seconds)."""
        self.decision_interval = interval
        self.steps_per_decision = int(interval * self.control_freq)

    def reset(self, position=None, seed=None):
        """
        Reset mover state for new episode.

        Args:
            position: Optional position array [x, y, z]. If None, keeps current position.
            seed: Optional new random seed.
        """
        self.step_count = 0
        self.total_decisions = 0
        self.speed_history = []

        # Reset lift detection
        self.initial_z = None
        self.is_lifted = False

        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.current_speed = self._random_speed()

        if not self.joint_name:
            return
        if position is not None:
            try:
                qpos = np.array(position, dtype=float)
                self.env.sim.data.set_joint_qpos(self.joint_name, qpos)
                self.env.sim.forward()
            except Exception:
                pass

    def get_stats(self):
        """Get movement statistics."""
        return {
            "total_decisions": self.total_decisions,
            "current_speed": self.current_speed,
            "speeds": self.speeds,
            "speed_history": self.speed_history[-10:],  # Last 10 decisions
            "is_lifted": self.is_lifted,
            "initial_z": self.initial_z,
        }

#!/usr/bin/env python3
"""
PatternMover - Move objects with predefined speed patterns for controlled inference testing.

Instead of random speed selection, this mover cycles through predefined patterns
to ensure reproducible and comparable test scenarios.
"""

import numpy as np
from typing import List, Optional


# # Predefined test patterns
# INFERENCE_PATTERNS = [
#     [0.000, -0.040, 0.000, -0.040, -0.040, 0.000, -0.040, -0.040, -0.040, -0.040],
#     [0.000, -0.040, -0.040, -0.040, 0.000, 0.000, -0.040, -0.040, 0.000, 0.000],
#     [-0.040, -0.040, -0.040, -0.040, -0.040, 0.000, 0.000, -0.040, -0.040, -0.040],
#     # [-0.040, -0.040, 0.000, 0.000, 0.000, -0.040, 0.000, 0.000, -0.040, 0.000],
#     # [-0.040, 0.000, 0.000, 0.000, 0.000, -0.040, 0.000, -0.040, 0.000, 0.000],
#     [0.000, -0.040, 0.000, -0.040, -0.040, 0.000, -0.040, -0.040, -0.040, -0.040],
#     [0.000, 0.000, -0.040, 0.000, 0.000, -0.040, 0.000, -0.040, 0.000, 0.000],
#     [-0.040, 0.000, 0.000, 0.000, 0.000, -0.040, 0.000, -0.040, 0.000, 0.000],
#     # [-0.040, -0.040, -0.040, -0.040, 0.000, 0.000, -0.040, 0.000, 0.000, -0.040],
#     [-0.040, 0.000, -0.040, 0.000, -0.040, -0.040, -0.040, -0.040, 0.000, 0.000],
# ]


INFERENCE_PATTERNS = [ # red_toy
    # [0.000, -0.040, 0.000, -0.040, -0.040, 0.000, -0.040, -0.040, -0.040, -0.040], #first
    # [-0.040, 0.000, -0.040, 0.000, -0.040, -0.040, -0.040, -0.040, 0.000, 0.000], # second

    # [0.000, -0.040, 0.000, 0.000, 0.000, -0.040, 0.000, -0.040]
    # [0.000, -0.050, 0.000, 0.000, 0.300, -0.030, 0.000, -0.050] # unseen speed

]


# INFERENCE_PATTERNS = [ # green_toy, should use pattern 0 only, pattern 1 is too hard

#     # Pattern 0 — 2D: right-diag, stop, right-diag, left-diag, right-diag, stop, left-diag×2, stop×2
#     [
#         [ 0.012, -0.038], [0.000, 0.000], [ 0.012, -0.038], [-0.012, -0.038], [ 0.012, -0.038],
#         [ 0.000,  0.000], [-0.012, -0.038], [-0.012, -0.038], [0.000,  0.000], [ 0.000,  0.000],
#     ],

#     # # Pattern 1 — 2D: right-diag, left-diag×3, stop×3, left-diag, right-diag, left-diag
#     # [
#     #     [ 0.012, -0.038], [-0.012, -0.038], [-0.012, -0.038], [-0.012, -0.038], [0.000, 0.000],
#     #     [ 0.000,  0.000], [ 0.000,  0.000], [-0.012, -0.038], [ 0.012, -0.038], [-0.012, -0.038],
#     # ],
# ]


class PatternMover:
    """
    Move a named joint/object with predefined speed patterns.

    Each episode uses a specific pattern from the list, cycling through them.
    This ensures reproducible test scenarios for comparing model performance.
    """

    def __init__(
        self,
        env,
        joint_name,
        control_freq,
        patterns: Optional[List[List[float]]] = None,
        pattern_id: int = 0,
        axis: str = 'y',
        decision_interval: float = 1.0,
        stop_when_lifted: bool = True,
        lift_threshold: float = 0.02,
    ):
        """
        Initialize PatternMover.

        Args:
            env: Robosuite environment object
            joint_name: Name of the joint to control
            control_freq: Control frequency in Hz
            patterns: List of speed patterns. If None, uses INFERENCE_PATTERNS
            pattern_id: Which pattern to use (0-indexed, wraps around)
            axis: Axis to move along ('x', 'y', or 'z')
            decision_interval: How often to change speed (seconds)
            stop_when_lifted: If True, stop moving when object is lifted
            lift_threshold: Height change (m) to consider object lifted
        """
        self.env = env
        self.joint_name = joint_name
        self.control_freq = control_freq
        self.axis = axis.lower()
        self.decision_interval = decision_interval

        # Patterns
        self.patterns = patterns if patterns is not None else INFERENCE_PATTERNS
        self.pattern_id = pattern_id % len(self.patterns)
        self.current_pattern = self.patterns[self.pattern_id]

        # Lift detection
        self.stop_when_lifted = stop_when_lifted
        self.lift_threshold = lift_threshold
        self.initial_z = None
        self.is_lifted = False

        # Map axis to index
        self.axis_map = {'x': 0, 'y': 1, 'z': 2}
        self.axis_idx = self.axis_map.get(self.axis, 0)

        # State tracking
        self.step_count = 0
        self.steps_per_decision = int(decision_interval * control_freq)
        self.decision_count = 0
        self.current_speed = self.current_pattern[0] if self.current_pattern else 0.0

        # Statistics
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
            self._next_speed()

        # Skip if speed is zero
        speed = self.current_speed
        if isinstance(speed, (list, np.ndarray)):
            if np.max(np.abs(speed)) < 1e-6:
                return
        elif abs(speed) < 1e-6:
            return

        try:
            qpos = self.env.sim.data.get_joint_qpos(self.joint_name)
            if qpos is None or len(qpos) < 3:
                return
            qpos = np.array(qpos, dtype=float)

            # Update position along specified axis (or axes for 2D patterns)
            if isinstance(speed, (list, np.ndarray)):
                for i, s in enumerate(speed):
                    qpos[i] += s / self.control_freq
            else:
                qpos[self.axis_idx] += speed / self.control_freq

            self.env.sim.data.set_joint_qpos(self.joint_name, qpos)
            self.env.sim.forward()
        except Exception:
            pass

    def _next_speed(self):
        """Get next speed from the pattern (cycles if pattern ends)."""
        if not self.current_pattern:
            return

        # Get speed at current position in pattern
        pattern_idx = self.decision_count % len(self.current_pattern)
        self.current_speed = self.current_pattern[pattern_idx]
        self.speed_history.append(self.current_speed)
        self.decision_count += 1

    def set_pattern(self, pattern_id: int):
        """Switch to a different pattern."""
        self.pattern_id = pattern_id % len(self.patterns)
        self.current_pattern = self.patterns[self.pattern_id]

    def reset(self, position=None, pattern_id: Optional[int] = None, seed=None):
        """
        Reset mover state for new episode.

        Args:
            position: Optional position array. If None, keeps current position.
            pattern_id: Pattern to use for this episode. If None, uses current.
            seed: Ignored (for compatibility with RandomMover interface)
        """
        self.step_count = 0
        self.decision_count = 0
        self.speed_history = []

        # Reset lift detection
        self.initial_z = None
        self.is_lifted = False

        # Set pattern if specified
        if pattern_id is not None:
            self.pattern_id = pattern_id % len(self.patterns)
            self.current_pattern = self.patterns[self.pattern_id]

        # Start with first speed in pattern
        self.current_speed = self.current_pattern[0] if self.current_pattern else 0.0

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
            "pattern_id": self.pattern_id,
            "pattern": self.current_pattern,
            "decision_count": self.decision_count,
            "current_speed": self.current_speed,
            "speed_history": self.speed_history,
            "is_lifted": self.is_lifted,
            "initial_z": self.initial_z,
        }

#!/usr/bin/env python3
"""
TargetMover - Tool for moving objects at constant speed in robosuite
"""

import numpy as np


class LinearMover:
    """Move a named joint/object at a constant speed along a specified axis."""
    
    def __init__(self, env, joint_name, control_freq, speed_y=0.0, axis='y'):
        """
        Initialize TargetMover
        
        Args:
            env: Robosuite environment object
            joint_name: Name of the joint to control (can be None if not available)
            control_freq: Control frequency in Hz
            speed_y: Speed along Y axis (units per second, negative = -Y direction)
            axis: Axis to move along ('x', 'y', or 'z')
        """
        self.env = env
        self.joint_name = joint_name
        self.control_freq = control_freq
        self.speed_y = speed_y
        self.axis = axis.lower()
        
        # Map axis to index
        self.axis_map = {'x': 0, 'y': 1, 'z': 2}
        self.axis_idx = self.axis_map.get(self.axis, 1)
    
    def step(self):
        """Update target position - call this every simulation step"""
        if not self.joint_name:
            return
        
        try:
            qpos = self.env.sim.data.get_joint_qpos(self.joint_name)
            if qpos is None or len(qpos) < 3:
                return
            qpos = np.array(qpos, dtype=float)
            
            # Update position along specified axis
            qpos[self.axis_idx] += self.speed_y / self.control_freq
            
            self.env.sim.data.set_joint_qpos(self.joint_name, qpos)
            self.env.sim.forward()
        except Exception:
            # Silently fail if joint doesn't exist or can't be moved
            pass
    
    def set_speed(self, speed_y):
        """Change the movement speed"""
        self.speed_y = speed_y
    
    def set_axis(self, axis):
        """Change the movement axis"""
        self.axis = axis.lower()
        self.axis_idx = self.axis_map.get(self.axis, 1)
    
    def get_stats(self):
        """Return mover stats (compatible interface with RandomMover/PatternMover)."""
        return {"speed_history": [self.speed_y]}

    def reset(self, position=None):
        """
        Reset target to a specific position
        
        Args:
            position: Optional position array [x, y, z]. If None, keeps current position.
        """
        if not self.joint_name:
            return
        if position is not None:
            try:
                qpos = np.array(position, dtype=float)
                self.env.sim.data.set_joint_qpos(self.joint_name, qpos)
                self.env.sim.forward()
            except Exception:
                pass


"""
Base classes for task definitions.

TaskConfig: Defines environment and controller settings for a task
TaskSuccess: Defines success criteria for a task
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Optional
from abc import ABC, abstractmethod


@dataclass
class TaskConfig:
    """
    Base configuration class for tasks.
    
    Each task should create a subclass with its specific settings.
    This defines which environment to use and how to configure it.
    """
    # Task identification
    task_name: str = "base_task"
    task_description: str = "Base task"
    task_id: int = 0
    
    # Environment settings
    env_name: str = "TableLadle"
    robot: str = "Panda"
    controller: str = "BASIC"
    control_freq: int = 10  # Hz
    
    # Teleoperation settings (for data collection)
    teleoperate_method: str = "spacemouse"  # or "keyboard"
    
    # Rendering and cameras
    render_camera: str = "custom_view"
    camera_names: List[str] = field(default_factory=lambda: ["robot0_eye_in_hand", "agentview"])
    camera_height: int = 256
    camera_width: int = 256
    show_camera_obs: bool = True
    free_view_mode: bool = False
    
    # Task-specific dynamics
    target_joint_name: Optional[str] = None
    target_speed_y: float = 0.0
    target_axis: str = "y"
    mover_type: str = "linear"  # 'linear' or 'random'
    decision_interval: float = 1.0  # For random mover: seconds between speed changes
    speeds: Optional[List[float]] = None  # For random mover: list of speeds to choose from

    # Training settings
    randomize_init: bool = True  # LIBERO-style randomization
    save_data: bool = True  # Enable/disable data saving
    max_episode_steps: int = 500
    
    # Difficulty and tags
    difficulty: str = "medium"  # easy, medium, hard
    tags: List[str] = field(default_factory=list)
    
    def to_env_kwargs(self) -> Dict:
        """
        Convert config to environment creation kwargs.
        
        Returns:
            Dictionary of kwargs for suite.make() or EnvRender()
        """
        return {
            "env_name": self.env_name,
            "robot": self.robot,
            "controller": self.controller,
            "control_freq": self.control_freq,
            "teleoperate_method": self.teleoperate_method,
            "render_camera": self.render_camera,
            "camera_names": self.camera_names,
            "camera_height": self.camera_height,
            "camera_width": self.camera_width,
            "show_camera_obs": self.show_camera_obs,
            "free_view_mode": self.free_view_mode,
            "target_joint_name": self.target_joint_name,
            "target_speed_y": self.target_speed_y,
            "target_axis": self.target_axis,
            "mover_type": self.mover_type,
            "decision_interval": self.decision_interval,
            "speeds": self.speeds,
            "randomize_init": self.randomize_init,
            "task_name": self.task_name,
        }


class TaskSuccess(ABC):
    """
    Base class for defining task success criteria.
    
    Each task should create a subclass that implements check_success().
    """
    
    def __init__(self, task_name: str, task_description: str):
        self.task_name = task_name
        self.task_description = task_description
        self.success_conditions: List[Dict] = []
    
    @abstractmethod
    def check_success(self, env, verbose: bool = False) -> Tuple[bool, float, Dict]:
        """
        Check if the task has been completed successfully.
        
        Args:
            env: The robosuite environment
            verbose: If True, print detailed success information
            
        Returns:
            (success, score, details):
                - success: Boolean indicating overall success
                - score: Float between 0-1 indicating partial success
                - details: Dict with per-condition results
        """
        pass
    
    def add_condition(
        self,
        name: str,
        check_fn: Callable,
        weight: float = 1.0,
        required: bool = True
    ):
        """
        Add a success condition.
        
        Args:
            name: Human-readable name
            check_fn: Function that takes env and returns bool
            weight: Weight for scoring (default 1.0)
            required: Whether this condition is required (default True)
        """
        self.success_conditions.append({
            "name": name,
            "check_fn": check_fn,
            "weight": weight,
            "required": required,
        })
    
    def evaluate_conditions(
        self,
        env,
        success_threshold: float = 1.0,
        verbose: bool = False
    ) -> Tuple[bool, float, Dict]:
        """
        Evaluate all success conditions.
        
        Args:
            env: The robosuite environment
            success_threshold: Fraction of weighted score needed (0-1)
            verbose: If True, print detailed results
            
        Returns:
            (success, score, details)
        """
        results = {}
        total_weight = 0.0
        achieved_weight = 0.0
        
        for condition in self.success_conditions:
            if not condition["required"]:
                continue
            
            try:
                success = condition["check_fn"](env)
                msg = f"✓ {condition['name']}" if success else f"✗ {condition['name']}"
            except Exception as e:
                success = False
                msg = f"✗ {condition['name']} (Error: {e})"
            
            results[condition["name"]] = {
                "success": success,
                "message": msg,
                "weight": condition["weight"],
                "required": condition["required"],
            }
            
            total_weight += condition["weight"]
            if success:
                achieved_weight += condition["weight"]
            
            if verbose:
                print(msg)
        
        # Calculate overall success
        score = achieved_weight / total_weight if total_weight > 0 else 0.0
        overall_success = score >= success_threshold
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Task: {self.task_description}")
            print(f"Score: {score:.2%} (threshold: {success_threshold:.2%})")
            print(f"Result: {'✓ SUCCESS' if overall_success else '✗ FAILURE'}")
            print(f"{'='*50}")
        
        return overall_success, score, results

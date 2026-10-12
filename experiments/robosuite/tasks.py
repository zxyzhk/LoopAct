"""
Task definitions and success checking for robotic manipulation.

This module provides a flexible system for defining tasks with:
- Initial conditions (object positions, robot state)
- Success criteria (spatial relationships, object states)
- Task metadata (name, description, difficulty)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Tuple
import numpy as np


@dataclass
class TaskObject:
    """Defines an object in the task."""
    name: str  # Object name in simulation (e.g., "ladle_1_main")
    type: str  # Object type (e.g., "ladle", "pot", "bowl")
    initial_pos: Optional[np.ndarray] = None  # Initial position (x, y, z)
    initial_quat: Optional[np.ndarray] = None  # Initial quaternion
    randomize: bool = True  # Whether to randomize initial position


@dataclass
class SuccessCondition:
    """Defines a single success condition."""
    name: str  # Human-readable name
    check_fn: Callable  # Function that takes env and returns bool
    required: bool = True  # If False, this is optional (bonus points)
    weight: float = 1.0  # Weight for partial success scoring
    
    def evaluate(self, env) -> Tuple[bool, str]:
        """
        Evaluate the condition.
        
        Returns:
            (success, message): Whether condition is met and explanation
        """
        try:
            success = self.check_fn(env)
            msg = f"✓ {self.name}" if success else f"✗ {self.name}"
            return success, msg
        except Exception as e:
            return False, f"✗ {self.name} (Error: {e})"


@dataclass
class Task:
    """
    Defines a complete manipulation task.
    
    Example:
        task = Task(
            name="pick_up_ladle",
            description="Pick up the ladle from the table",
            objects=[...],
            success_conditions=[...],
        )
    """
    # Task metadata
    name: str  # Machine-readable name (e.g., "pick_up_ladle")
    description: str  # Human-readable description
    task_id: int = 0  # Unique task ID for multi-task learning
    
    # Environment setup
    objects: List[TaskObject] = field(default_factory=list)
    robot_initial_qpos: Optional[np.ndarray] = None
    
    # Success criteria
    success_conditions: List[SuccessCondition] = field(default_factory=list)
    success_threshold: float = 1.0  # Fraction of conditions that must be met (0-1)
    
    # Additional metadata
    difficulty: str = "medium"  # easy, medium, hard
    max_episode_steps: int = 500  # Maximum steps before timeout
    tags: List[str] = field(default_factory=list)  # e.g., ["pick", "place", "bimanual"]
    
    def check_success(self, env, verbose: bool = False) -> Tuple[bool, float, Dict]:
        """
        Check if the task has been completed successfully.
        
        Args:
            env: The robosuite environment
            verbose: If True, print detailed success check results
            
        Returns:
            (success, score, details): 
                - success: Boolean indicating overall success
                - score: Float between 0-1 indicating partial success
                - details: Dict with per-condition results
        """
        results = {}
        total_weight = 0.0
        achieved_weight = 0.0
        
        for condition in self.success_conditions:
            if not condition.required:
                continue
                
            success, message = condition.evaluate(env)
            results[condition.name] = {
                "success": success,
                "message": message,
                "weight": condition.weight,
                "required": condition.required,
            }
            
            total_weight += condition.weight
            if success:
                achieved_weight += condition.weight
            
            if verbose:
                print(message)
        
        # Calculate overall success
        score = achieved_weight / total_weight if total_weight > 0 else 0.0
        overall_success = score >= self.success_threshold
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Task: {self.description}")
            print(f"Score: {score:.2%} (threshold: {self.success_threshold:.2%})")
            print(f"Result: {'✓ SUCCESS' if overall_success else '✗ FAILURE'}")
            print(f"{'='*50}")
        
        return overall_success, score, results
    
    def get_task_language(self) -> str:
        """Get natural language description for language-conditioned policies."""
        return self.description
    
    def get_task_id(self) -> int:
        """Get task ID for multi-task learning."""
        return self.task_id


# =============================================================================
# Reusable Success Check Functions
# =============================================================================

def check_object_grasped(env, object_name: str, gripper_site: str = "gripper0_grip_site", 
                         threshold: float = 0.08) -> bool:
    """Check if an object is grasped by the gripper."""
    try:
        obj_pos = env.sim.data.get_body_xpos(object_name)
        gripper_pos = env.sim.data.get_site_xpos(gripper_site)
        dist = np.linalg.norm(obj_pos - gripper_pos)
        
        # Also check gripper is closed
        gripper_qpos = env.robots[0].gripper[env.robots[0].eef_name].current_action
        gripper_closed = gripper_qpos > 0.5  # Gripper action > 0.5 means closed
        
        return dist < threshold and gripper_closed
    except:
        return False


def check_object_lifted(env, object_name: str, min_height: float = 0.92) -> bool:
    """Check if an object is lifted above a certain height."""
    try:
        obj_pos = env.sim.data.get_body_xpos(object_name)
        return obj_pos[2] > min_height
    except:
        return False


def check_object_above_target(env, object_name: str, target_name: str, 
                              threshold: float = 0.1) -> bool:
    """Check if an object is above a target (within XY threshold)."""
    try:
        obj_pos = env.sim.data.get_body_xpos(object_name)
        target_pos = env.sim.data.get_body_xpos(target_name)
        
        # Check XY distance
        xy_dist = np.linalg.norm(obj_pos[:2] - target_pos[:2])
        
        # Check Z is above
        z_above = obj_pos[2] > target_pos[2]
        
        return xy_dist < threshold and z_above
    except:
        return False


def check_object_in_container(env, object_name: str, container_name: str,
                              xy_threshold: float = 0.08, z_threshold: float = 0.15) -> bool:
    """Check if an object is inside a container."""
    try:
        obj_pos = env.sim.data.get_body_xpos(object_name)
        container_pos = env.sim.data.get_body_xpos(container_name)
        
        # Check XY distance (must be close)
        xy_dist = np.linalg.norm(obj_pos[:2] - container_pos[:2])
        
        # Check Z is slightly above container bottom
        z_in_range = (container_pos[2] < obj_pos[2] < container_pos[2] + z_threshold)
        
        return xy_dist < xy_threshold and z_in_range
    except:
        return False


def check_object_near_target(env, object_name: str, target_name: str,
                             threshold: float = 0.15) -> bool:
    """Check if an object is near a target."""
    try:
        obj_pos = env.sim.data.get_body_xpos(object_name)
        target_pos = env.sim.data.get_body_xpos(target_name)
        dist = np.linalg.norm(obj_pos - target_pos)
        return dist < threshold
    except:
        return False


def check_gripper_open(env, threshold: float = -0.5) -> bool:
    """Check if gripper is open."""
    try:
        gripper_qpos = env.robots[0].gripper[env.robots[0].eef_name].current_action
        return gripper_qpos < threshold
    except:
        return False


def check_gripper_closed(env, threshold: float = 0.5) -> bool:
    """Check if gripper is closed."""
    try:
        gripper_qpos = env.robots[0].gripper[env.robots[0].eef_name].current_action
        return gripper_qpos > threshold
    except:
        return False


# =============================================================================
# Task Registry
# =============================================================================

class TaskRegistry:
    """
    Central registry for all tasks.
    
    Usage:
        registry = TaskRegistry()
        registry.register(task)
        task = registry.get("pick_up_ladle")
    """
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self._initialize_default_tasks()
    
    def register(self, task: Task):
        """Register a new task."""
        self.tasks[task.name] = task
        print(f"Registered task: {task.name} - {task.description}")
    
    def get(self, name: str) -> Task:
        """Get a task by name."""
        if name not in self.tasks:
            raise ValueError(f"Task '{name}' not found. Available: {list(self.tasks.keys())}")
        return self.tasks[name]
    
    def list_tasks(self) -> List[str]:
        """List all registered task names."""
        return list(self.tasks.keys())
    
    def list_tasks_detailed(self):
        """Print detailed information about all tasks."""
        print("\n" + "="*70)
        print("REGISTERED TASKS")
        print("="*70)
        for i, (name, task) in enumerate(self.tasks.items(), 1):
            print(f"{i}. {name}")
            print(f"   Description: {task.description}")
            print(f"   Difficulty: {task.difficulty}")
            print(f"   Success conditions: {len(task.success_conditions)}")
            print(f"   Objects: {len(task.objects)}")
            print()
    
    def _initialize_default_tasks(self):
        """Initialize default tasks."""
        # Task 1: Pick up the ladle
        pick_ladle = Task(
            name="pick_up_ladle",
            description="Pick up the ladle from the table",
            task_id=0,
            objects=[
                TaskObject(name="ladle_1_main", type="ladle", randomize=True),
            ],
            success_conditions=[
                SuccessCondition(
                    name="Ladle is grasped",
                    check_fn=lambda env: check_object_grasped(env, "ladle_1_main"),
                    weight=1.0,
                ),
                SuccessCondition(
                    name="Ladle is lifted",
                    check_fn=lambda env: check_object_lifted(env, "ladle_1_main", min_height=0.95),
                    weight=1.0,
                ),
            ],
            success_threshold=1.0,  # Both conditions must be met
            difficulty="easy",
            max_episode_steps=500,
            tags=["pick", "single_object"],
        )
        self.register(pick_ladle)
        
        # Task 2: Pick up ladle and place in pot
        ladle_to_pot = Task(
            name="place_ladle_in_pot",
            description="Pick up the ladle and place it in the pot",
            task_id=1,
            objects=[
                TaskObject(name="ladle_1_main", type="ladle", randomize=True),
                TaskObject(name="pot_1_main", type="pot", randomize=True),
            ],
            success_conditions=[
                SuccessCondition(
                    name="Ladle is near pot",
                    check_fn=lambda env: check_object_in_container(env, "ladle_1_main", "pot_1_main"),
                    weight=1.0,
                ),
                SuccessCondition(
                    name="Gripper is open (released ladle)",
                    check_fn=lambda env: check_gripper_open(env),
                    weight=1.0,
                ),
            ],
            success_threshold=1.0,
            difficulty="medium",
            max_episode_steps=1000,
            tags=["pick", "place", "two_objects"],
        )
        self.register(ladle_to_pot)
        
        # Task 3: Pick up plate (simpler version for testing)
        pick_plate = Task(
            name="pick_up_plate",
            description="Pick up the plate from the table",
            task_id=2,
            objects=[
                TaskObject(name="plate_1_main", type="plate", randomize=True),
            ],
            success_conditions=[
                SuccessCondition(
                    name="Plate is grasped",
                    check_fn=lambda env: check_object_grasped(env, "plate_1_main", threshold=0.1),
                    weight=1.0,
                ),
                SuccessCondition(
                    name="Plate is lifted",
                    check_fn=lambda env: check_object_lifted(env, "plate_1_main", min_height=0.95),
                    weight=1.0,
                ),
            ],
            success_threshold=1.0,
            difficulty="easy",
            max_episode_steps=500,
            tags=["pick", "single_object"],
        )
        self.register(pick_plate)
        
        # You can add more tasks here as you develop them


# Global task registry instance
TASK_REGISTRY = TaskRegistry()


# =============================================================================
# Helper Functions
# =============================================================================

def get_task(name: str) -> Task:
    """Convenience function to get a task."""
    return TASK_REGISTRY.get(name)


def list_tasks() -> List[str]:
    """Convenience function to list all tasks."""
    return TASK_REGISTRY.list_tasks()


def check_task_success(env, task_name: str, verbose: bool = False) -> Tuple[bool, float, Dict]:
    """Convenience function to check task success."""
    task = get_task(task_name)
    return task.check_success(env, verbose=verbose)

"""
Reusable success checking functions for robotic manipulation tasks.

These functions can be used to define success criteria for tasks.
"""

import numpy as np
from typing import Optional


def check_object_grasped(
    env,
    object_name: str,
    gripper_site: str = "gripper0_grip_site",
    threshold: float = 0.08
) -> bool:
    """
    Check if an object is grasped by the gripper.
    
    Args:
        env: The robosuite environment
        object_name: Name of the object body (e.g., "ladle_1_main")
        gripper_site: Name of the gripper site
        threshold: Maximum distance for grasp detection (meters)
        
    Returns:
        True if object is grasped
    """
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


def check_object_lifted(
    env,
    object_name: str,
    min_height: float = 0.92
) -> bool:
    """
    Check if an object is lifted above a certain height.
    
    Args:
        env: The robosuite environment
        object_name: Name of the object body
        min_height: Minimum height threshold (meters)
        
    Returns:
        True if object is above the threshold
    """
    try:
        obj_pos = env.sim.data.get_body_xpos(object_name)
        return obj_pos[2] > min_height
    except:
        return False


def check_object_lifted_relative(
    env,
    object_name: str,
    min_lift_height: float = 0.05,
    verbose: bool = False
) -> bool:
    """
    Check if an object has been lifted by a certain amount from its initial position.
    Requires the environment to store initial object positions.
    
    Args:
        env: The robosuite environment
        object_name: Name of the object body (e.g., "ladle_1_main")
        min_lift_height: Minimum relative lift height (meters)
        verbose: If True, print debug information
        
    Returns:
        True if object has been lifted by at least min_lift_height
    """
    try:
        # Get current position
        current_pos = env.sim.data.get_body_xpos(object_name)
        current_z = current_pos[2]
        
        # Get initial position (stored during reset)
        # Look for the object in object_base_positions (without the suffix)
        initial_z = None
        if hasattr(env, 'object_initial_heights'):
            # Direct storage of initial heights
            initial_z = env.object_initial_heights.get(object_name)
        elif hasattr(env, 'object_base_positions'):
            # Try to find base name (e.g., "ladle_1_main" -> "ladle")
            for obj_key, base_pos in env.object_base_positions.items():
                if obj_key in object_name or object_name.startswith(obj_key):
                    initial_z = base_pos[2]
                    break
        
        if initial_z is None:
            # Fallback: use a reasonable default based on table height
            # Table top is typically around 0.875-0.9m
            initial_z = 0.887  # Default ladle height
            if verbose:
                print(f"[DEBUG] No initial height stored for {object_name}, using default: {initial_z:.3f}m")
        
        # Check if lifted by the required amount
        lift_amount = current_z - initial_z
        
        # Debug output
        if verbose:
            print(f"[SUCCESS CHECK] {object_name}:")
            print(f"  Initial Z: {initial_z:.3f}m")
            print(f"  Current Z: {current_z:.3f}m")
            print(f"  Lift amount: {lift_amount:.3f}m (required: {min_lift_height:.3f}m)")
            print(f"  Success: {lift_amount >= min_lift_height}")
        
        return lift_amount >= min_lift_height
    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to check lift: {e}")
        return False


def check_object_continue_lifted(
    env,
    object_name: str,
    min_consecutive: int = 3,
    verbose: bool = False,
) -> bool:
    """
    Check if an object's height is continuously increasing.

    Tracks the object's Z position across consecutive calls and returns
    True when Z has risen for min_consecutive calls in a row. On a slope,
    an ungrasped object can only slide down, so a sustained Z increase
    is strong evidence the robot is lifting it.

    State is stored on the env as env._lift_history[object_name].

    Args:
        env: The robosuite environment
        object_name: Name of the object body (e.g., "cylinder_main")
        min_consecutive: How many consecutive Z increases needed
        verbose: If True, print debug information

    Returns:
        True if the object Z has been increasing for min_consecutive
        consecutive checks.
    """
    try:
        current_z = env.sim.data.get_body_xpos(object_name)[2]

        if not hasattr(env, "_lift_history"):
            env._lift_history = {}
        history = env._lift_history.setdefault(object_name, {"prev_z": None, "streak": 0, "latched": False})

        # Once success is detected, latch it so post-episode re-checks stay True
        if history["latched"]:
            if verbose:
                print(f"[LIFT CHECK] {object_name}: latched SUCCESS")
            return True

        prev_z = history["prev_z"]
        if prev_z is not None and current_z > prev_z:
            history["streak"] += 1
        else:
            history["streak"] = 0
        history["prev_z"] = current_z

        success = history["streak"] >= min_consecutive
        if success:
            history["latched"] = True

        if verbose:
            delta = (current_z - prev_z) if prev_z is not None else 0.0
            print(f"[LIFT CHECK] {object_name}:")
            print(f"  Z: {current_z:.4f}m (delta: {delta:+.4f}m)")
            print(f"  Streak: {history['streak']}/{min_consecutive}")
            print(f"  Result: {success}")

        return success
    except Exception as e:
        if verbose:
            print(f"[ERROR] check_object_continue_lifted failed: {e}")
        return False


def check_object_lifted_off_surface(
    env,
    object_name: str,
    surface_geoms: list,
    min_consecutive: int = 5,
    min_height: float = 0.88,
    verbose: bool = False,
) -> bool:
    """
    Check if an object has been lifted off one or more named surface geoms.

    Uses MuJoCo contact data to verify the object body has no active contact
    with any of the listed surface geoms. Requires min_consecutive consecutive
    frames of no-contact AND Z above min_height (to exclude the cylinder having
    fallen off the table). Latches on success like check_object_continue_lifted.

    State is stored on the env as env._surface_lift_history[object_name].

    Args:
        env: The robosuite environment
        object_name: Name of the object body (e.g., "cylinder_main")
        surface_geoms: List of surface geom names to check contact against
                       (e.g., ["slope_collision", "table_collision"])
        min_consecutive: Frames of no-contact needed to declare success
        min_height: Minimum Z the object must exceed (excludes floor-fallen objects)
        verbose: If True, print debug information

    Returns:
        True if the object has been lifted clear of all listed surfaces
    """
    try:
        if not hasattr(env, "_surface_lift_history"):
            env._surface_lift_history = {}
        history = env._surface_lift_history.setdefault(
            object_name, {"streak": 0, "latched": False}
        )

        if history["latched"]:
            if verbose:
                print(f"[SURFACE LIFT] {object_name}: latched SUCCESS")
            return True

        # Geom IDs for the object body
        obj_body_id = env.sim.model.body_name2id(object_name)
        geom_start = env.sim.model.body_geomadr[obj_body_id]
        geom_count = env.sim.model.body_geomnum[obj_body_id]
        obj_geom_ids = set(range(geom_start, geom_start + geom_count))

        # Geom IDs for the named surfaces
        surface_geom_ids = set()
        for gname in surface_geoms:
            try:
                surface_geom_ids.add(env.sim.model.geom_name2id(gname))
            except Exception:
                pass  # geom not present in this env variant

        # Scan active contacts
        in_contact = False
        for i in range(env.sim.data.ncon):
            c = env.sim.data.contact[i]
            if (c.geom1 in surface_geom_ids and c.geom2 in obj_geom_ids) or \
               (c.geom2 in surface_geom_ids and c.geom1 in obj_geom_ids):
                in_contact = True
                break

        current_z = env.sim.data.get_body_xpos(object_name)[2]
        above_floor = current_z > min_height

        if not in_contact and above_floor:
            history["streak"] += 1
        else:
            history["streak"] = 0

        success = history["streak"] >= min_consecutive
        if success:
            history["latched"] = True

        if verbose:
            print(f"[SURFACE LIFT] {object_name}:")
            print(f"  Surfaces: {surface_geoms}")
            print(f"  In contact with surface: {in_contact}")
            print(f"  Z: {current_z:.4f}m (above floor {min_height:.3f}m: {above_floor})")
            print(f"  Streak: {history['streak']}/{min_consecutive}")
            print(f"  Result: {success}")

        return success

    except Exception as e:
        if verbose:
            print(f"[ERROR] check_object_lifted_off_surface failed: {e}")
        return False


def check_object_above_target(
    env,
    object_name: str,
    target_name: str,
    threshold: float = 0.1
) -> bool:
    """
    Check if an object is above a target (within XY threshold).
    
    Args:
        env: The robosuite environment
        object_name: Name of the object body
        target_name: Name of the target body
        threshold: Maximum XY distance (meters)
        
    Returns:
        True if object is above target
    """
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


def check_object_in_container(
    env,
    object_name: str,
    container_name: str,
    xy_threshold: float = 0.08,
    z_threshold: float = 0.15
) -> bool:
    """
    Check if an object is inside a container.
    
    Args:
        env: The robosuite environment
        object_name: Name of the object body
        container_name: Name of the container body
        xy_threshold: Maximum XY distance (meters)
        z_threshold: Maximum Z distance above container (meters)
        
    Returns:
        True if object is in container
    """
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


def check_object_near_target(
    env,
    object_name: str,
    target_name: str,
    threshold: float = 0.15
) -> bool:
    """
    Check if an object is near a target.
    
    Args:
        env: The robosuite environment
        object_name: Name of the object body
        target_name: Name of the target body
        threshold: Maximum distance (meters)
        
    Returns:
        True if object is near target
    """
    try:
        obj_pos = env.sim.data.get_body_xpos(object_name)
        target_pos = env.sim.data.get_body_xpos(target_name)
        dist = np.linalg.norm(obj_pos - target_pos)
        return dist < threshold
    except:
        return False


def check_gripper_open(
    env,
    threshold: float = -0.5
) -> bool:
    """
    Check if gripper is open.
    
    Args:
        env: The robosuite environment
        threshold: Threshold for open gripper (action < threshold)
        
    Returns:
        True if gripper is open
    """
    try:
        gripper_qpos = env.robots[0].gripper[env.robots[0].eef_name].current_action
        return gripper_qpos < threshold
    except:
        return False


def check_gripper_closed(
    env,
    threshold: float = 0.5
) -> bool:
    """
    Check if gripper is closed.
    
    Args:
        env: The robosuite environment
        threshold: Threshold for closed gripper (action > threshold)
        
    Returns:
        True if gripper is closed
    """
    try:
        gripper_qpos = env.robots[0].gripper[env.robots[0].eef_name].current_action
        return gripper_qpos > threshold
    except:
        return False

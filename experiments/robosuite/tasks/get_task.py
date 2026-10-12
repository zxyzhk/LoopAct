"""
Task registry — single place to look up (config, success) by short name.

Usage:
    from tasks.get_task import get_task
    config, success = get_task("random_block_2d")
"""

import importlib


# maps task key -> (module_path, ConfigClass, SuccessClass, env_module)
# env_module is imported for its register_env() side-effect
TASK_REGISTRY = {
    # Ladle tasks
    "pick_ladle": (
        "tasks.pick_ladle",
        "PickLadleConfig", "PickLadleSuccess",
        "environment.TableLadleEnv",
    ),
    "pick_ladle_move": (
        "tasks.pick_ladle_move",
        "PickLadleMoveConfig", "PickLadleMoveSuccess",
        "environment.TableLadleEnv",
    ),

    # Block tasks (1D movement)
    "random_block": (
        "tasks.random_block",
        "RandomBlockConfig", "RandomBlockSuccess",
        "environment.RedBlockEnv",
    ),
    "fast_random_block": (
        "tasks.random_block",
        "FastRandomBlockConfig", "RandomBlockSuccess",
        "environment.RedBlockEnv",
    ),
    "steady_block": (
        "tasks.random_block",
        "SteadyBlockConfig", "RandomBlockSuccess",
        "environment.RedBlockEnv",
    ),

    # Block tasks (2D movement)
    "random_block_2d": (
        "tasks.random_block_2d",
        "RandomBlock2DConfig", "RandomBlockSuccess",
        "environment.GreenBlockEnv",
    ),

    # ToyCan tasks
    "toy_can": (
        "tasks.toy_can",
        "ToyCanConfig", "ToyCanSuccess",
        "environment.ToyCanEnv",
    ),
    "slope": (
        "tasks.slope",
        "SlopeConfig", "SlopeSuccess",
        "environment.SlopeEnv",
    ),

    # Toy Cart tasks
    "toy_cart": (
        "tasks.toy_cart",
        "ToyCartConfig", "ToyCartSuccess",
        "environment.ToyCartEnv",
    ),

    # Ablation tasks
    "catch_block": (
        "tasks.high_block",
        "HighBlockConfig", "HighBlockSuccess",
        "environment.HighBlockEnv",
    ),
    # Toy Robot tasks
    "toy_robot": (
        "tasks.toy_robot",
        "ToyRobotConfig", "ToyRobotSuccess",
        "environment.ToyRobotEnv",
    ),
    "toy_red": (
        "tasks.toy_red",
        "ToyRedConfig", "ToyRedSuccess",
        "environment.ToyRedEnv",
    ),
    "toy_red_speed": (
        "tasks.toy_red_speed",
        "ToyRedSpeedConfig", "ToyRedSpeedSuccess",
        "environment.ToyRedEnv",
    ),
    "toy_green": (
        "tasks.toy_green",
        "ToyGreenConfig", "ToyGreenSuccess",
        "environment.ToyGreenEnv",
    ),

    # Swing pendulum (physics-driven, no mover)
    "swing": (
        "tasks.swing",
        "SwingConfig", "SwingSuccess",
        "environment.SwingEnv",
    ),

    # Drop ball — toy_red geometry with only a ball and a (moving) cart+cup
    "drop_ball": (
        "tasks.drop_ball",
        "DropBallConfig", "DropBallSuccess",
        "environment.DropBallEnv",
    ),
    "drop_ball_speed": (
        "tasks.drop_ball_speed",
        "DropBallSpeedConfig", "DropBallSpeedSuccess",
        "environment.DropBallEnv",
    ),

    # Drop ball color — same as drop_ball but with a round orange-red cup
    "drop_ball_color": (
        "tasks.drop_ball_color",
        "DropBallColorConfig", "DropBallColorSuccess",
        "environment.DropBall2Env",
    ),

    # Catch carrot — toy_can geometry, tapered carrot rolls in arc down slope
    "catch_carrot": (
        "tasks.catch_carrot",
        "CatchCarrotConfig", "CatchCarrotSuccess",
        "environment.ConeEnv",
    ),

    # Block cart — blue block on moving cart (same physics footprint as toy_red)
    "block_cart": (
        "tasks.block_cart",
        "BlockCartConfig", "BlockCartSuccess",
        "environment.BlockCartEnv",
    ),
}


def get_task(task_key: str):
    """
    Get task config and success checker by key.

    Args:
        task_key: Short name from TASK_REGISTRY (e.g. "random_block_2d")

    Returns:
        (config, success): Instantiated TaskConfig and TaskSuccess objects

    Also imports the environment module to trigger register_env().
    """
    if task_key not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(f"Unknown task '{task_key}'. Available: {available}")

    module_path, config_cls_name, success_cls_name, env_module = TASK_REGISTRY[task_key]

    # Import env module for register_env() side-effect
    importlib.import_module(env_module)

    # Import task module and get classes
    task_module = importlib.import_module(module_path)
    config_cls = getattr(task_module, config_cls_name)
    success_cls = getattr(task_module, success_cls_name)

    return config_cls(), success_cls()


def list_tasks():
    """Print all available tasks."""
    print("Available tasks:")
    for key in sorted(TASK_REGISTRY.keys()):
        _, config_cls_name, _, _ = TASK_REGISTRY[key]
        print(f"  {key:25s} -> {config_cls_name}")

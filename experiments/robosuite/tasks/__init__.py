"""
Task definitions for robosuite environments.

Each task is defined in its own file with:
- TaskConfig: Configuration for environment setup
- TaskSuccess: Success criteria definition
"""

from .base import TaskConfig, TaskSuccess
from .pick_ladle import PickLadleConfig, PickLadleSuccess
from .random_block import RandomBlockConfig, RandomBlockSuccess, FastRandomBlockConfig, SteadyBlockConfig
from .toy_can import ToyCanConfig, ToyCanSuccess
from .slope import SlopeConfig, SlopeSuccess
from .toy_cart import ToyCartConfig, ToyCartSuccess
from .toy_robot import ToyRobotConfig, ToyRobotSuccess
from .catch_carrot import CatchCarrotConfig, CatchCarrotSuccess

__all__ = [
    'TaskConfig',
    'TaskSuccess',
    'PickLadleConfig',
    'PickLadleSuccess',
    'RandomBlockConfig',
    'RandomBlockSuccess',
    'FastRandomBlockConfig',
    'SteadyBlockConfig',
    'ToyCanConfig',
    'ToyCanSuccess',
    'SlopeConfig',
    'SlopeSuccess',
    'ToyCartConfig',
    'ToyCartSuccess',
    'ToyRobotConfig',
    'ToyRobotSuccess',
    'CatchCarrotConfig',
    'CatchCarrotSuccess',
]

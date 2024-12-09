"""
This file is used to define the functions that process the states to compute individual observation terms.
"""

from typing import Callable, Dict, Any
import numpy as np
import torch
import time


from utils.helpers import reorder_robot_states
from utils.math import quat_rotate_inverse, GRAVITY_DIR


def joint_pos(states: Dict[str, Any]) -> np.ndarray:
    """
    The joint positions of the asset.

    :param states: State dictionary
    :return: Joint positions
    """
    joint_pos = reorder_robot_states(
        states["joint_pos"],
        origin_order=["FL", "FR", "RL", "RR"],
        target_order=["FR", "FL", "RR", "RL"],
    )
    return joint_pos


def joint_pos_rel(states: Dict[str, Any], default_joint_pos: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """
    Compute relative joint positions.

    :param states: State dictionary
    :param default_joint_pos: Default joint positions
    :param mapping: Mapping from Unitree to Isaac Joint Order
    :return: Relative joint positions
    """
    return states["joint_pos"][mapping] - default_joint_pos


def joint_vel(states: Dict[str, Any], mapping: np.ndarray) -> np.ndarray:
    """
    The joint positions of the asset.

    :param states: State dictionary
    :param mapping: Mapping from Unitree to Isaac Joint Order
    :return: Joint velocities
    """
    return states["joint_vel"][mapping]


def lin_vel_b(states: Dict[str, Any]) -> np.ndarray:
    """
    The linear velocity of the asset in base frame.

    :param states: State dictionary
    :return: Linear velocity in the base frame
    """

    return states["lin_vel_b"]


def ang_vel_b(states: Dict[str, Any]) -> np.ndarray:
    """
    The angular velocity of the asset in base frame.

    :param states: State dictionary
    :return: Angular velocity in the base frame
    """

    return states["gyroscope"]


# TODO: Convert to pure NumPy function
def projected_gravity_b(states: Dict[str, Any]) -> torch.Tensor:
    """
    The projected gravity vector.

    :param states: State dictionary
    :return: Angular velocity in the base frame
    """

    return quat_rotate_inverse(torch.tensor([states["base_quat"]]).squeeze(0), GRAVITY_DIR)


def last_action(states: Dict[str, Any], last_action: Callable) -> np.ndarray:
    """
    The previous action from the policy. We use a callable (lambda) to fetch the latest value from the controller class.

    :param states: State dictionary
    :return: The previous action from the agent
    """
    return last_action()


def velocity_commands(states: Dict[str, Any], velocity_commands: Callable) -> np.ndarray:
    """
    The velocity commands. We use a callable (lambda) to fetch the latest value from the controller class.

    :param states: State dictionary
    :return: Velocity commands (Vx, Vy, Wz)
    """
    return velocity_commands()


def feet_pos(states: Dict[str, Any], pin_model_wrapper) -> np.ndarray:
    """
    The feet positions of the robot. Calculated from the pinocchio wrapper

    :param states: State dictionary
    :return: Feet pos (4, 3)
    """
    feet_pos_ = pin_model_wrapper.get_foot_pos_base()

    return feet_pos_


def starting_time(states: Dict[str, Any]):
    return time.time()

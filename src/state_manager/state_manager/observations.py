"""
This file is used to define the functions that process the states to compute individual observation terms.
"""

import time
from typing import TYPE_CHECKING, Any, Callable, Dict

import numpy as np
import torch
from utils.helpers import reorder_robot_states
from utils.math import GRAVITY_DIR, quat_rotate_inverse

if TYPE_CHECKING:
    from utils.mj_wrapper.mj_robot import MjQuadRobotWrapper


def joint_pos(states: Dict[str, Any]) -> torch.Tensor:
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
    return torch.tensor(joint_pos)


def joint_pos_rel(states: Dict[str, Any], default_joint_pos: np.ndarray, mapping: np.ndarray) -> torch.Tensor:
    """
    Compute relative joint positions.

    :param states: State dictionary
    :param default_joint_pos: Default joint positions
    :param mapping: Mapping from Unitree to Isaac Joint Order
    :return: Relative joint positions
    """
    return torch.tensor(states["joint_pos"][mapping] - default_joint_pos)


def joint_vel(states: Dict[str, Any], mapping: np.ndarray) -> torch.Tensor:
    """
    The joint positions of the asset.

    :param states: State dictionary
    :param mapping: Mapping from Unitree to Isaac Joint Order
    :return: Joint velocities
    """
    return torch.tensor(states["joint_vel"][mapping])


def lin_vel_b(states: Dict[str, Any]) -> torch.Tensor:
    """
    The linear velocity of the asset in base frame.

    :param states: State dictionary
    :return: Linear velocity in the base frame
    """

    return torch.tensor(states["lin_vel_b"])


def ang_vel_b(states: Dict[str, Any]) -> torch.Tensor:
    """
    The angular velocity of the asset in base frame.

    :param states: State dictionary
    :return: Angular velocity in the base frame
    """

    return torch.tensor(states["gyroscope"])


def projected_gravity_b(states: Dict[str, Any]) -> torch.Tensor:
    """
    The projected gravity vector.

    :param states: State dictionary
    :return: Projected Gravity vector in the base frame
    """

    quat = torch.tensor([states["base_quat"]], dtype=torch.float64).squeeze(0)
    gravity_dir = torch.tensor([0, 0, -1.0], dtype=torch.float64)
    
    return quat_rotate_inverse(quat, gravity_dir)


def last_action(states: Dict[str, Any], last_action: Callable) -> torch.Tensor:
    """
    The previous action from the policy. We use a callable (lambda) to fetch the latest value from the controller class.

    :param states: State dictionary
    :return: The previous action from the agent
    """
    return last_action()


def velocity_commands(states: Dict[str, Any], velocity_commands: Callable) -> torch.Tensor:
    """
    The velocity commands. We use a callable (lambda) to fetch the latest value from the controller class.

    :param states: State dictionary
    :return: Velocity commands (Vx, Vy, Wz)
    """
    return velocity_commands()

def global_velocity_commands(states: Dict[str, Any], velocity_commands: Callable) -> torch.Tensor:
    """
    The velocity commands. We use a callable (lambda) to fetch the latest value from the controller class.

    :param states: State dictionary
    :return: Velocity commands (Vx, Vy, Wz)
    """
    
    dtype = torch.float64
    
    cmd_b = torch.tensor(velocity_commands()[:3], dtype=dtype)
    quat = torch.tensor(states["base_quat"], dtype=dtype)
    rotated = quat_rotate_inverse(quat, cmd_b)
    wz = torch.tensor([velocity_commands()[3]], dtype=dtype)
    cmd_w = torch.cat((rotated, wz))
    return cmd_w


def relative_distance_to_box(states: Dict[str, Any]) -> torch.Tensor:
    """
    Distance between robot base and start of the box.
    """
    
    
    return torch.tensor(states["base_pos_w"][0] - states["object_pos_w"][0])


def box_parameters(states: Dict[str, Any]) -> torch.Tensor:
    """
    This is just the height of the box.
    """
    
    return torch.tensor([0.3])

def starting_time(states: Dict[str, Any]):
    return time.time()


# Contact Explicit Additional Observations


def contact_plan(states: Dict[str, Any], contact_plan: Callable) -> torch.Tensor:
    """
    The contact plan. We use a callable (lambda) to fetch the latest value from the controller class.

    :param states: State dictionary
    :return: Contact plan
    """
    return contact_plan().view(-1)


def contact_status(states: Dict[str, Any]) -> torch.Tensor:
    """
    The contact status. We use a callable (lambda) to fetch the latest value from the controller class.
    """
    contact_status = np.zeros(4)
    contact_forces = np.array(states["foot_forces"])
    contact_forces_norm = np.linalg.norm(contact_forces)
    contact_status[contact_forces_norm > 1.0] = 1
    return torch.tensor(contact_status)


def contact_time_left(states: Dict[str, Any], contact_time_left: Callable) -> torch.Tensor:
    """
    The contact timing. We use a callable (lambda) to fetch the latest value from the controller class.

    :param states: State dictionary
    :return: Contact timing
    """
    return torch.tensor([contact_time_left()])


def base_height(states: Dict[str, Any], mj_model_wrapper: "MjQuadRobotWrapper") -> torch.Tensor:
    """
    The height of the base of the robot.
    """
    return torch.tensor([mj_model_wrapper.get_base_height_init_frame()])


def ee_pos_rel_b(
    states: Dict[str, Any],
    mj_model_wrapper: "MjQuadRobotWrapper",
    future_feet_positions_w: Callable,
    current_goal_idx: Callable,
) -> torch.Tensor:
    """
    Compute the end-effector positions relative to the desired contact locations.

    Args:
        states: Dictionary of states
        mj_model_wrapper: MuJoCo model wrapper
        future_feet_positions_w: Function to get future feet positions
        current_goal_idx: Function to get current goal index

    Returns:
        End-effector positions relative to base frame
    """
    # Get current feet positions in world frame
    feet_positions_w = mj_model_wrapper.get_feet_positions_world()
    # Get future feet positions in init frame
    desired_feet_positions = future_feet_positions_w()[:, current_goal_idx()]
    # Compute the distance between the current feet positions and the desired feet positions
    ee_pos_rel = torch.tensor(desired_feet_positions - feet_positions_w).norm(dim=1)

    return ee_pos_rel


def contact_locations(
    states: Dict[str, Any],
    mj_model_wrapper: "MjQuadRobotWrapper",
    future_feet_positions_init_frame: Callable,
    current_goal_idx: Callable,
    obs_horizon: int,
) -> torch.Tensor:
    """
    The future desired feet positions in the base frame.
    """
    return torch.tensor(
        mj_model_wrapper.transform_init_to_base(
            future_feet_positions_init_frame()[:, current_goal_idx() : current_goal_idx() + obs_horizon]
        ).flatten()
    )


def contact_locations_b(
    states: Dict[str, Any],
    mj_model_wrapper: "MjQuadRobotWrapper",
    future_feet_positions_w: Callable,
    current_goal_idx: Callable,
    obs_horizon: int,
) -> torch.Tensor:
    """
    The future desired feet positions in the base frame.
    """
    current_base_index = current_goal_idx() // 2
    return torch.tensor(
        mj_model_wrapper.transform_world_to_base(
            future_feet_positions_w()[:, current_goal_idx() : current_goal_idx() + obs_horizon].numpy()
        ).flatten()
    )

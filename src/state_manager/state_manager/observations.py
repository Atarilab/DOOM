"""
This file is used to define the functions that process the states to compute individual observation terms.
"""

import time
from typing import TYPE_CHECKING, Any, Callable, Dict

import numpy as np
import torch

from utils.helpers import reorder_robot_states
from utils.math import quat_rotate_inverse

import logging

if TYPE_CHECKING:
    pass


def joint_pos(states: Dict[str, Any], asset_name: str = "robot") -> torch.Tensor:
    """
    The joint positions of the asset.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :return: Joint positions
    """
    joint_pos = reorder_robot_states(
        states[f"{asset_name}/joint_pos"],
        origin_order=["FL", "FR", "RL", "RR"],
        target_order=["FR", "FL", "RR", "RL"],
    )
    return torch.tensor(joint_pos)


def joint_pos_rel(
    states: Dict[str, Any],
    default_joint_pos: np.ndarray,
    asset_name: str = "robot",
    scale=1.0,
    mapping: np.ndarray = None,
) -> torch.Tensor:
    """
    Compute relative joint positions.

    :param states: State dictionary
    :param default_joint_pos: Default joint positions
        :param scale: Scale factor
    :param mapping: Mapping from Unitree to Isaac Joint Order
    :return: Relative joint positions
    """
    if mapping is None:
        mapping = np.arange(len(states[f"{asset_name}/joint_pos"]))
    return torch.tensor((states[f"{asset_name}/joint_pos"][mapping] - default_joint_pos), dtype=torch.float32) * scale


def joint_pos_limit_normalized(
    states: Dict[str, Any], soft_dof_limits: np.ndarray, asset_name: str = "robot"
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's soft joint limits.

    :param states: State dictionary
    :param soft_dof_limits: Soft joint limits
    :param asset_name: Name of the asset
    :return: Normalized joint positions
    """

    joint_pos = states[f"{asset_name}/joint_pos"]
    lower_limit = soft_dof_limits[:, 0]
    upper_limit = soft_dof_limits[:, 1]

    offset = (lower_limit + upper_limit) * 0.5

    return 2 * (joint_pos - offset) / (upper_limit - lower_limit)


def joint_vel(states: Dict[str, Any], asset_name: str = "robot", scale=1.0, mapping: np.ndarray = None) -> torch.Tensor:
    """
    The joint velocities of the asset.

    :param states: State dictionary
    :param scale: Scale factor
    :param mapping: Mapping from Unitree to Isaac Joint Order
    :return: Joint velocities
    """
    if mapping is None:
        mapping = np.arange(len(states[f"{asset_name}/joint_vel"]))
    return torch.tensor((states[f"{asset_name}/joint_vel"][mapping]), dtype=torch.float32) * scale


def lin_vel_w(states: Dict[str, Any], asset_name: str = "robot", scale=1.0) -> torch.Tensor:
    """
    The linear velocity of the asset in world frame.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param scale: Scale factor
    :return: Linear velocity in the base frame
    """

    return torch.tensor((states[f"{asset_name}/lin_vel_w"])) * scale


def ang_vel_w(states: Dict[str, Any], asset_name: str = "robot", scale=1.0) -> torch.Tensor:
    """
    The angular velocity of the asset in world frame.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param scale: Scale factor
    :return: Angular velocity in the base frame
    """

    return torch.tensor((states[f"{asset_name}/ang_vel_w"])) * scale


def lin_vel_b(states: Dict[str, Any], asset_name: str = "robot", scale=1.0) -> torch.Tensor:
    """
    The linear velocity of the asset in base frame.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param scale: Scale factor
    :return: Linear velocity in the base frame
    """

    return torch.tensor((states[f"{asset_name}/lin_vel_b"])) * scale


def ang_vel_b(states: Dict[str, Any], asset_name: str = "robot", scale=1.0) -> torch.Tensor:
    """
    The angular velocity of the asset in base frame.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param scale: Scale factor
    :return: Angular velocity in the base frame
    """

    return torch.tensor((states[f"{asset_name}/gyroscope"])) * scale


def root_pos_w(states: Dict[str, Any], asset_name: str = "robot", scale=1.0) -> torch.Tensor:
    """
    The position of the root of the asset in world frame.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param scale: Scale factor
    :return: Position of the root in the world frame
    """
    return torch.tensor((states[f"{asset_name}/base_pos_w"])) * scale


def root_quat_w(states: Dict[str, Any], asset_name: str = "robot", scale=1.0) -> torch.Tensor:
    """
    The orientation of the root of the asset in world frame.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param scale: Scale factor
    :return: Orientation of the root in the world frame
    """
    return torch.tensor((states[f"{asset_name}/base_quat"])) * scale


def unitree_gravity_orientation(states: Dict[str, Any], asset_name: str = "robot", scale=1.0) -> torch.Tensor:
    """
    The orientation of the gravity vector in the unitree frame.
    """
    base_quat = states[f"{asset_name}/base_quat"]
    if isinstance(base_quat, np.ndarray):
        quat = torch.tensor(base_quat, dtype=torch.float32)
    elif torch.is_tensor(base_quat):
        quat = base_quat.clone().detach().to(dtype=torch.float32)
    else:
        quat = torch.tensor(np.array(base_quat), dtype=torch.float32)
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return torch.tensor(gravity_orientation, dtype=torch.float32) * scale


def projected_gravity_b(states: Dict[str, Any], asset_name: str = "robot", scale=1.0) -> torch.Tensor:
    """
    The projected gravity vector.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param scale: Scale factor
    :return: Projected Gravity vector in the base frame
    """

    base_quat = states[f"{asset_name}/base_quat"]
    if isinstance(base_quat, np.ndarray):
        quat = torch.tensor(base_quat, dtype=torch.float32)
    elif torch.is_tensor(base_quat):
        quat = base_quat.clone().detach().to(dtype=torch.float32)
    else:
        quat = torch.tensor(np.array(base_quat), dtype=torch.float32)
    gravity_dir = torch.tensor([0, 0, -1.0], dtype=torch.float32)

    return quat_rotate_inverse(quat, gravity_dir) * scale


def last_action(states: Dict[str, Any], last_action: Callable, scale=1.0) -> torch.Tensor:
    """
    The previous action from the policy. We use a callable (lambda) to fetch the latest value from the controller class.

    :param states: State dictionary
    :param scale: Scale factor
    :return: The previous action from the agent
    """
    return last_action() * scale


def velocity_commands(states: Dict[str, Any], velocity_commands: Callable, scale=1.0) -> torch.Tensor:
    """
    The velocity commands. We use a callable (lambda) to fetch the latest value from the controller class.

    :param states: State dictionary
    :param scale: Scale factor
    :return: Velocity commands (Vx, Vy, Wz)
    """
    return velocity_commands() * scale


def phase(states: Dict[str, Any], counter: Callable, period: float, control_dt: float):
    count = counter() * control_dt
    phase = count % period / period
    sin_phase = np.sin(phase * np.pi * 2)
    cos_phase = np.cos(phase * np.pi * 2)
    return torch.tensor([sin_phase, cos_phase])


def phase_with_timing(states: Dict[str, Any], logger: Callable, period: float, control_dt: float, decimation: int):
    """
    Phase observation that increments an internal counter only when elapsed time exceeds control_dt * decimation.
    The counter is stored as a function attribute and persists across calls.
    
    :param states: State dictionary
    :param period: Period of the phase signal
    :param control_dt: Control timestep
    :param decimation: Decimation factor for counter increment
    :return: Phase observation as [sin_phase, cos_phase]
    """
    if not hasattr(phase_with_timing, "counter"):
        phase_with_timing.counter = 0
        phase_with_timing.start_time = time.time()
    
    # Calculate actual elapsed time
    current_time = time.time()
    elapsed_time = current_time - phase_with_timing.start_time
    
    # Increment counter every control_dt * decimation seconds
    if elapsed_time >= control_dt:
        phase_with_timing.counter += 1
        phase_with_timing.start_time = time.time()
        
    
    logger().debug(f"Phase counter incremented to {phase_with_timing.counter}")
    # Use counter for phase calculation
    count = phase_with_timing.counter * control_dt
    phase = count % period / period
    sin_phase = np.sin(phase * np.pi * 2)
    cos_phase = np.cos(phase * np.pi * 2)
    return torch.tensor([sin_phase, cos_phase], dtype=torch.float32)


def current_time(states: Dict[str, Any]):
    return time.time()


# Contact Explicit Additional Observations


def contact_plan(states: Dict[str, Any], contact_plan: Callable) -> torch.Tensor:
    """
    The contact plan. We use a callable (lambda) to fetch the latest value from the controller class.

    :param states: State dictionary
    :return: Contact plan
    """
    return contact_plan().view(-1)


def contact_status(states: Dict[str, Any], asset_name: str = "robot") -> torch.Tensor:
    """
    The contact status. We use a callable (lambda) to fetch the latest value from the controller class.
    """
    contact_status = np.zeros(4)
    contact_forces = np.array(states[f"{asset_name}/foot_forces"])
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


def object_size(states: Dict[str, Any], size: tuple[float, float, float], asset_name: str = "object") -> torch.Tensor:
    """
    The size of the object.

    :param states: State dictionary
    :param size: Size of the object
    :param asset_name: Name of the asset
    :return: Size of the object
    """
    return torch.tensor(size)


def dummy_contact_status(states: Dict[str, Any]) -> torch.Tensor:
    """
    Dummy contact status.
    """
    return torch.zeros(4)


def base_height(states: Dict[str, Any], mj_model: "MjRobotWrapper") -> torch.Tensor:
    """
    The height of the base of the robot.
    """
    return torch.tensor([mj_model.get_base_height_init_frame()])


def ee_pos_rel_b(
    states: Dict[str, Any],
    mj_model: "MjRobotWrapper",
    future_feet_positions_w: Callable,
    current_goal_idx: Callable,
) -> torch.Tensor:
    """
    Compute the end-effector positions relative to the desired contact locations.

    Args:
        states: Dictionary of states
        mj_model: MuJoCo model wrapper
        future_feet_positions_w: Function to get future feet positions
        current_goal_idx: Function to get current goal index

    Returns:
        End-effector positions relative to base frame
    """
    # Get current feet positions in world frame
    feet_positions_w = mj_model.get_feet_positions_world()
    # Get future feet positions in init frame
    desired_feet_positions = future_feet_positions_w()[:, current_goal_idx()]
    # Compute the distance between the current feet positions and the desired feet positions
    diff = desired_feet_positions - feet_positions_w
    if torch.is_tensor(diff):
        return diff.clone().norm(dim=1)
    return torch.tensor(diff).norm(dim=1)


def contact_locations(
    states: Dict[str, Any],
    mj_model: "MjRobotWrapper",
    future_feet_positions_init_frame: Callable,
    current_goal_idx: Callable,
    obs_horizon: int,
) -> torch.Tensor:
    """
    The future desired feet positions in the base frame.
    """
    return torch.tensor(
        mj_model.transform_init_to_base(
            future_feet_positions_init_frame()[:, current_goal_idx(): current_goal_idx() + obs_horizon]
        ).flatten()
    )


def contact_locations_b(
    states: Dict[str, Any],
    mj_model: "MjRobotWrapper",
    future_feet_positions_w: Callable,
    current_goal_idx: Callable,
    obs_horizon: int,
) -> torch.Tensor:
    """
    The future desired feet positions in the base frame.
    """
    return torch.tensor(
        mj_model.transform_world_to_base(
            future_feet_positions_w()[:, current_goal_idx(): current_goal_idx() + obs_horizon].numpy()
        ).flatten()
    )

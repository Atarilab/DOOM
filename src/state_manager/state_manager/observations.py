"""
This file is used to define the functions that process the states to compute individual observation terms.
"""

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np
import torch

from utils.helpers import reorder_robot_states, tensorify
from utils.math import quat_rotate_inverse

import logging

if TYPE_CHECKING:
    pass  # MjRobotWrapper will be imported when needed


def joint_pos(states: Dict[str, Any], asset_name: str = "robot", dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The joint positions of the asset.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Joint positions
    """
    joint_pos = reorder_robot_states(
        states[f"{asset_name}/joint_pos"],
        origin_order=["FL", "FR", "RL", "RR"],
        target_order=["FR", "FL", "RR", "RL"],
    )
    return tensorify(joint_pos, dtype=dtype, device=device)


def joint_pos_rel(
    states: Dict[str, Any],
    default_joint_pos: np.ndarray,
    asset_name: str = "robot",
    scale=1.0,
    mapping: Optional[np.ndarray] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute relative joint positions.

    :param states: State dictionary
    :param default_joint_pos: Default joint positions
    :param scale: Scale factor
    :param mapping: Mapping from Unitree to Isaac Joint Order
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Relative joint positions
    """
    if mapping is None:
        mapping = np.arange(len(states[f"{asset_name}/joint_pos"]))
    
    # Get joint positions and ensure they're numpy arrays for consistent subtraction
    joint_pos_data = states[f"{asset_name}/joint_pos"][mapping]
    if isinstance(joint_pos_data, torch.Tensor):
        joint_pos_data = joint_pos_data.cpu().numpy()
    
    # Ensure default_joint_pos is also a numpy array
    if isinstance(default_joint_pos, torch.Tensor):
        default_joint_pos = default_joint_pos.cpu().numpy()
    
    result = joint_pos_data - default_joint_pos
    return tensorify(result, dtype=dtype, device=device) * scale


def joint_pos_limit_normalized(
    states: Dict[str, Any], soft_dof_limits: np.ndarray, asset_name: str = "robot", dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's soft joint limits.

    :param states: State dictionary
    :param soft_dof_limits: Soft joint limits
    :param asset_name: Name of the asset
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Normalized joint positions
    """

    joint_pos = states[f"{asset_name}/joint_pos"]
    
    # Ensure joint_pos is a numpy array for consistent arithmetic operations
    if isinstance(joint_pos, torch.Tensor):
        joint_pos = joint_pos.cpu().numpy()
    
    # Ensure soft_dof_limits is a numpy array
    if isinstance(soft_dof_limits, torch.Tensor):
        soft_dof_limits = soft_dof_limits.cpu().numpy()
    
    lower_limit = soft_dof_limits[:, 0]
    upper_limit = soft_dof_limits[:, 1]

    offset = (lower_limit + upper_limit) * 0.5
    result = 2 * (joint_pos - offset) / (upper_limit - lower_limit)
    return tensorify(result, dtype=dtype, device=device)


def joint_vel(states: Dict[str, Any], asset_name: str = "robot", scale=1.0, mapping: Optional[np.ndarray] = None, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The joint velocities of the asset.

    :param states: State dictionary
    :param scale: Scale factor
    :param mapping: Mapping from Unitree to Isaac Joint Order
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Joint velocities
    """
    if mapping is None:
        mapping = np.arange(len(states[f"{asset_name}/joint_vel"]))
    result = states[f"{asset_name}/joint_vel"][mapping]
    return tensorify(result, dtype=dtype, device=device) * scale


def lin_vel_w(states: Dict[str, Any], asset_name: str = "robot", scale=1.0, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The linear velocity of the asset in world frame.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param scale: Scale factor
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Linear velocity in the base frame
    """
    result = states[f"{asset_name}/lin_vel_w"]
    return tensorify(result, dtype=dtype, device=device) * scale


def ang_vel_w(states: Dict[str, Any], asset_name: str = "robot", scale=1.0, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The angular velocity of the asset in world frame.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param scale: Scale factor
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Angular velocity in the base frame
    """
    result = states[f"{asset_name}/ang_vel_w"]
    return tensorify(result, dtype=dtype, device=device) * scale


def lin_vel_b(states: Dict[str, Any], asset_name: str = "robot", scale=1.0, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The linear velocity of the asset in base frame.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param scale: Scale factor
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Linear velocity in the base frame
    """
    result = states[f"{asset_name}/lin_vel_b"]
    return tensorify(result, dtype=dtype, device=device) * scale


def ang_vel_b(states: Dict[str, Any], asset_name: str = "robot", scale=1.0, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The angular velocity of the asset in base frame.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param scale: Scale factor
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Angular velocity in the base frame
    """
    result = states[f"{asset_name}/gyroscope"]
    return tensorify(result, dtype=dtype, device=device) * scale


def root_pos_w(states: Dict[str, Any], asset_name: str = "robot", scale=1.0, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The position of the root of the asset in world frame.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param scale: Scale factor
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Position of the root in the world frame
    """
    result = states[f"{asset_name}/base_pos_w"]
    return tensorify(result, dtype=dtype, device=device) * scale


def root_quat_w(states: Dict[str, Any], asset_name: str = "robot", scale=1.0, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The orientation of the root of the asset in world frame.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param scale: Scale factor
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Orientation of the root in the world frame
    """
    result = states[f"{asset_name}/base_quat"]
    return tensorify(result, dtype=dtype, device=device) * scale


def unitree_gravity_orientation(states: Dict[str, Any], asset_name: str = "robot", scale=1.0, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The orientation of the gravity vector in the unitree frame.
    """
    base_quat = states[f"{asset_name}/base_quat"]
    quat = tensorify(base_quat, dtype=dtype, device=device)
    
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    gravity_orientation = torch.zeros(3, dtype=dtype, device=device)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    
    return gravity_orientation * scale


def projected_gravity_b(states: Dict[str, Any], asset_name: str = "robot", scale=1.0, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The projected gravity vector.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param scale: Scale factor
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Projected Gravity vector in the base frame
    """
    base_quat = states[f"{asset_name}/base_quat"]
    quat = tensorify(base_quat, dtype=dtype, device=device)
    gravity_dir = torch.tensor([0, 0, -1.0], dtype=dtype, device=device)

    return quat_rotate_inverse(quat, gravity_dir) * scale


def last_action(states: Dict[str, Any], last_action: Callable, scale=1.0, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The previous action from the policy. We use a callable (lambda) to fetch the latest value from the controller class.

    :param states: State dictionary
    :param scale: Scale factor
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: The previous action from the agent
    """
    result = last_action()
    return tensorify(result, dtype=dtype, device=device) * scale


def velocity_commands(states: Dict[str, Any], velocity_commands: Callable, scale=1.0, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The velocity commands. We use a callable (lambda) to fetch the latest value from the controller class.

    :param states: State dictionary
    :param scale: Scale factor
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Velocity commands (Vx, Vy, Wz)
    """
    result = velocity_commands()
    return tensorify(result, dtype=dtype, device=device) * scale


def phase(states: Dict[str, Any], counter: Callable, period: float, control_dt: float, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Compute phase observation.
    
    :param states: State dictionary
    :param counter: Function to get current counter value
    :param period: Period of the phase signal
    :param control_dt: Control timestep
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Phase observation as [sin_phase, cos_phase]
    """
    count = counter() * control_dt
    phase_val = count % period / period
    sin_phase = np.sin(phase_val * np.pi * 2)
    cos_phase = np.cos(phase_val * np.pi * 2)
    return tensorify([sin_phase, cos_phase], dtype=dtype, device=device)


def phase_with_timing(states: Dict[str, Any], counter: Callable, period: float, control_dt: float, decimation: int, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Phase observation that increments an internal counter only when elapsed time exceeds control_dt * decimation.
    The counter is stored as a function attribute and persists across calls.
    
    :param states: State dictionary
    :param period: Period of the phase signal
    :param control_dt: Control timestep
    :param decimation: Decimation factor for counter increment
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Phase observation as [sin_phase, cos_phase]
    """
    
    
    # Use counter for phase calculation
    count = counter() * control_dt
    phase_val = count % period / period
    sin_phase = np.sin(phase_val * np.pi * 2)
    cos_phase = np.cos(phase_val * np.pi * 2)
    return tensorify([sin_phase, cos_phase], dtype=dtype, device=device)


def current_time(states: Dict[str, Any], dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Get current time as a tensor.
    
    :param states: State dictionary
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Current time as tensor
    """
    return time.time()


# Contact Explicit Additional Observations


def contact_plan(states: Dict[str, Any], contact_plan: Callable, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The contact plan. We use a callable (lambda) to fetch the latest value from the controller class.

    :param states: State dictionary
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Contact plan
    """
    result = contact_plan().view(-1)
    return tensorify(result, dtype=dtype, device=device)


def contact_status(states: Dict[str, Any], asset_name: str = "robot", dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The contact status based on foot forces.
    
    :param states: State dictionary
    :param asset_name: Name of the asset
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Contact status tensor
    """
    contact_status = np.zeros(4)
    contact_forces = states[f"{asset_name}/foot_forces"]
    
    # Ensure contact_forces is a numpy array for consistent operations
    if isinstance(contact_forces, torch.Tensor):
        contact_forces = contact_forces.cpu().numpy()
    elif not isinstance(contact_forces, np.ndarray):
        contact_forces = np.array(contact_forces)
    
    contact_forces_norm = np.linalg.norm(contact_forces)
    contact_status[contact_forces_norm > 1.0] = 1
    return tensorify(contact_status, dtype=dtype, device=device)


def contact_time_left(states: Dict[str, Any], contact_time_left: Callable, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The contact timing. We use a callable (lambda) to fetch the latest value from the controller class.

    :param states: State dictionary
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Contact timing
    """
    result = contact_time_left()
    return tensorify([result], dtype=dtype, device=device)


def object_size(states: Dict[str, Any], size: tuple[float, float, float], asset_name: str = "object", dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The size of the object.

    :param states: State dictionary
    :param size: Size of the object
    :param asset_name: Name of the asset
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Size of the object
    """
    return tensorify(size, dtype=dtype, device=device)


def dummy_contact_status(states: Dict[str, Any], dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Dummy contact status.
    
    :param states: State dictionary
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Zero tensor for contact status
    """
    return tensorify(np.zeros(4), dtype=dtype, device=device)


def base_height(states: Dict[str, Any], mj_model: Any, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    The height of the base of the robot.
    
    :param states: State dictionary
    :param mj_model: MuJoCo model wrapper
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Base height as tensor
    """
    result = mj_model.get_base_height_init_frame()
    return tensorify([result], dtype=dtype, device=device)


def ee_pos_rel_b(
    states: Dict[str, Any],
    mj_model: Any,
    future_feet_positions_w: Callable,
    current_goal_idx: Callable,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute the end-effector positions relative to the desired contact locations.

    Args:
        states: Dictionary of states
        mj_model: MuJoCo model wrapper
        future_feet_positions_w: Function to get future feet positions
        current_goal_idx: Function to get current goal index
        dtype: Desired tensor dtype
        device: Desired tensor device

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
        result = diff.clone().norm(dim=1)
    else:
        result = torch.tensor(diff, dtype=dtype, device=device).norm(dim=1)
    return tensorify(result, dtype=dtype, device=device)


def contact_locations(
    states: Dict[str, Any],
    mj_model: Any,
    future_feet_positions_init_frame: Callable,
    current_goal_idx: Callable,
    obs_horizon: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    The future desired feet positions in the base frame.
    
    :param states: State dictionary
    :param mj_model: MuJoCo model wrapper
    :param future_feet_positions_init_frame: Function to get future feet positions in init frame
    :param current_goal_idx: Function to get current goal index
    :param obs_horizon: Observation horizon
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Contact locations tensor
    """
    result = mj_model.transform_init_to_base(
        future_feet_positions_init_frame()[:, current_goal_idx(): current_goal_idx() + obs_horizon]
    ).flatten()
    return tensorify(result, dtype=dtype, device=device)


def contact_locations_b(
    states: Dict[str, Any],
    mj_model: Any,
    future_feet_positions_w: Callable,
    current_goal_idx: Callable,
    obs_horizon: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    The future desired feet positions in the base frame.
    
    :param states: State dictionary
    :param mj_model: MuJoCo model wrapper
    :param future_feet_positions_w: Function to get future feet positions in world frame
    :param current_goal_idx: Function to get current goal index
    :param obs_horizon: Observation horizon
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Contact locations in base frame tensor
    """
    result = mj_model.transform_world_to_base(
        future_feet_positions_w()[:, current_goal_idx(): current_goal_idx() + obs_horizon].numpy()
    ).flatten()
    return tensorify(result, dtype=dtype, device=device)

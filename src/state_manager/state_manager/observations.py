"""
This file is used to define the functions that process the states to compute individual observation terms.
"""

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from utils.helpers import reorder_robot_states, tensorify
from utils.math import (
    pos_diff,
    pose_diff,
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    pass  # MjRobotWrapper will be imported when needed


def joint_pos(
    states: Dict[str, Any],
    asset_name: str = "robot",
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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
    default_joint_pos: torch.Tensor,
    asset_name: str = "robot",
    scale=1.0,
    mapping: Optional[torch.Tensor] = None,
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
        mapping = torch.arange(len(states[f"{asset_name}/joint_pos"]), dtype=torch.long, device=device)

    # Get joint positions and ensure they're tensors for consistent subtraction
    joint_pos_data = states[f"{asset_name}/joint_pos"][mapping]
    default_joint_pos_tensor = tensorify(default_joint_pos, dtype=dtype, device=device)

    result = joint_pos_data - default_joint_pos_tensor
    return result * scale


def joint_pos_limit_normalized(
    states: Dict[str, Any],
    soft_dof_limits: torch.Tensor,
    mapping: Optional[torch.Tensor] = None,
    asset_name: str = "robot",
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's soft joint limits.

    :param states: State dictionary
    :param soft_dof_limits: Soft joint limits
    :param asset_name: Name of the asset
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Normalized joint positions
    """

    joint_pos = tensorify(states[f"{asset_name}/joint_pos"], dtype=dtype, device=device)[mapping]
    soft_dof_limits_tensor = tensorify(soft_dof_limits, dtype=dtype, device=device)

    lower_limit = soft_dof_limits_tensor[0][mapping]
    upper_limit = soft_dof_limits_tensor[1][mapping]

    offset = (lower_limit + upper_limit) * 0.5
    result = 2 * (joint_pos - offset) / (upper_limit - lower_limit)
    return result


def joint_vel(
    states: Dict[str, Any],
    asset_name: str = "robot",
    scale=1.0,
    mapping: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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
        mapping = torch.arange(len(states[f"{asset_name}/joint_vel"]), dtype=torch.long, device=device)
    result = states[f"{asset_name}/joint_vel"][mapping]
    return result * scale


def lin_vel_w(
    states: Dict[str, Any],
    asset_name: str = "robot",
    scale=1.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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


def ang_vel_w(
    states: Dict[str, Any],
    asset_name: str = "robot",
    scale=1.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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


def lin_vel_b(
    states: Dict[str, Any],
    asset_name: str = "robot",
    scale=1.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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


def ang_vel_b(
    states: Dict[str, Any],
    asset_name: str = "robot",
    scale=1.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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


def root_pos_w(
    states: Dict[str, Any],
    asset_name: str = "robot",
    scale=1.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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


def root_quat_w(
    states: Dict[str, Any],
    asset_name: str = "robot",
    scale=1.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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


def projected_gravity_b(
    states: Dict[str, Any],
    asset_name: str = "robot",
    scale=1.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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


def last_action(
    states: Dict[str, Any],
    last_action: Callable,
    scale=1.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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


def velocity_commands(
    states: Dict[str, Any],
    velocity_commands: Callable,
    scale=1.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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


def phase(
    states: Dict[str, Any],
    counter: Callable,
    period: float,
    control_dt: float,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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
    sin_phase = torch.sin(torch.tensor(phase_val * torch.pi * 2, dtype=dtype, device=device))
    cos_phase = torch.cos(torch.tensor(phase_val * torch.pi * 2, dtype=dtype, device=device))
    return torch.stack([sin_phase, cos_phase])


def phase_with_timing(
    states: Dict[str, Any],
    counter: Callable,
    period: float,
    control_dt: float,
    decimation: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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
    sin_phase = torch.sin(torch.tensor(phase_val * torch.pi * 2, dtype=dtype, device=device))
    cos_phase = torch.cos(torch.tensor(phase_val * torch.pi * 2, dtype=dtype, device=device))
    return torch.stack([sin_phase, cos_phase])


def current_time(
    states: Dict[str, Any], dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Get current time as a tensor.

    :param states: State dictionary
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Current time as tensor
    """
    return tensorify(time.time(), dtype=dtype, device=device)


# Contact Explicit Additional Observations


def contact_plan(
    states: Dict[str, Any],
    contact_plan: Callable,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    The contact plan. We use a callable (lambda) to fetch the latest value from the controller class.

    :param states: State dictionary
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Contact plan
    """
    return contact_plan().reshape(-1)


def contact_status(
    states: Dict[str, Any],
    asset_name: str = "robot",
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    The contact status based on foot forces.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Contact status tensor
    """
    contact_status = torch.zeros(4, dtype=dtype, device=device)
    contact_forces = tensorify(states[f"{asset_name}/foot_forces"], dtype=dtype, device=device)

    contact_forces_norm = torch.norm(contact_forces, dim=-1)
    contact_status[contact_forces_norm > 1.0] = 1
    return contact_status


def contact_time_left(
    states: Dict[str, Any],
    contact_time_left: Callable,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    The contact timing. We use a callable (lambda) to fetch the latest value from the controller class.

    :param states: State dictionary
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Contact timing
    """
    result = contact_time_left()
    return tensorify([result], dtype=dtype, device=device)


def object_size(
    states: Dict[str, Any],
    size: tuple[float, float, float],
    asset_name: str = "object",
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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


def dummy_contact_status(
    states: Dict[str, Any], dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Dummy contact status.

    :param states: State dictionary
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Zero tensor for contact status
    """
    return torch.zeros(4, dtype=dtype, device=device)


def base_height(
    states: Dict[str, Any], mj_model: Any, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None
) -> torch.Tensor:
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
    feet_positions_w = mj_model.get_ee_positions_w()
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
        future_feet_positions_init_frame()[:, current_goal_idx() : current_goal_idx() + obs_horizon]
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
        future_feet_positions_w()[:, current_goal_idx() : current_goal_idx() + obs_horizon]
    ).flatten()
    return tensorify(result, dtype=dtype, device=device)


def contact_pos_error(
    states: Dict[str, Any],
    mj_model: Any,
    contact_pose_w: Callable,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    The contact position error.

    """
    current_hand_pos_w = mj_model.get_ee_positions_w()[:2]
    return pos_diff(current_hand_pos_w, contact_pose_w()).flatten()


def contact_pose_b(
    states: Dict[str, Any],
    contact_pose_b: Callable,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    The contact poses in the base frame.

    :param states: State dictionary
    :param contact_pose_b: Function to get contact poses in base frame
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Contact poses in base frame tensor
    """
    result = contact_pose_b().flatten()
    return tensorify(result, dtype=dtype, device=device)


def goal_pose_diff(
    states: Dict[str, Any],
    goal_poses_w: Callable,
    asset_name: str = "object",
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Goal pose relative to the asset's root frame for all future poses.

    The quaternion is represented as (w, x, y, z). The real part is always positive.
    Returns concatenated pose differences for all future poses: [pos_diff_0, quat_diff_0, pos_diff_1, quat_diff_1, ...]
    """
    # Get data from states - avoid unnecessary tensorify calls if already tensors
    asset_pos_w = states[f"{asset_name}/base_pos_w"]  # Shape: (3,)
    asset_quat_w = states[f"{asset_name}/base_quat"]  # Shape: (4,)

    # Get goal poses queue and ensure it's a tensor
    goal_poses = goal_poses_w()  # Shape: (num_future_poses, 7)
    num_future_poses = goal_poses.shape[0]

    # Extract positions and quaternions for all goal poses
    goal_pos_w = goal_poses[:, :3]  # Shape: (num_future_poses, 3)
    goal_quat_w = goal_poses[:, 3:7]  # Shape: (num_future_poses, 4)

    # Compute position differences for all poses simultaneously
    pos_diff = goal_pos_w - asset_pos_w.unsqueeze(0)  # Shape: (num_future_poses, 3)

    # Compute quaternion differences for all poses simultaneously
    # quat_mul expects matching shapes, so we need to handle broadcasting
    quat_diff = quat_mul(
        asset_quat_w.unsqueeze(0).expand(num_future_poses, -1),  # Shape: (num_future_poses, 4)
        quat_conjugate(goal_quat_w),  # Shape: (num_future_poses, 4)
    )  # Shape: (num_future_poses, 4)

    # Reshape and concatenate efficiently
    pos_diff_flat = pos_diff.reshape(-1)  # Shape: (num_future_poses * 3,)
    quat_diff_flat = quat_diff.reshape(-1)  # Shape: (num_future_poses * 4,)

    return torch.cat([pos_diff_flat, quat_diff_flat], dim=0)  # Shape: (num_future_poses * 7,)


def object_pose_b(
    states: Dict[str, Any],
    asset_name: str = "object",
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    The pose of the object in the robot base frame.
    """
    robot_pos_w = torch.tensor(states["robot/base_pos_w"], dtype=dtype, device=device)
    robot_quat_w = torch.tensor(states["robot/base_quat"], dtype=dtype, device=device)
    object_pos_w = torch.tensor(states[f"{asset_name}/base_pos_w"], dtype=dtype, device=device)
    object_quat_w = torch.tensor(states[f"{asset_name}/base_quat"], dtype=dtype, device=device)

    object_pos_b, object_quat_b = subtract_frame_transforms(robot_pos_w, robot_quat_w, object_pos_w, object_quat_w)
    return tensorify(torch.cat([object_pos_b, object_quat_b], dim=-1), dtype=dtype, device=device)


def object_lin_vel_b(
    states: Dict[str, Any],
    asset_name: str = "object",
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    The linear velocity of the object in the robot base frame.
    """
    robot_lin_vel_w = torch.tensor(states["robot/lin_vel_w"], dtype=dtype, device=device)
    object_lin_vel_w = torch.tensor(states[f"{asset_name}/lin_vel_w"], dtype=dtype, device=device)
    robot_quat_w = torch.tensor(states["robot/base_quat"], dtype=dtype, device=device)
    relative_lin_vel_w = object_lin_vel_w - robot_lin_vel_w

    # Then rotate to robot frame using robot's orientation
    object_lin_vel_b = quat_apply(quat_conjugate(robot_quat_w), relative_lin_vel_w)

    return tensorify(object_lin_vel_b, dtype=dtype, device=device)


def object_ang_vel_b(
    states: Dict[str, Any],
    asset_name: str = "object",
    scale=1.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    The angular velocity of the object in the robot base frame.
    """
    robot_ang_vel_w = torch.tensor(states["robot/ang_vel_w"], dtype=dtype, device=device)
    object_ang_vel_w = torch.tensor(states[f"{asset_name}/ang_vel_w"], dtype=dtype, device=device)
    robot_quat_w = torch.tensor(states["robot/base_quat"], dtype=dtype, device=device)
    relative_ang_vel_w = object_ang_vel_w - robot_ang_vel_w

    # Then rotate to robot frame using robot's orientation
    object_ang_vel_b = quat_apply(quat_conjugate(robot_quat_w), relative_ang_vel_w)

    return tensorify(object_ang_vel_b, dtype=dtype, device=device) * scale


def object_pos_robot_xy_frame(
    states: Dict[str, Any],
    asset_name: str = "object",
    scale=1.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    The position of the object with the origin at robot's xy.

    :param states: State dictionary
    :param asset_name: Name of the asset
    :param scale: Scale factor
    :param dtype: Desired tensor dtype
    :param device: Desired tensor device
    :return: Position of the object in the robot's xy frame
    """
    object_pos_w = states[f"{asset_name}/base_pos_w"]
    robot_pos_w = states["robot/base_pos_w"].clone()
    robot_pos_w[2] = 0.0

    result = object_pos_w - robot_pos_w
    return result * scale


def reach_commands(
    states: Dict[str, Any],
    reach_commands: Callable,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    The reach commands for the hands in the base frame.
    """
    return reach_commands().flatten()


def waist_commands(
    states: Dict[str, Any],
    waist_commands: Callable,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    The waist commands for the waist.
    """
    return waist_commands().flatten()

import time
from typing import Dict, List, Optional, Any

import torch

from state_manager.estimators import VelocityEstimator
from utils.logger import logging
from utils.math import quat_to_rotmatrix
from utils.helpers import tensorify


def go2_low_state_handler(msg: Dict[str, List], logger: Optional[logging.Logger] = None, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """Extracts and filters the low-level state of the robot.

    Args:
        msg (Dict): Low-level state message
        logger (logging.Logger): Logger for debugging
        device (torch.device): Device to use for torch operations

    Returns:
        Dict: Filtered low-level state of the robot
    """
    if device is None:
        device = torch.device('cuda:0')

    # Extract motor states
    motor_states = msg["motor_state"][:12]  # 12 joint for the legs, the remaining 8 are unactuated
    joint_positions = tensorify([motor.q for motor in motor_states], device=device)
    joint_velocities = tensorify([motor.dq for motor in motor_states], device=device)
    joint_accelerations = tensorify([motor.ddq for motor in motor_states], device=device)
    joint_tau_est = tensorify([motor.tau_est for motor in motor_states], device=device)

    # Extract foot forces
    foot_forces = tensorify(msg["foot_force"], device=device)
    foot_forces_est = tensorify(msg["foot_force_est"], device=device)

    # Extract and filter IMU data
    imu_state = msg["imu_state"]
    gyroscope = tensorify(imu_state.gyroscope, device=device)
    accelerometer = tensorify(imu_state.accelerometer, device=device)
    quaternion = tensorify(imu_state.quaternion, device=device)

    # Filter joint states and IMU data
    alpha = 0.5  # Adjust this value based on your needs (higher = more responsive)

    if not hasattr(go2_low_state_handler, "filtered_joint_pos"):
        go2_low_state_handler.filtered_joint_pos = joint_positions
        go2_low_state_handler.filtered_joint_vel = joint_velocities
        go2_low_state_handler.filtered_joint_acc = joint_accelerations
        go2_low_state_handler.filtered_joint_tau = joint_tau_est
        go2_low_state_handler.filtered_gyro = gyroscope
        go2_low_state_handler.filtered_acc = accelerometer
        go2_low_state_handler.filtered_quat = quaternion
    else:
        go2_low_state_handler.filtered_joint_pos = (
            alpha * joint_positions + (1 - alpha) * go2_low_state_handler.filtered_joint_pos
        )
        go2_low_state_handler.filtered_joint_vel = (
            alpha * joint_velocities + (1 - alpha) * go2_low_state_handler.filtered_joint_vel
        )
        go2_low_state_handler.filtered_joint_acc = (
            alpha * joint_accelerations + (1 - alpha) * go2_low_state_handler.filtered_joint_acc
        )
        go2_low_state_handler.filtered_joint_tau = (
            alpha * joint_tau_est + (1 - alpha) * go2_low_state_handler.filtered_joint_tau
        )
        go2_low_state_handler.filtered_gyro = alpha * gyroscope + (1 - alpha) * go2_low_state_handler.filtered_gyro
        go2_low_state_handler.filtered_acc = alpha * accelerometer + (1 - alpha) * go2_low_state_handler.filtered_acc
        go2_low_state_handler.filtered_quat = alpha * quaternion + (1 - alpha) * go2_low_state_handler.filtered_quat

    # Normalize the filtered quaternion
    go2_low_state_handler.filtered_quat = go2_low_state_handler.filtered_quat / torch.norm(
        go2_low_state_handler.filtered_quat
    )

    # Construct and return the parsed states dictionary
    states = {
        "robot/joint_pos": go2_low_state_handler.filtered_joint_pos,
        "robot/joint_vel": go2_low_state_handler.filtered_joint_vel,
        "robot/joint_acc": go2_low_state_handler.filtered_joint_acc,
        "robot/joint_tau_est": go2_low_state_handler.filtered_joint_tau,
        "robot/foot_forces": foot_forces,
        "robot/foot_forces_est": foot_forces_est,
        "robot/gyroscope": go2_low_state_handler.filtered_gyro,
        "robot/accelerometer": go2_low_state_handler.filtered_acc,
        "robot/base_quat": go2_low_state_handler.filtered_quat,
    }

    return states

def g1_low_state_handler(msg: Dict[str, Any], logger: Optional[logging.Logger] = None, device: Optional[torch.device] = None):
    """Extracts and filters the joint and feet states, and returns the joint positions, joint velocities,
    feet forces, joint accelerations, estimated torques, base quaternion, base rpy, and other IMU states.

    Args:
        msg (Dict): Low Level State Unitree Message
        logger (logging.Logger): Logger for debugging
        device (torch.device): Device to use for torch operations

    Returns:
        Dict: Filtered low level states directly from the robot
    """
    if device is None:
        device = torch.device('cuda:0')
        
    # # Wait for the first message
    # while msg["tick"] == 0:
    #     time.sleep(0.01)

    leg_joint2motor_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    arm_waist_joint2motor_idx = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

    # Extract motor states
    motor_states = msg["motor_state"]

    # Extract IMU states
    imu_state = msg["imu_state"]

    # Extract raw sensor data
    joint_positions = tensorify(
        [motor_states[i].q for i in range(len(leg_joint2motor_idx + arm_waist_joint2motor_idx))],
        device=device
    )
    joint_velocities = tensorify(
        [motor_states[i].dq for i in range(len(leg_joint2motor_idx + arm_waist_joint2motor_idx))],
        device=device
    )
    joint_tau_est = tensorify(
        [motor_states[i].tau_est for i in leg_joint2motor_idx + arm_waist_joint2motor_idx],
        device=device
    )
    gyroscope = tensorify(imu_state.gyroscope, device=device)
    accelerometer = tensorify(imu_state.accelerometer, device=device)
    quaternion = tensorify(imu_state.quaternion, device=device)

    # Filter joint states and IMU data
    alpha = 0.5  # Adjust this value based on your needs (higher = more responsive)

    if not hasattr(g1_low_state_handler, "filtered_joint_pos"):
        g1_low_state_handler.filtered_joint_pos = joint_positions
        g1_low_state_handler.filtered_joint_vel = joint_velocities
        g1_low_state_handler.filtered_joint_tau = joint_tau_est
        g1_low_state_handler.filtered_gyro = gyroscope
        g1_low_state_handler.filtered_acc = accelerometer
        g1_low_state_handler.filtered_quat = quaternion
    else:
        g1_low_state_handler.filtered_joint_pos = (
            alpha * joint_positions + (1 - alpha) * g1_low_state_handler.filtered_joint_pos
        )
        g1_low_state_handler.filtered_joint_vel = (
            alpha * joint_velocities + (1 - alpha) * g1_low_state_handler.filtered_joint_vel
        )
        g1_low_state_handler.filtered_joint_tau = (
            alpha * joint_tau_est + (1 - alpha) * g1_low_state_handler.filtered_joint_tau
        )
        g1_low_state_handler.filtered_gyro = alpha * gyroscope + (1 - alpha) * g1_low_state_handler.filtered_gyro
        g1_low_state_handler.filtered_acc = alpha * accelerometer + (1 - alpha) * g1_low_state_handler.filtered_acc
        g1_low_state_handler.filtered_quat = alpha * quaternion + (1 - alpha) * g1_low_state_handler.filtered_quat

    # Normalize the filtered quaternion
    g1_low_state_handler.filtered_quat = g1_low_state_handler.filtered_quat / torch.norm(
        g1_low_state_handler.filtered_quat
    )

    states = {
        "mode_machine": msg["mode_machine"],
        "robot/joint_pos": g1_low_state_handler.filtered_joint_pos,
        "robot/joint_vel": g1_low_state_handler.filtered_joint_vel,
        "robot/joint_tau_est": g1_low_state_handler.filtered_joint_tau,
        "robot/gyroscope": g1_low_state_handler.filtered_gyro,
        "robot/accelerometer": g1_low_state_handler.filtered_acc,
        "robot/base_quat": g1_low_state_handler.filtered_quat,
        "robot/base_rpy": tensorify(imu_state.rpy, device=device),
    }

    return states

def g1_upper_low_state_handler(msg: Dict[str, Any], logger: Optional[logging.Logger] = None, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """Extracts the joint and feet states, and returns the joint positions, joint velocities,
    feet forces, joint accelerations, estimated torques, base quaternion, base rpy, and other IMU states.

    Args:
        msg (Dict): Low Level State Unitree Message
        logger (logging.Logger): Logger for debugging
        device (torch.device): Device to use for torch operations

    Returns:
        Dict: Low level states directly from the robot
    """
    if device is None:
        device = torch.device('cuda:0')
        
    # # Wait for the first message
    # while msg["tick"] == 0:
    #     time.sleep(0.01)

    joint2motor_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # Extract motor states
    motor_states = msg["motor_state"]

    # Extract IMU states
    imu_state = msg["imu_state"]
    
    # Extract raw sensor data
    joint_positions = tensorify(
        [motor_states[i].q for i in range(len(joint2motor_idx))],
        device=device
    )
    joint_velocities = tensorify(
        [motor_states[i].dq for i in range(len(joint2motor_idx))],
        device=device
    )
    joint_tau_est = tensorify(
        [motor_states[i].tau_est for i in joint2motor_idx],
        device=device
    )
    gyroscope = tensorify(imu_state.gyroscope, device=device)
    accelerometer = tensorify(imu_state.accelerometer, device=device)
    quaternion = tensorify(imu_state.quaternion, device=device)

    # Filter joint states and IMU data
    alpha = 0.5  # Adjust this value based on your needs (higher = more responsive)

    if not hasattr(g1_upper_low_state_handler, "filtered_joint_pos"):
        g1_upper_low_state_handler.filtered_joint_pos = joint_positions
        g1_upper_low_state_handler.filtered_joint_vel = joint_velocities
        g1_upper_low_state_handler.filtered_joint_tau = joint_tau_est
        g1_upper_low_state_handler.filtered_gyro = gyroscope
        g1_upper_low_state_handler.filtered_acc = accelerometer
        g1_upper_low_state_handler.filtered_quat = quaternion
    else:
        g1_upper_low_state_handler.filtered_joint_pos = (
            alpha * joint_positions + (1 - alpha) * g1_upper_low_state_handler.filtered_joint_pos
        )
        g1_upper_low_state_handler.filtered_joint_vel = (
            alpha * joint_velocities + (1 - alpha) * g1_upper_low_state_handler.filtered_joint_vel
        )
        g1_upper_low_state_handler.filtered_joint_tau = (
            alpha * joint_tau_est + (1 - alpha) * g1_upper_low_state_handler.filtered_joint_tau
        )
        g1_upper_low_state_handler.filtered_gyro = alpha * gyroscope + (1 - alpha) * g1_upper_low_state_handler.filtered_gyro
        g1_upper_low_state_handler.filtered_acc = alpha * accelerometer + (1 - alpha) * g1_upper_low_state_handler.filtered_acc
        g1_upper_low_state_handler.filtered_quat = alpha * quaternion + (1 - alpha) * g1_upper_low_state_handler.filtered_quat

    # Normalize the filtered quaternion
    g1_upper_low_state_handler.filtered_quat = g1_upper_low_state_handler.filtered_quat / torch.norm(
        g1_upper_low_state_handler.filtered_quat
    )

    states = {
        "mode_machine": msg["mode_machine"],
        "robot/joint_pos": g1_upper_low_state_handler.filtered_joint_pos,
        "robot/joint_vel": g1_upper_low_state_handler.filtered_joint_vel,
        "robot/joint_tau_est": g1_upper_low_state_handler.filtered_joint_tau,
        "robot/gyroscope": g1_upper_low_state_handler.filtered_gyro,
        "robot/accelerometer": g1_upper_low_state_handler.filtered_acc,
        "robot/base_quat": g1_upper_low_state_handler.filtered_quat,
        "robot/base_rpy": tensorify(imu_state.rpy, device=device),
    }

    return states

def g1_lower_low_state_handler(msg: Dict[str, Any], logger: Optional[logging.Logger] = None, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """Extracts the joint and feet states, and returns the joint positions, joint velocities,
    feet forces, joint accelerations, estimated torques, base quaternion, base rpy, and other IMU states.

    Args:
        msg (Dict): Low Level State Unitree Message
        logger (logging.Logger): Logger for debugging
        device (torch.device): Device to use for torch operations

    Returns:
        Dict: Low level states directly from the robot
    """
    if device is None:
        device = torch.device('cuda:0')
        
    # # Wait for the first message
    # while msg["tick"] == 0:
    #     time.sleep(0.01)

    joint2motor_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] + [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

    # Extract motor states
    motor_states = msg["motor_state"]

    # Extract IMU states
    imu_state = msg["imu_state"]

    states = {
        "mode_machine": msg["mode_machine"],
        "robot/joint_pos": tensorify(
            [motor_states[i].q for i in joint2motor_idx],
            device=device
        ),
        "robot/joint_vel": tensorify(
            [motor_states[i].dq for i in joint2motor_idx],
            device=device
        ),
        "robot/joint_tau_est": tensorify(
            [motor_states[i].tau_est for i in joint2motor_idx],
            device=device
        ),
        "robot/gyroscope": tensorify(imu_state.gyroscope, device=device),
        "robot/accelerometer": tensorify(imu_state.accelerometer, device=device),
        "robot/base_quat": tensorify(imu_state.quaternion, device=device),
        "robot/base_rpy": tensorify(imu_state.rpy, device=device),
    }

    return states



def vicon_handler(msg: Dict[str, float], logger: Optional[logging.Logger] = None, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """
    Handles Vicon messages to extract base states and estimate velocities.

    Args:
        msg (Dict): Vicon Position Message
        logger (Optional[logging.Logger]): Logger for debugging
        device (torch.device): Device to use for torch operations

    Returns:
        Dict[str, torch.Tensor]: Base states from the Vicon Receiver including velocities
    """
    if device is None:
        device = torch.device('cuda:0')

    # Position offsets in the Vicon frame (in millimeters)
    x_offset, y_offset, z_offset = 0.0, 0.0, -230.0

    # Initialize velocity estimator once using a singleton pattern
    if not hasattr(vicon_handler, "velocity_estimator"):
        vicon_handler.velocity_estimator = VelocityEstimator(method="finite_diff", alpha=0.5, device=device)

    # Calculate base position in meters
    base_position = tensorify(
        [(msg["x_trans"] + x_offset) * 0.001, (msg["y_trans"] + y_offset) * 0.001, (msg["z_trans"] + z_offset) * 0.001],
        device=device
    )
    # Base quaternion in order (w, x, y, z)
    base_quaternion = tensorify([msg["w"], msg["x_rot"], msg["y_rot"], msg["z_rot"]], device=device)

    # Initialize or update filtered quaternion using spherical linear interpolation (slerp)
    if not hasattr(vicon_handler, "filtered_quaternion"):
        vicon_handler.filtered_quaternion = base_quaternion
    else:
        # TODO: Check if this is necessary (is this about makeing quat unique?)
        dot_product = torch.dot(vicon_handler.filtered_quaternion, base_quaternion)
        if dot_product < 0:
            base_quaternion = -base_quaternion  # Ensure shortest path
        vicon_handler.filtered_quaternion = 0.5 * base_quaternion + 0.5 * vicon_handler.filtered_quaternion
        vicon_handler.filtered_quaternion /= torch.norm(vicon_handler.filtered_quaternion)  # Normalize

    # Estimate velocities
    current_time = time.time()
    lin_vel_w, ang_vel_w = vicon_handler.velocity_estimator.update(
        base_position, vicon_handler.filtered_quaternion, current_time, logger
    )

    # Transform linear velocity to base frame
    rotation_matrix = quat_to_rotmatrix(vicon_handler.filtered_quaternion, order="wxyz")
    lin_vel_b = torch.matmul(rotation_matrix.T, lin_vel_w)
    # logger.debug(f"Base position: {base_position}")
    return {
        "robot/base_pos_w": base_position,
        "robot/base_quat": vicon_handler.filtered_quaternion,
        "robot/lin_vel_w": lin_vel_w,
        "robot/lin_vel_b": lin_vel_b,
    }
    
def vicon_object_handler(msg: Dict[str, float], logger: Optional[logging.Logger] = None, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """
    Handles Vicon messages to extract base states and estimate velocities.

    Args:
        msg (Dict): Vicon Position Message
        logger (Optional[logging.Logger]): Logger for debugging
        device (torch.device): Device to use for torch operations

    Returns:
        Dict[str, torch.Tensor]: Base states from the Vicon Receiver including velocities
    """
    if device is None:
        device = torch.device('cuda:0')

    # Position offsets in the Vicon frame (in millimeters)
    x_offset, y_offset, z_offset = 0.0, 0.0, 0.0

    # Initialize velocity estimator once using a singleton pattern
    if not hasattr(vicon_object_handler, "velocity_estimator"):
        vicon_object_handler.velocity_estimator = VelocityEstimator(method="finite_diff", alpha=0.25, device=device)

    # Calculate base position in meters
    base_position = tensorify(
        [(msg["x_trans"] + x_offset) * 0.001, (msg["y_trans"] + y_offset) * 0.001, (msg["z_trans"] + z_offset) * 0.001],
        device=device
    )
    # Base quaternion in order (w, x, y, z)
    base_quaternion = tensorify([msg["w"], msg["x_rot"], msg["y_rot"], msg["z_rot"]], device=device)

    # Initialize or update filtered quaternion using spherical linear interpolation (slerp)
    if not hasattr(vicon_object_handler, "filtered_quaternion"):
        vicon_object_handler.filtered_quaternion = base_quaternion
    else:
        # TODO: Check if this is necessary (is this about makeing quat unique?)
        dot_product = torch.dot(vicon_object_handler.filtered_quaternion, base_quaternion)
        if dot_product < 0:
            base_quaternion = -base_quaternion  # Ensure shortest path
        vicon_object_handler.filtered_quaternion = 0.5 * base_quaternion + 0.5 * vicon_object_handler.filtered_quaternion
        vicon_object_handler.filtered_quaternion /= torch.norm(vicon_object_handler.filtered_quaternion)  # Normalize

    # Estimate velocities
    current_time = time.time()
    lin_vel_w, ang_vel_w = vicon_object_handler.velocity_estimator.update(
        base_position, vicon_object_handler.filtered_quaternion, current_time, logger
    )

    # Transform linear velocity to base frame
    rotation_matrix = quat_to_rotmatrix(vicon_object_handler.filtered_quaternion, order="wxyz")
    lin_vel_b = torch.matmul(rotation_matrix.T, lin_vel_w)
    
    # logger.debug(f"Object Base position: {base_position}")
    
    return {
        "object/base_pos_w": base_position,
        "object/base_quat": vicon_object_handler.filtered_quaternion,
        "object/ang_vel_w": ang_vel_w,
        "object/lin_vel_w": lin_vel_w,
        "object/lin_vel_b": lin_vel_b,
    }


def sport_mode_state_handler(msg: Dict[str, Any], logger: Optional[logging.Logger] = None, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """Uses the Sports Mode states of the Unitree SDK to extract bose position, base velocity, and base orientation

    Args:
        msg (Dict): High Level Sports State Unitree Message
        logger (logging.Logger): Logger for debugging
        device (torch.device): Device to use for torch operations

    Returns:
        Dict: High level states directly from the robot
    """
    if device is None:
        device = torch.device('cuda:0')
        
    # Singleton pattern for velocity estimator
    if not hasattr(sport_mode_state_handler, "velocity_estimator"):
        sport_mode_state_handler.velocity_estimator = VelocityEstimator(method="finite_diff", device=device)

    base_pos_w = tensorify(msg["position"], device=device)
    base_quat = tensorify(msg["imu_state"].quaternion, device=device)

    # Estimate velocities
    current_time = time.time()
    lin_vel_w, ang_vel_w = sport_mode_state_handler.velocity_estimator.update(
        base_pos_w, base_quat, current_time, logger
    )
    
    states = {
        "robot/base_pos_w": base_pos_w,
        "robot/lin_vel_b": tensorify(msg["velocity"], device=device),
        "robot/lin_vel_w": lin_vel_w,
        "robot/ang_vel_w": ang_vel_w,
    }

    return states


def object_state_handler(msg: Dict[str, Any], logger: Optional[logging.Logger] = None, device: Optional[torch.device] = None):
    """Extracts the object state, and returns the object position, object velocity, object orientation, and object angular velocity.
    This message is published by the Vicon Receiver.
    
    Args:
        msg (Dict): Object state message
        logger (logging.Logger): Logger for debugging
        device (torch.device): Device to use for torch operations

    Returns:
        Dict: Object state
    """
    if device is None:
        device = torch.device('cuda:0')

    base_pos_w = tensorify(msg["position"], device=device)
    base_quat = tensorify(msg["imu_state"].quaternion, device=device)
    lin_vel_w = tensorify(msg["velocity"], device=device)
    ang_vel_w = tensorify(msg["imu_state"].gyroscope, device=device)
    

    lin_vel_b = torch.matmul(quat_to_rotmatrix(base_quat, order="wxyz").T, lin_vel_w)
    ang_vel_b = torch.matmul(quat_to_rotmatrix(base_quat, order="wxyz").T, ang_vel_w)

    return {
        "object/base_pos_w": base_pos_w,
        "object/base_quat": base_quat,
        "object/lin_vel_w": lin_vel_w,
        "object/ang_vel_w": ang_vel_w,
        "object/lin_vel_b": lin_vel_b,
        "object/ang_vel_b": ang_vel_b,
    }
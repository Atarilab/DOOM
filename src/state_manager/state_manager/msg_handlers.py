import time
from typing import Dict, List, Optional

import numpy as np
from state_manager.estimators import VelocityEstimator
from utils.logger import logging
from utils.math import quat_to_rotmatrix


def low_state_handler(msg: Dict[str, List], logger: Optional[logging.Logger] = None):
    """Extracts the joint and feet states, and returns the joint positions, joint velocities,
    feet forces, joint accelerations, estimated torques, base quaternion, base rpy, and other IMU states.

    Args:
        msg (Dict): Low Level Unitree Message
        logger (logging.Logger): Logger for debugging

    Returns:
        Dict: Low level states directly from the robot
    """
    # Extract motor states directly without reordering
    motor_states = msg["motor_state"][:12]  # 12 joint for the legs, the remaining 8 are unactuated
    joint_positions = np.array([motor.q for motor in motor_states])
    joint_velocities = np.array([motor.dq for motor in motor_states])
    joint_accelerations = np.array([motor.ddq for motor in motor_states])
    joint_tau_est = np.array([motor.tau_est for motor in motor_states])

    # Extract foot forces directly without reordering
    foot_forces = msg["foot_force"]

    # Extract IMU states
    imu_state = msg["imu_state"]
    
    # Extract and filter IMU data
    try:        
        # Get gyroscope data
        gyroscope = np.array([
            imu_state.gyroscope[0],
            imu_state.gyroscope[1],
            imu_state.gyroscope[2]
        ])
        
        # Get accelerometer data
        accelerometer = np.array([
            imu_state.accelerometer[0],
            imu_state.accelerometer[1],
            imu_state.accelerometer[2]
        ])
        
        alpha = 0.3  # Adjust this value based on your needs (higher = more responsive)
        
        # Filter gyroscope data
        if not hasattr(low_state_handler, "filtered_gyro"):
            low_state_handler.filtered_gyro = gyroscope
        else:
            low_state_handler.filtered_gyro = alpha * gyroscope + (1 - alpha) * low_state_handler.filtered_gyro
            
        # Filter accelerometer data
        if not hasattr(low_state_handler, "filtered_acc"):
            low_state_handler.filtered_acc = accelerometer
        else:
            low_state_handler.filtered_acc = alpha * accelerometer + (1 - alpha) * low_state_handler.filtered_acc
            
    except (AttributeError, IndexError) as e:
        if logger:
            logger.warning(f"Error accessing IMU state attributes: {e}")
        # Provide default values if IMU data is not available
        gyroscope = np.zeros(3)
        accelerometer = np.zeros(3)
        low_state_handler.filtered_gyro = np.zeros(3)
        low_state_handler.filtered_acc = np.zeros(3)

    # Construct and return the parsed states dictionary
    states = {
        "joint_pos": joint_positions,
        "joint_vel": joint_velocities,
        "joint_acc": joint_accelerations,
        "joint_tau_est": joint_tau_est,
        "foot_forces": foot_forces,
        "gyroscope": low_state_handler.filtered_gyro,
        "accelerometer": low_state_handler.filtered_acc,
        "base_quat": imu_state.quaternion,
    }

    return states


def vicon_handler(msg: Dict[str, float], logger: Optional[logging.Logger] = None):
    """
    Vicon msg handler with velocity estimation.

    Args:
        msg (Dict): Vicon Position Message
        logger (logging.Logger): Logger for debugging

    Returns:
        Dict: Base states from the Vicon Receiver including velocities
    """
    # Offsets for position of the robot base from the vicon frame (if using Go2with6markers)
    # x_offset = 0.0
    # y_offset = 62.5
    # z_offset = -75.0

    x_offset = 0.0
    y_offset = 0.0
    z_offset = 0.0

    # Singleton pattern for velocity estimator
    if not hasattr(vicon_handler, "velocity_estimator"):
        vicon_handler.velocity_estimator = VelocityEstimator(method="finite_diff", alpha=0.15)

    # Base Position (in m)
    base_pos = np.array(
        [
            (msg["y_trans"] + y_offset) * 0.001,
            -(msg["x_trans"] + x_offset) * 0.001,
            (msg["z_trans"] + z_offset) * 0.001,
        ]
    )

    # Base quaternion (w, x, y, z)
    base_quat = np.array(
        [
            msg["w"],
            msg["x_rot"],
            msg["y_rot"],
            msg["z_rot"],
        ]
    )
    
    # Apply a -90-degree rotation around the Z axis to align the robot's frame with the world frame
    # This creates a rotation quaternion for -90 degrees around Z axis
    rot_neg90_z = np.array([np.cos(-np.pi/4), 0, 0, np.sin(-np.pi/4)])  # w, x, y, z format
    
    # Multiply the quaternions to apply the rotation
    # Quaternion multiplication formula:
    # q1 * q2 = [w1*w2 - x1*x2 - y1*y2 - z1*z2,
    #            w1*x2 + x1*w2 + y1*z2 - z1*y2,
    #            w1*y2 - x1*z2 + y1*w2 + z1*x2,
    #            w1*z2 + x1*y2 - y1*x2 + z1*w2]
    w1, x1, y1, z1 = base_quat
    w2, x2, y2, z2 = rot_neg90_z
    
    base_quat = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
    
    # Swap roll and pitch components to fix orientation
    # Original: [w, x, y, z] where x is roll, y is pitch, z is yaw
    # New: [w, y, x, z] where y is now roll, x is now pitch, z is still yaw
    w, x, y, z = base_quat
    base_quat = np.array([w, y, x, z])
    
    # Invert the pitch component (now at index 2) to fix the pitch direction
    base_quat[2] = -base_quat[2]

    # Estimate velocities using EKF
    current_timestamp = time.time()
    lin_vel_w, ang_vel_w = vicon_handler.velocity_estimator.update(base_pos, base_quat, current_timestamp, logger)

    # Convert quaternion to a rotation matrix
    rotation_matrix = quat_to_rotmatrix(base_quat, order="wxyz")
    # Transform linear velocity to base frame
    lin_vel_b = np.dot(rotation_matrix.T, lin_vel_w)

    states = {
        "base_pos_w": base_pos,
        'base_quat': base_quat,
        "lin_vel_w": lin_vel_w.tolist(),  # Linear velocities in world frame
        "lin_vel_b": lin_vel_b,
        # 'ang_vel_w': angular_velocities.tolist(),  # Angular velocities in world frame
    }

    return states


def sport_mode_state_handler(msg: Dict[str, List], logger: Optional[logging.Logger] = None):
    """Uses the Sports Mode states of the Unitree SDK to extract bose position, base velocity, and base orientation

    Args:
        msg (Dict): High Level Sports State Unitree Message
        logger (logging.Logger): Logger for debugging

    Returns:
        Dict: High level states directly from the robot
    """
    # Singleton pattern for velocity estimator
    if not hasattr(sport_mode_state_handler, "velocity_estimator"):
        sport_mode_state_handler.velocity_estimator = VelocityEstimator(method="finite_diff")

    base_pos_w = msg["position"]
    base_quat = msg["imu_state"].quaternion

    states = {
        "base_pos_w": base_pos_w,
        "lin_vel_b": msg["velocity"],
    }

    return states

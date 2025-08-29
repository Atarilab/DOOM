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
    foot_forces_est = msg["foot_force_est"]

    # Extract IMU states
    imu_state = msg["imu_state"]
    
    # Filter parameters
    alpha = 0.5  # Adjust this value based on your needs (higher = more responsive)
    
    # Initialize or update filtered joint states
    if not hasattr(low_state_handler, "filtered_joint_pos"):
        low_state_handler.filtered_joint_pos = joint_positions
        low_state_handler.filtered_joint_vel = joint_velocities
        low_state_handler.filtered_joint_acc = joint_accelerations
        low_state_handler.filtered_joint_tau = joint_tau_est
    else:
        low_state_handler.filtered_joint_pos = alpha * joint_positions + (1 - alpha) * low_state_handler.filtered_joint_pos
        low_state_handler.filtered_joint_vel = alpha * joint_velocities + (1 - alpha) * low_state_handler.filtered_joint_vel
        low_state_handler.filtered_joint_acc = alpha * joint_accelerations + (1 - alpha) * low_state_handler.filtered_joint_acc
        low_state_handler.filtered_joint_tau = alpha * joint_tau_est + (1 - alpha) * low_state_handler.filtered_joint_tau
    
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
            
        # Get and filter quaternion
        quaternion = np.array([
            imu_state.quaternion[0],
            imu_state.quaternion[1],
            imu_state.quaternion[2],
            imu_state.quaternion[3]
        ])
        
        # Filter quaternion
        if not hasattr(low_state_handler, "filtered_quat"):
            low_state_handler.filtered_quat = quaternion
        else:
            # For quaternions, we need to ensure the filtered result is still a valid quaternion
            # We'll use spherical linear interpolation (slerp) for quaternion filtering
            dot_product = np.dot(low_state_handler.filtered_quat, quaternion)
            if dot_product < 0:
                quaternion = -quaternion  # Ensure shortest path
            low_state_handler.filtered_quat = alpha * quaternion + (1 - alpha) * low_state_handler.filtered_quat
            # Normalize the filtered quaternion
            low_state_handler.filtered_quat = low_state_handler.filtered_quat / np.linalg.norm(low_state_handler.filtered_quat)
            
    except (AttributeError, IndexError) as e:
        if logger:
            logger.warning(f"Error accessing IMU state attributes: {e}")
        # Provide default values if IMU data is not available
        gyroscope = np.zeros(3)
        accelerometer = np.zeros(3)
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        low_state_handler.filtered_gyro = np.zeros(3)
        low_state_handler.filtered_acc = np.zeros(3)
        low_state_handler.filtered_quat = quaternion

    # Construct and return the parsed states dictionary
    states = {
        "joint_pos": low_state_handler.filtered_joint_pos,
        "joint_vel": low_state_handler.filtered_joint_vel,
        "joint_acc": low_state_handler.filtered_joint_acc,
        "joint_tau_est": low_state_handler.filtered_joint_tau,
        "foot_forces": foot_forces,
        "foot_forces_est": foot_forces_est,
        "gyroscope": low_state_handler.filtered_gyro,
        "accelerometer": low_state_handler.filtered_acc,
        "base_quat": low_state_handler.filtered_quat,
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
        vicon_handler.velocity_estimator = VelocityEstimator(method="finite_diff", alpha=0.5)

    # Base Position (in m)
    base_pos = np.array(
        [   
            (msg["x_trans"] + x_offset) * 0.001,
            (msg["y_trans"] + y_offset) * 0.001,
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
    
    # Filter parameters
    alpha = 0.5  # Same alpha as in low_state_handler for consistency
    
    # Initialize or update filtered quaternion
    if not hasattr(vicon_handler, "filtered_quat"):
        vicon_handler.filtered_quat = base_quat
    else:
        # For quaternions, we need to ensure the filtered result is still a valid quaternion
        # We'll use spherical linear interpolation (slerp) for quaternion filtering
        dot_product = np.dot(vicon_handler.filtered_quat, base_quat)
        if dot_product < 0:
            base_quat = -base_quat  # Ensure shortest path
        vicon_handler.filtered_quat = alpha * base_quat + (1 - alpha) * vicon_handler.filtered_quat
        # Normalize the filtered quaternion
        vicon_handler.filtered_quat = vicon_handler.filtered_quat / np.linalg.norm(vicon_handler.filtered_quat)

    # Estimate velocities using EKF
    current_timestamp = time.time()
    lin_vel_w, ang_vel_w = vicon_handler.velocity_estimator.update(base_pos, vicon_handler.filtered_quat, current_timestamp, logger)

    # Convert quaternion to a rotation matrix
    rotation_matrix = quat_to_rotmatrix(vicon_handler.filtered_quat, order="wxyz")
    # Transform linear velocity to base frame
    lin_vel_b = np.dot(rotation_matrix.T, lin_vel_w)

    states = {
        "base_pos_w": base_pos,
        'base_quat': vicon_handler.filtered_quat,
        "lin_vel_w": lin_vel_w.tolist(),  # Linear velocities in world frame
        "lin_vel_b": lin_vel_b,
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


def object_state_handler(msg: Dict[str, List], logger: Optional[logging.Logger] = None):
    """Extracts the object state, and returns the object position, object velocity, object orientation, and object angular velocity."""
    
    base_pos_w = msg["position"]
    base_quat = msg["imu_state"].quaternion
    #lin_vel_w = msg["velocity"]
    #ang_vel = msg["imu_state"].gyroscope
    #
    #lin_vel_b = np.dot(quat_to_rotmatrix(base_quat, order="wxyz").T, lin_vel_w)
    #ang_vel_b = np.dot(quat_to_rotmatrix(base_quat, order="wxyz").T, ang_vel)
    
    return {
        "object_pos_w": list(np.array(base_pos_w) - np.array([0.7, 0.0, 0.0] )), # I dont really know what value needs to be substracted here. I basically do try-and-error with the box size.
        "object_quat": base_quat,
        #"object_lin_vel_w": lin_vel_w,
        #"object_ang_vel": ang_vel,
        #"object_lin_vel_b": lin_vel_b,
        #"object_ang_vel_b": ang_vel_b,
        }

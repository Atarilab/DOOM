import time
from typing import Dict, List, Optional

import numpy as np

from state_manager.estimators import VelocityEstimator
from utils.logger import logging
from utils.math import quat_to_rotmatrix


def go2_low_state_handler(msg: Dict[str, List], logger: Optional[logging.Logger] = None) -> Dict[str, np.ndarray]:
    """Extracts and filters the low-level state of the robot.

    Args:
        msg (Dict): Low-level state message
        logger (logging.Logger): Logger for debugging

    Returns:
        Dict: Filtered low-level state of the robot
    """

    # Extract motor states
    motor_states = msg["motor_state"][:12]  # 12 joint for the legs, the remaining 8 are unactuated
    joint_positions = np.array([motor.q for motor in motor_states])
    joint_velocities = np.array([motor.dq for motor in motor_states])
    joint_accelerations = np.array([motor.ddq for motor in motor_states])
    joint_tau_est = np.array([motor.tau_est for motor in motor_states])

    # Extract foot forces
    foot_forces = np.array(msg["foot_force"])
    foot_forces_est = np.array(msg["foot_force_est"])

    # Extract and filter IMU data
    imu_state = msg["imu_state"]
    gyroscope = np.array(imu_state.gyroscope)
    accelerometer = np.array(imu_state.accelerometer)
    quaternion = np.array(imu_state.quaternion)

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
    go2_low_state_handler.filtered_quat = go2_low_state_handler.filtered_quat / np.linalg.norm(
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


def go2_vicon_handler(msg: Dict[str, float], logger: Optional[logging.Logger] = None) -> Dict[str, np.ndarray]:
    """
    Handles Vicon messages to extract base states and estimate velocities.

    Args:
        msg (Dict): Vicon Position Message
        logger (Optional[logging.Logger]): Logger for debugging

    Returns:
        Dict[str, np.ndarray]: Base states from the Vicon Receiver including velocities
    """

    # Position offsets in the Vicon frame (in millimeters)
    x_offset, y_offset, z_offset = 0.0, 0.0, 0.0

    # Initialize velocity estimator once using a singleton pattern
    if not hasattr(go2_vicon_handler, "velocity_estimator"):
        go2_vicon_handler.velocity_estimator = VelocityEstimator(method="finite_diff", alpha=0.5)

    # Calculate base position in meters
    base_position = np.array(
        [(msg["x_trans"] + x_offset) * 0.001, (msg["y_trans"] + y_offset) * 0.001, (msg["z_trans"] + z_offset) * 0.001]
    )

    # Base quaternion in order (w, x, y, z)
    base_quaternion = np.array([msg["w"], msg["x_rot"], msg["y_rot"], msg["z_rot"]])

    # Initialize or update filtered quaternion using spherical linear interpolation (slerp)
    if not hasattr(go2_vicon_handler, "filtered_quaternion"):
        go2_vicon_handler.filtered_quaternion = base_quaternion
    else:
        # TODO: Check if this is required (is this about makeing quat unique?)
        dot_product = np.dot(go2_vicon_handler.filtered_quaternion, base_quaternion)
        if dot_product < 0:
            base_quaternion = -base_quaternion  # Ensure shortest path
        go2_vicon_handler.filtered_quaternion = 0.5 * base_quaternion + 0.5 * go2_vicon_handler.filtered_quaternion
        go2_vicon_handler.filtered_quaternion /= np.linalg.norm(go2_vicon_handler.filtered_quaternion)  # Normalize

    # Estimate velocities
    current_time = time.time()
    lin_vel_w, ang_vel_w = go2_vicon_handler.velocity_estimator.update(
        base_position, go2_vicon_handler.filtered_quaternion, current_time, logger
    )

    # Transform linear velocity to base frame
    rotation_matrix = quat_to_rotmatrix(go2_vicon_handler.filtered_quaternion, order="wxyz")
    lin_vel_b = np.dot(rotation_matrix.T, lin_vel_w)

    return {
        "robot/base_pos_w": base_position,
        "robot/base_quat": go2_vicon_handler.filtered_quaternion,
        "robot/lin_vel_w": lin_vel_w.tolist(),
        "robot/lin_vel_b": lin_vel_b,
    }


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
        "robot/base_pos_w": base_pos_w,
        "robot/lin_vel_b": msg["velocity"],
    }

    return states


def object_state_handler(msg: Dict[str, List], logger: Optional[logging.Logger] = None):
    """Extracts the object state, and returns the object position, object velocity, object orientation, and object angular velocity."""

    base_pos_w = msg["position"]
    base_quat = msg["imu_state"].quaternion
    lin_vel_w = msg["velocity"]
    ang_vel_w = msg["imu_state"].gyroscope

    lin_vel_b = np.dot(quat_to_rotmatrix(base_quat, order="wxyz").T, lin_vel_w)
    ang_vel_b = np.dot(quat_to_rotmatrix(base_quat, order="wxyz").T, ang_vel_w)

    return {
        "object/base_pos_w": base_pos_w,
        "object/base_quat": base_quat,
        "object/lin_vel_w": lin_vel_w,
        "object/ang_vel_w": ang_vel_w,
        "object/lin_vel_b": lin_vel_b,
        "object/ang_vel_b": ang_vel_b,
    }


def g1_low_state_handler(msg: Dict[str, List], logger: Optional[logging.Logger] = None):
    """Extracts the joint and feet states, and returns the joint positions, joint velocities,
    feet forces, joint accelerations, estimated torques, base quaternion, base rpy, and other IMU states.

    Args:
        msg (Dict): Low Level State Unitree Message
        logger (logging.Logger): Logger for debugging

    Returns:
        Dict: Low level states directly from the robot
    """
    # # Wait for the first message
    # while msg["tick"] == 0:
    #     time.sleep(0.01)

    leg_joint2motor_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    arm_waist_joint2motor_idx = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

    # Extract motor states
    motor_states = msg["motor_state"]

    # Extract IMU states
    imu_state = msg["imu_state"]

    states = {
        "mode_machine": msg["mode_machine"],
        "robot/joint_pos": np.array(
            [motor_states[i].q for i in range(len(leg_joint2motor_idx + arm_waist_joint2motor_idx))]
        ),
        "robot/joint_vel": np.array(
            [motor_states[i].dq for i in range(len(leg_joint2motor_idx + arm_waist_joint2motor_idx))]
        ),
        "robot/joint_tau_est": np.array(
            [motor_states[i].tau_est for i in leg_joint2motor_idx + arm_waist_joint2motor_idx]
        ),
        "robot/gyroscope": imu_state.gyroscope,
        "robot/accelerometer": imu_state.accelerometer,
        "robot/base_quat": imu_state.quaternion,
        "robot/base_rpy": imu_state.rpy,
    }

    return states

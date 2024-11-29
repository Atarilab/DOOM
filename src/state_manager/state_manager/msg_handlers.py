import time

from typing import Dict, List
from utils.logger import logging

from state_manager.estimators import ViconVelocityEstimator

def vicon_handler(msg: Dict[str, List], logger: logging.Logger):
    """
    Vicon msg handler with velocity estimation.

    Args:
        msg (Dict): Vicon Position Message
        logger (logging.Logger): Logger for debugging

    Returns:
        Dict: Base states from the Vicon Receiver including velocities
    """
    # Offsets for position
    x_offset = 0.0
    y_offset = 62.5 
    z_offset = -75.0  

    # Singleton pattern for velocity estimator
    if not hasattr(vicon_handler, 'velocity_estimator'):
        vicon_handler.velocity_estimator = ViconVelocityEstimator()

    # Base Position (in m)
    base_pos = [
        (msg['x_trans'] + x_offset)*0.001,
        (msg['y_trans'] + y_offset)*0.001, 
        (msg['z_trans'] + z_offset)*0.001
    ]

    # Base quaternion
    base_quat = [
        msg['x_rot'],
        msg['y_rot'],
        msg['z_rot'],
        msg['w']
    ]

    # Estimate velocities using EKF
    current_timestamp = time.time()
    linear_velocities, angular_velocities = vicon_handler.velocity_estimator.ekf_update(
        base_pos, base_quat, current_timestamp
    )

    states = {
        'base_pos_w': base_pos,
        'base_quat': base_quat,
        'lin_vel_w': linear_velocities.tolist(),  # Linear velocities in world frame
        'ang_vel_w': angular_velocities.tolist()  # Angular velocities in world frame
    }
    
    # logger.debug(f"{states['lin_vel_w']}, {states['ang_vel_w']}")
    
    return states
    
def low_state_handler(msg: Dict[str, List], logger: logging.Logger):
    """Re-orders the joint and feet states [FL, FR, RL, RR], and returns the joint positions, joint velocities,
    feet forces, joint accelerations, estimated torques, base quaternion, base rpy and other IMU states.

    Args:
        msg (Dict): Low Level Unitree Message
        logger (logging.Logger): Logger for debugging

    Returns:
        Dict: Low level states directly from the robot
    """
    # Define the desired order of leg indices (ref to `unitree_legged_const.py`)
    leg_order = [
        # FL leg: 3, 4, 5
        3, 4, 5,
        # FR leg: 0, 1, 2
        0, 1, 2,
        # RL leg: 9, 10, 11
        9, 10, 11,
        # RR leg: 6, 7, 8
        6, 7, 8
    ]
    
    # Reorder motor states based on the defined leg order
    motor_states = msg['motor_state']
    reordered_motor_states = [motor_states[idx] for idx in leg_order]
    
    # Extract joint states from reordered motor states
    joint_positions = [motor.q for motor in reordered_motor_states]
    joint_velocities = [motor.dq for motor in reordered_motor_states]
    joint_accelerations = [motor.ddq for motor in reordered_motor_states]
    joint_tau_est = [motor.tau_est for motor in reordered_motor_states]
    
    # Extract IMU states
    imu_state = msg['imu_state']
    
    # Construct and return the parsed states dictionary
    states = {
        'joint_pos': joint_positions,
        'joint_vel': joint_velocities,
        'joint_acc': joint_accelerations,
        'joint_tau_est': joint_tau_est,
        'foot_forces': [msg['foot_force'][idx] for idx in [1, 0, 3, 2]],  # Reorder foot forces
        'foot_force_est': [msg['foot_force_est'][idx] for idx in [1, 0, 3, 2]],  # Reorder estimated foot forces
        'quat': imu_state.quaternion,
        'rpy': imu_state.rpy,
        'gyroscope': imu_state.gyroscope,
        'accelerometer': imu_state.accelerometer
    }
    # logger.debug(f"Received low state at {time.time()}")
    return states
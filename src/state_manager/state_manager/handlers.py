from typing import Dict, List
from utils.logger import logging


def vicon_handler(msg: Dict[str, List], logger: logging.Logger):

    """Returns base position in the world frame and the base orientation in quaternions.

    Args:
        msg (Dict): Vicon Position Message
        logger (logging.Logger): Logger for debugging

    Returns:
        Dict: Base states from the Vicon Receiver
    """
    states = {
        'base_pos_w': [
            msg['x_trans'],
            msg['y_trans'], 
            msg['z_trans']
        ],
        'base_quat': [
            msg['x_rot'],
            msg['y_rot'],
            msg['z_rot'],
            msg['w']
        ]
    }
    # logger.debug(f"Received Vicon data at {time.time()}")\
    # logger.debug(states)
    
    return states

def sport_states_handler(msg: Dict[str, List], logger: logging.Logger):
    return {key: msg[key] for key in ["body_height", "imu_state", "position", "yaw speed", "velocity"] if key in msg}
    
def low_state_handler(msg: Dict[str, List], logger: logging.Logger):
    """Re-orders the joint and feet states [FL, FR, RL, RR], and returns the joint positions, joint velocities,
    feet forces, joint accelerations, estimated torques, base quaternion, base rpy and other IMU states.

    Args:
        msg (Dict): Low Level Unitree Message
        logger (logging.Logger): Logger for debugging

    Returns:
        Dict: Low level states directly from the robot
    """
    
    # Reorder motor states based on the defined leg order
    motor_states = msg['motor_state']
    
    # Extract joint states
    joint_positions = [motor.q for motor in motor_states]
    joint_velocities = [motor.dq for motor in motor_states]
    joint_accelerations = [motor.ddq for motor in motor_states]
    joint_tau_est = [motor.tau_est for motor in motor_states]
    
    # Extract IMU states
    imu_state = msg['imu_state']
    
    # Construct and return the parsed states dictionary
    states = {
        'motor/joint_pos': joint_positions,
        'motor/joint_vel': joint_velocities,
        'motor/joint_acc': joint_accelerations,
        'motor/joint_tau_est': joint_tau_est,
        # 'foot_forces': [msg['foot_force'][idx] for idx in [1, 0, 3, 2]],  # Reorder foot forces
        # 'foot_force_est': [msg['foot_force_est'][idx] for idx in [1, 0, 3, 2]],  # Reorder estimated foot forces
        'imu/quat': imu_state.quaternion,
        'imu/rpy': imu_state.rpy,
        'imu/gyroscope': imu_state.gyroscope,
        'imu/accelerometer': imu_state.accelerometer
    }
    return states
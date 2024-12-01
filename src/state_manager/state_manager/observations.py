from typing import Callable, Dict, Any
import numpy as np
import time

from scipy.spatial.transform import Rotation as R

from utils.helpers import reorder_robot_states
from utils.math import quat_rotate_inverse

GRAVITY_VEC_W = np.array([0, 0, -9.81])  # Standard gravity in the Z direction

def joint_pos(states: Dict[str, Any]) -> np.ndarray:
    """
    The joint positions of the asset.
    
    :param states: State dictionary
    :return: Joint positions
    """
    joint_pos = reorder_robot_states(states['joint_pos'][:12], 
                                                origin_order=['FL', 'FR', 'RL', 'RR'], 
                                                target_order=['FR', 'FL', 'RR', 'RL'])
    return joint_pos

def joint_pos_rel(states: Dict[str, Any], default_joint_pos: np.ndarray) -> np.ndarray:
    """
    Compute relative joint positions.
    
    :param states: State dictionary
    :param default_joint_pos: Default joint positions
    :return: Relative joint positions
    """
    joint_pos = reorder_robot_states(states['joint_pos'][:12], 
                                            origin_order=['FL', 'FR', 'RL', 'RR'], 
                                            target_order=['FR', 'FL', 'RR', 'RL'])
    default_joint_pos = reorder_robot_states(default_joint_pos, 
                                            origin_order=['FL', 'FR', 'RL', 'RR'], 
                                            target_order=['FR', 'FL', 'RR', 'RL'])
    return np.array(joint_pos) - np.array(default_joint_pos)

def joint_vel(states: Dict[str, Any]) -> np.ndarray:
    """
    The joint positions of the asset.
    
    :param states: State dictionary
    :return: Joint velocities
    """
    joint_vel = reorder_robot_states(states['joint_vel'][:12], 
                                                origin_order=['FL', 'FR', 'RL', 'RR'], 
                                                target_order=['FR', 'FL', 'RR', 'RL'])
    return joint_vel

def lin_vel_b(states: Dict[str, Any]) -> np.ndarray:
    """
    The linear velocity of the asset in base frame.
    
    :param states: State dictionary
    :return: Linear velocity in the base frame
    """
    
    # Convert quaternion to a rotation matrix
    rotation_matrix = R.from_quat(states["base_quat"]).as_matrix()

    # Transform linear velocity to base frame
    lin_vel_base = np.dot(rotation_matrix.T, states["lin_vel_w"])  # Transpose is inverse for rotation matrices
    return lin_vel_base
    
    
    
def ang_vel_b(states: Dict[str, Any]) -> np.ndarray:
    """
    The angular velocity of the asset in base frame.
    
    :param states: State dictionary
    :return: Angular velocity in the base frame
    """
    
    # Convert quaternion to a rotation matrix
    rotation_matrix = R.from_quat(states["base_quat"]).as_matrix()
    
    # Transform angular velocity to base frame
    ang_vel_base = np.dot(rotation_matrix.T, states["ang_vel_w"])
    return ang_vel_base

def projected_gravity_b(states: Dict[str, Any]) -> np.ndarray:
    """
    The projected gravity vector.
    
    :param states: State dictionary
    :return: Angular velocity in the base frame
    """
    
    return quat_rotate_inverse(states["base_quat"], GRAVITY_VEC_W)
    

def last_action(states: Dict[str, Any], last_action: Callable) -> np.ndarray:
    """
    The previous action from the policy. We use a callable (lambda) to fetch the latest value from the controller class.
    
    :param states: State dictionary
    :return: Angular velocity in the base frame
    """
    return last_action()


def starting_time(states: Dict[str, Any]):
    return time.time()
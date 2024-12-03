from typing import Callable, Dict, Any
import numpy as np
import torch
import time

from scipy.spatial.transform import Rotation as R

from utils.helpers import reorder_robot_states
from utils.math import quat_rotate_inverse

GRAVITY_DIR = torch.tensor([0, 0, -1.0])  # Standard gravity in the Z direction

def joint_pos(states: Dict[str, Any]) -> np.ndarray:
    """
    The joint positions of the asset.
    
    :param states: State dictionary
    :return: Joint positions
    """
    joint_pos = reorder_robot_states(states['joint_pos'], 
                                    origin_order=['FL', 'FR', 'RL', 'RR'], 
                                    target_order=['FR', 'FL', 'RR', 'RL'])
    return joint_pos

def joint_pos_rel(states: Dict[str, Any], default_joint_pos: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """
    Compute relative joint positions.
    
    :param states: State dictionary
    :param default_joint_pos: Default joint positions
    :return: Relative joint positions
    """
    return states['joint_pos'][mapping] - default_joint_pos

def joint_vel(states: Dict[str, Any], mapping: np.ndarray) -> np.ndarray:
    """
    The joint positions of the asset.
    
    :param states: State dictionary
    :return: Joint velocities
    """
    return states['joint_vel'][mapping]

def lin_vel_b(states: Dict[str, Any]) -> np.ndarray:
    """
    The linear velocity of the asset in base frame.
    
    :param states: State dictionary
    :return: Linear velocity in the base frame
    """
    
    return states['lin_vel_b']
    
def ang_vel_b(states: Dict[str, Any]) -> np.ndarray:
    """
    The angular velocity of the asset in base frame.
    
    :param states: State dictionary
    :return: Angular velocity in the base frame
    """
   
    return states['gyroscope']

def projected_gravity_b(states: Dict[str, Any], logger=None) -> np.ndarray:
    """
    The projected gravity vector.
    
    :param states: State dictionary
    :return: Angular velocity in the base frame
    """

    return quat_rotate_inverse(torch.tensor([states["base_quat"]]).squeeze(0), GRAVITY_DIR)
    
def last_action(states: Dict[str, Any], last_action: Callable) -> np.ndarray:
    """
    The previous action from the policy. We use a callable (lambda) to fetch the latest value from the controller class.
    
    :param states: State dictionary
    :return: Angular velocity in the base frame
    """
    return last_action()

def velocity_commands(states: Dict[str, Any]) -> np.ndarray:
    return [0.2, 0, 0]

def starting_time(states: Dict[str, Any]):
    return time.time()
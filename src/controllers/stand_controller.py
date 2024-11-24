#######################################
# TODO: MAKE THIS CLASS ROBOT-AGNOSTIC#
#######################################

import numpy as np
from controllers.controller_base import ControllerBase

STAND_UP_JOINT_POS = np.array([
    0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,
    0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763
], dtype=float)

STAND_DOWN_JOINT_POS = np.array([
    0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375,
    0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
], dtype=float)

class IdleController(ControllerBase):
    """
    Used to set zero commands to the motor. This is particularly useful when exiting the controller to reset the torques to 0.
    """
    def compute_command(self, state, desired_goal):
        cmd = {}
        for i in range(12):
            cmd[f'motor_{i}'] = {
                'q': 0.0,
                'kp': 0.0,
                'dq': 0.0,
                'kd': 0.0,
                'tau': 0.0,
            }
        return cmd


class StandUpController(ControllerBase):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants 
    """
    def compute_command(self, state, desired_goal=None, **kwargs=None):
        elapsed_time = state["elapsed_time"]
        phase = np.tanh(elapsed_time / 1.2)
        cmd = {}
        for i in range(12):
            cmd[f'motor_{i}'] = {
                'q': phase * STAND_UP_JOINT_POS[i] + (1 - phase) * STAND_DOWN_JOINT_POS[i],
                'kp': phase * 50.0 + (1 - phase) * 20.0,
                'dq': 0.0,
                'kd': 3.5,
                'tau': 0.0,
            }
        return cmd


class StandDownController(ControllerBase):
    def compute_command(self, state, desired_goal=None):
        elapsed_time = state["elapsed_time"]
        phase = np.tanh(elapsed_time / 1.2)
        cmd = {}
        for i in range(12):
            cmd[f'motor_{i}'] = {
                'q': phase * STAND_DOWN_JOINT_POS[i] + (1 - phase) * STAND_UP_JOINT_POS[i],
                'kp': phase * 50.0 + (1 - phase) * 20.0,
                'dq': 0.0,
                'kd': 3.5,
                'tau': 0.0,
            }
        return cmd


class StayDownController(ControllerBase):
    def compute_command(self, state, desired_goal):
        elapsed_time = state["elapsed_time"]
        phase = np.tanh(elapsed_time / 1.2)
        cmd = {}
        for i in range(12):
            cmd[f'motor_{i}'] = {
                'q': STAND_DOWN_JOINT_POS[i],
                'kp': 20.0,
                'dq': 0.0,
                'kd': 3.5,
                'tau': 0.0,
            }
        return cmd
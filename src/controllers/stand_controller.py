import numpy as np
from controllers.controller_base import ControllerBase

class IdleController(ControllerBase):
    """
    Used to set zero commands to the motor. This is particularly useful when exiting the controller to reset the torques to 0.
    """

    def compute_torques(self, state, desired_goal):
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
    to the stand up joint positions which are constants.
    """
    def __init__(self, robot_config):
        self.stand_up_joint_pos = robot_config["STAND_UP_JOINT_POS"]
        self.stand_down_joint_pos = robot_config["STAND_DOWN_JOINT_POS"]

    def compute_torques(self, state, desired_goal=None):
        elapsed_time = state["elapsed_time"]
        phase = np.tanh(elapsed_time / 1.2)
        cmd = {}
        for i in range(12):
            cmd[f'motor_{i}'] = {
                'q': phase * self.stand_up_joint_pos[i] + (1 - phase) * self.stand_down_joint_pos[i],
                'kp': phase * 50.0 + (1 - phase) * 20.0,
                'dq': 0.0,
                'kd': 3.5,
                'tau': 0.0,
            }
        return cmd


class StandDownController(ControllerBase):
    """
    The Stand Down Controller is used to sit down from the nominal position. It is an interpolation from the stand up joint positions
    to the stand down joint positions which are constants.
    """
    def __init__(self, robot_config):
        self.stand_up_joint_pos = robot_config["STAND_UP_JOINT_POS"]
        self.stand_down_joint_pos = robot_config["STAND_DOWN_JOINT_POS"]

    def compute_torques(self, state, desired_goal=None):
        elapsed_time = state["elapsed_time"]
        phase = np.tanh(elapsed_time / 1.2)
        cmd = {}
        for i in range(12):
            cmd[f'motor_{i}'] = {
                'q': phase * self.stand_down_joint_pos[i] + (1 - phase) * self.stand_up_joint_pos[i],
                'kp': phase * 50.0 + (1 - phase) * 20.0,
                'dq': 0.0,
                'kd': 3.5,
                'tau': 0.0,
            }
        return cmd


class StayDownController(ControllerBase):
    """
    The Stay Down Controller is used to stay down close the ground, to prepare to get up.
    """
    def __init__(self, robot_config):
        self.stand_down_joint_pos = robot_config["STAND_DOWN_JOINT_POS"]

    def compute_torques(self, state, desired_goal):
        elapsed_time = state["elapsed_time"]
        phase = np.tanh(elapsed_time / 1.2)
        cmd = {}
        for i in range(12):
            cmd[f'motor_{i}'] = {
                'q': self.stand_down_joint_pos[i],
                'kp': 20.0,
                'dq': 0.0,
                'kd': 3.5,
                'tau': 0.0,
            }
        return cmd


class StanceController(ControllerBase):
    """
    The Stance Controller is used to stay in stance. Used to prepare to go to rest from other controllers.
    """
    def __init__(self, robot_config):
        self.stand_up_joint_pos = robot_config["STAND_UP_JOINT_POS"]

    def compute_torques(self, state, desired_goal):
        elapsed_time = state["elapsed_time"]
        phase = np.tanh(elapsed_time / 1.2)
        cmd = {}
        for i in range(12):
            cmd[f'motor_{i}'] = {
                'q': self.stand_up_joint_pos[i],
                'kp': 50.0,
                'dq': 0.0,
                'kd': 3.5,
                'tau': 0.0,
            }
        return cmd
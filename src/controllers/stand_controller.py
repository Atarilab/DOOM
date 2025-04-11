from typing import Any, Dict

import numpy as np
from controllers.controller_base import ControllerBase
from state_manager.obs_manager import ObsTerm
from state_manager.observations import starting_time
from utils.mj_pin_wrapper.pin_robot import PinQuadRobotWrapper
from utils.mj_wrapper import MjQuadRobotWrapper


class IdleController(ControllerBase):
    """
    Used to set zero commands to the motor. This is particularly useful when exiting the controller to reset the torques to 0.
    """

    def __init__(self, pin_model_wrapper, mj_model_wrapper, configs):
        super().__init__(pin_model_wrapper, mj_model_wrapper, configs=configs)

    def register_observations(self):
        """
        Register observations for this controller.
        """
        pass

    def compute_torques(self, state, desired_goal):

        # When Init Controller is called, set the init frame
        self.mj_model_wrapper.set_initial_world_frame(state, caller=self.__class__.__name__)

        super().compute_torques(state, desired_goal=desired_goal)

        cmd = {}
        for i in range(12):
            cmd[f"motor_{i}"] = {
                "q": 0.0,
                "kp": 0.0,
                "dq": 0.0,
                "kd": 0.0,
                "tau": 0.0,
            }
        return cmd


class StandUpController(ControllerBase):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(
        self, pin_model_wrapper: PinQuadRobotWrapper, mj_model_wrapper: MjQuadRobotWrapper, configs: Dict[str, Any]
    ):
        super().__init__(pin_model_wrapper=pin_model_wrapper, mj_model_wrapper=mj_model_wrapper, configs=configs)

        self.name = "StandUpController"
        self.stand_up_joint_pos = configs["robot_config"]["stand_up_joint_pos"]
        self.stand_down_joint_pos = configs["robot_config"]["stand_down_joint_pos"]
        self.start_time = 0.0

    def register_observations(self):
        """
        Register observations for this controller.
        """

        # Register observations using the mode-specific obs_manager
        self.obs_manager.register(
            "time",
            ObsTerm(
                starting_time,
            ),
        )

    def compute_torques(self, state, desired_goal=None):
        super().compute_torques(state, desired_goal=desired_goal)
        obs = self.obs_manager.compute(state)

        time = obs["time"] - self.start_time
        phase = np.tanh(time / 1.2)

        cmd = {}
        for i in range(12):
            cmd[f"motor_{i}"] = {
                "q": phase * self.stand_up_joint_pos[i] + (1 - phase) * self.stand_down_joint_pos[i],
                "kp": phase * 50.0 + (1 - phase) * 20.0,
                "dq": 0.0,
                "kd": 3.5,
                "tau": 0.0,
            }
        return cmd


class StandDownController(ControllerBase):
    """
    The Stand Down Controller is used to sit down from the nominal position. It is an interpolation from the stand up joint positions
    to the stand down joint positions which are constants.
    """

    def __init__(self, pin_model_wrapper, mj_model_wrapper, configs):
        super().__init__(pin_model_wrapper, mj_model_wrapper, configs=configs)

        self.stand_up_joint_pos = configs["robot_config"]["stand_up_joint_pos"]
        self.stand_down_joint_pos = configs["robot_config"]["stand_down_joint_pos"]
        self.start_time = 0.0

    def register_observations(self):
        """
        Register observations for this controller.
        """

        # Register observations using the mode-specific obs_manager
        self.obs_manager.register(
            "time",
            ObsTerm(
                starting_time,
            ),
        )

    def compute_torques(self, state, desired_goal=None):
        super().compute_torques(state, desired_goal=desired_goal)

        obs = self.obs_manager.compute(state)

        time = obs["time"] - self.start_time
        phase = np.tanh(time / 1.2)
        cmd = {}
        for i in range(12):
            cmd[f"motor_{i}"] = {
                "q": phase * self.stand_down_joint_pos[i] + (1 - phase) * self.stand_up_joint_pos[i],
                "kp": phase * 50.0 + (1 - phase) * 20.0,
                "dq": 0.0,
                "kd": 3.5,
                "tau": 0.0,
            }
        return cmd


class StayDownController(ControllerBase):
    """
    The Stay Down Controller is used to stay down close the ground, to prepare to get up.
    """

    def __init__(self, pin_model_wrapper, mj_model_wrapper, configs):
        super().__init__(pin_model_wrapper, mj_model_wrapper, configs=configs)

        self.stand_down_joint_pos = configs["robot_config"]["stand_down_joint_pos"]
        self.start_time = 0.0

    def register_observations(self):
        """
        Register observations for this controller.
        """

        # Register observations using the mode-specific obs_manager
        self.obs_manager.register(
            "time",
            ObsTerm(
                starting_time,
            ),
        )

    def compute_torques(self, state, desired_goal):
        super().compute_torques(state, desired_goal=desired_goal)

        cmd = {}
        for i in range(12):
            cmd[f"motor_{i}"] = {
                "q": self.stand_down_joint_pos[i],
                "kp": 15.0,
                "dq": 0.0,
                "kd": 3.5,
                "tau": 0.0,
            }
        return cmd


class StanceController(ControllerBase):
    """
    The Stance Controller is used to stay in stance. Used to prepare to go to rest from other controllers.
    """

    def __init__(self, pin_model_wrapper, mj_model_wrapper, configs):
        super().__init__(pin_model_wrapper, mj_model_wrapper=mj_model_wrapper, configs=configs)

        self.stand_up_joint_pos = configs["robot_config"]["stand_up_joint_pos"]
        self.start_time = 0.0

    def register_observations(self):
        """
        Register observations for this controller.
        """

        # Register observations using the mode-specific obs_manager
        self.obs_manager.register(
            "time",
            ObsTerm(
                starting_time,
            ),
        )

    def compute_torques(self, state, desired_goal):
        super().compute_torques(state, desired_goal=desired_goal)
        cmd = {}
        for i in range(12):
            cmd[f"motor_{i}"] = {
                "q": self.stand_up_joint_pos[i],
                "kp": 15.0,
                "dq": 0.0,
                "kd": 3.5,
                "tau": 0.0,
            }
        return cmd

from typing import Any, Dict, TYPE_CHECKING

import numpy as np
from controllers.controller_base import ControllerBase
from state_manager.obs_manager import ObsTerm
from state_manager.observations import starting_time
import time

if TYPE_CHECKING:
    from robots.robot_base import RobotBase

class IdleController(ControllerBase):
    """
    Used to set zero commands to the motor. This is particularly useful when exiting the controller to reset the torques to 0.
    """

    def __init__(self, robot, configs):
        super().__init__(robot, configs=configs)

    def register_observations(self):
        """
        Register observations for this controller.
        """
        pass
    
    def set_mode(self):
        pass

    def compute_torques(self, state, desired_goal):

        # When Init Controller is called, set the init frame
        if self.robot.mj_model is not None:
            self.robot.mj_model.set_initial_world_frame(state, caller=self.__class__.__name__)

        super().compute_torques(state, desired_goal=desired_goal)

        cmd = {}
        for i in range(self.robot.num_joints):
            cmd[f"motor_{i}"] = {
                "q": 0.0,
                "kp": 0.0,
                "dq": 0.0,
                "kd": 3.0,
                "tau": 0.0,
            }
        return cmd


class Go2StandUpController(ControllerBase):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any]):
        super().__init__(robot=robot, configs=configs)

        self.name = "Go2StandUpController"
        self.stand_up_joint_pos = robot.stand_up_joint_pos
        self.stand_down_joint_pos = robot.stand_down_joint_pos
        self.start_time = 0.0

    def set_mode(self):
        self.start_time = time.time()

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
        for i in range(self.robot.num_joints):
            cmd[f"motor_{i}"] = {
                "q": phase * self.stand_up_joint_pos[i] + (1 - phase) * self.stand_down_joint_pos[i],
                "kp": phase * 50.0 + (1 - phase) * 20.0,
                "dq": 0.0,
                "kd": 3.5,
                "tau": 0.0,
            }
        return cmd


class Go2StandDownController(ControllerBase):
    """
    The Stand Down Controller is used to sit down from the nominal position. It is an interpolation from the stand up joint positions
    to the stand down joint positions which are constants.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any]):
        super().__init__(robot=robot, configs=configs)

        self.stand_up_joint_pos = robot.stand_up_joint_pos
        self.stand_down_joint_pos = robot.stand_down_joint_pos
        self.start_time = 0.0

    def set_mode(self):
        self.start_time = time.time()

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
        for i in range(self.robot.num_joints):
            cmd[f"motor_{i}"] = {
                "q": phase * self.stand_down_joint_pos[i] + (1 - phase) * self.stand_up_joint_pos[i],
                "kp": phase * 50.0 + (1 - phase) * 20.0,
                "dq": 0.0,
                "kd": 3.5,
                "tau": 0.0,
            }
        return cmd


class Go2StayDownController(ControllerBase):
    """
    The Stay Down Controller is used to stay down close the ground, to prepare to get up.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any]):
        super().__init__(robot=robot, configs=configs)

        self.stand_down_joint_pos = robot.stand_down_joint_pos
        self.start_time = 0.0

    def set_mode(self):
        self.start_time = time.time()

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
        for i in range(self.robot.num_joints):
            cmd[f"motor_{i}"] = {
                "q": self.stand_down_joint_pos[i],
                "kp": 15.0,
                "dq": 0.0,
                "kd": 3.5,
                "tau": 0.0,
            }
        return cmd


class Go2StanceController(ControllerBase):
    """
    The Stance Controller is used to stay in stance. Used to prepare to go to rest from other controllers.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any]):
        super().__init__(robot=robot, configs=configs)

        self.stand_up_joint_pos = robot.stand_up_joint_pos
        self.start_time = 0.0

    def set_mode(self):
        self.start_time = time.time()

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
        for i in range(self.robot.num_joints):
            cmd[f"motor_{i}"] = {
                "q": self.stand_up_joint_pos[i],
                "kp": 15.0,
                "dq": 0.0,
                "kd": 3.5,
                "tau": 0.0,
            }
        return cmd

class G1StandUpController(ControllerBase):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any]):
        super().__init__(robot=robot, configs=configs)

        self.name = "G1StandUpController"
        self.total_time = 2.0 # 2 seconds
        self.num_steps = int(self.total_time / self.control_dt)
        
        self.leg_joint2motor_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.leg_kps = [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40]
        self.leg_kds = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2]
        self.default_angles = [-0.1,  0.0,  0.0,  0.3, -0.2, 0.0, 
                  -0.1,  0.0,  0.0,  0.3, -0.2, 0.0]
        self.arm_waist_joint2motor_idx = [12, 13, 14, 
                            15, 16, 17, 18, 19, 20, 21, 
                            22, 23, 24, 25, 26, 27, 28]
        
        self.arm_waist_kps = [300, 300, 300,
                100, 100, 50, 50, 20, 20, 20,
                100, 100, 50, 50, 20, 20, 20]

        self.arm_waist_kds = [3, 3, 3, 
                2, 2, 2, 2, 1, 1, 1,
                2, 2, 2, 2, 1, 1, 1]
        
        self.arm_waist_target = [ 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0]

        self.dof_idx = self.leg_joint2motor_idx + self.arm_waist_joint2motor_idx
        self.kps = self.leg_kps + self.arm_waist_kps
        self.kds = self.leg_kds + self.arm_waist_kds
        
        self.default_pos = np.concatenate((self.default_angles, self.arm_waist_target), axis=0)
        self.dof_size = len(self.dof_idx)
        self.init_dof_pos = np.zeros(self.dof_size, dtype=np.float32)
        
        self.step_counter = 0
        
    def set_mode(self):
        self.start_time = time.time()
        self.step_counter = 0
        
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
        # record the current pos
        for i in range(self.dof_size):
            self.init_dof_pos[i] = state["robot/joint_pos"][self.dof_idx[i]]
            
        alpha = self.step_counter / self.num_steps

        cmd = {}
        for i in range(self.dof_size):
            cmd[f"motor_{self.dof_idx[i]}"] = {
                "q": alpha * self.default_pos[i] + (1 - alpha) * self.init_dof_pos[i],
                "kp": self.kps[i],
                "dq": 0.0,
                "kd": self.kds[i],
                "tau": 0.0,
            }
        self.step_counter = min(self.step_counter + 1, self.num_steps)
        return cmd
    

class G1StayUpController(ControllerBase):
    """
    The Stay Up Controller is used to stay up close the ground, to prepare to get up.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any]):
        super().__init__(robot=robot, configs=configs)

        self.name = "G1StayUpController"
        self.total_time = 2.0 # 2 seconds
        self.num_steps = int(self.total_time / self.control_dt)
        
        self.leg_joint2motor_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.leg_kps = [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40]
        self.leg_kds = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2]
        self.default_angles = [-0.1,  0.0,  0.0,  0.3, -0.2, 0.0, 
                  -0.1,  0.0,  0.0,  0.3, -0.2, 0.0]
        self.arm_waist_joint2motor_idx = [12, 13, 14, 
                            15, 16, 17, 18, 19, 20, 21, 
                            22, 23, 24, 25, 26, 27, 28]
        
        self.arm_waist_kps = [300, 300, 300,
                100, 100, 50, 50, 20, 20, 20,
                100, 100, 50, 50, 20, 20, 20]

        self.arm_waist_kds = [3, 3, 3, 
                2, 2, 2, 2, 1, 1, 1,
                2, 2, 2, 2, 1, 1, 1]
        
        self.arm_waist_target = [ 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0]

        self.dof_idx = self.leg_joint2motor_idx + self.arm_waist_joint2motor_idx
        self.kps = self.leg_kps + self.arm_waist_kps
        self.kds = self.leg_kds + self.arm_waist_kds
        
        self.default_pos = np.concatenate((self.default_angles, self.arm_waist_target), axis=0)
        self.dof_size = len(self.dof_idx)
        self.init_dof_pos = np.zeros(self.dof_size, dtype=np.float32)
        
    def set_mode(self):
        self.start_time = time.time()
    
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

        cmd = {}
        for i in range(len(self.leg_joint2motor_idx)):
            motor_idx = self.leg_joint2motor_idx[i]
            cmd[f"motor_{motor_idx}"] = {
                "q": self.default_angles[i],
                "kp": self.kps[i],
                "dq": 0.0,
                "kd": self.kds[i],
                "tau": 0.0,
            }
        
        for i in range(len(self.arm_waist_joint2motor_idx)):
            motor_idx = self.arm_waist_joint2motor_idx[i]
            cmd[f"motor_{motor_idx}"] = {
                "q": self.arm_waist_target[i],
                "kp": self.arm_waist_kps[i],
                "dq": 0.0,
                "kd": self.arm_waist_kds[i],
                "tau": 0.0,
            }

        return cmd
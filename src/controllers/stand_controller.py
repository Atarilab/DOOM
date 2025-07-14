import time
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from controllers.controller_base import ControllerBase
from state_manager.obs_manager import ObsTerm

if TYPE_CHECKING:
    from robots.robot_base import RobotBase


class ZeroTorqueController(ControllerBase):
    """
    Used to set zero commands to the motor. This is particularly useful when exiting the controller to reset the torques to 0.
    """

    def __init__(self, robot, configs):
        super().__init__(robot, configs=configs)

    def register_observations(self):
        """
        Register observations for this controller.
        """

    def set_mode(self):
        pass

    def compute_lowlevelcmd(self, state):

        # When Init Controller is called, set the init frame
        if self.robot.mj_model is not None:
            self.robot.mj_model.set_initial_world_frame(state, caller=self.__class__.__name__)

        super().compute_lowlevelcmd(state)
        cmd = {}
        for i in range(self.robot.num_joints):
            cmd[f"motor_{i}"] = {
                "q": 0.0,
                "kp": 0.0,
                "dq": 0.0,
                "kd": 0.0,
                "tau": 0.0,
            }
        return cmd


class DampingController(ControllerBase):
    """
    Used to set damping commands to the motor. This is particularly useful for safely stopping experiments.
    """

    def __init__(self, robot, configs):
        super().__init__(robot, configs=configs)

    def register_observations(self):
        """
        Register observations for this controller.
        """

    def set_mode(self):
        pass

    def compute_lowlevelcmd(self, state):
        super().compute_lowlevelcmd(state)
        cmd = {}
        for i in range(self.robot.num_joints):
            cmd[f"motor_{i}"] = {
                "q": 0.0,
                "kp": 0.0,
                "dq": 0.0,
                "kd": 2.0,
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
        from state_manager.observations import current_time

        # Register observations using the mode-specific obs_manager
        self.obs_manager.register(
            "time",
            ObsTerm(
                current_time,
            ),
        )

    def compute_lowlevelcmd(self, state):
        super().compute_lowlevelcmd(state)
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
        from state_manager.observations import current_time

        # Register observations using the mode-specific obs_manager
        self.obs_manager.register(
            "time",
            ObsTerm(
                current_time,
            ),
        )

    def compute_lowlevelcmd(self, state):
        super().compute_lowlevelcmd(state)

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
        from state_manager.observations import current_time

        # Register observations using the mode-specific obs_manager
        self.obs_manager.register(
            "time",
            ObsTerm(
                current_time,
            ),
        )

    def compute_lowlevelcmd(self, state):
        super().compute_lowlevelcmd(state)

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
        from state_manager.observations import current_time

        # Register observations using the mode-specific obs_manager
        self.obs_manager.register(
            "time",
            ObsTerm(
                current_time,
            ),
        )

    def compute_lowlevelcmd(self, state):
        super().compute_lowlevelcmd(state)
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
        self.total_time = 5.0  # 2 seconds
        self.num_steps = int(self.total_time / self.control_dt)

        self.leg_joint2motor_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.leg_kps = [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40]
        self.leg_kds = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2]
        self.default_angles = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0]
        
        self.arm_waist_joint2motor_idx = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        self.arm_waist_kps = [300, 300, 300, 100, 100, 50, 50, 20, 20, 20, 100, 100, 50, 50, 20, 20, 20]
        self.arm_waist_kds = [3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1]
        self.arm_waist_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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
        from state_manager.observations import current_time

        # Register observations using the mode-specific obs_manager
        self.obs_manager.register(
            "time",
            ObsTerm(
                current_time,
            ),
        )

    def compute_lowlevelcmd(self, state):
        super().compute_lowlevelcmd(state)
        self.obs_manager.compute(state)
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

        self.leg_joint2motor_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.leg_kps = [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40]
        self.leg_kds = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2]
        self.default_angles = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0]
        
        self.arm_waist_joint2motor_idx = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        self.arm_waist_kps = [300, 300, 300, 100, 100, 50, 50, 20, 20, 20, 100, 100, 50, 50, 20, 20, 20]
        self.arm_waist_kds = [3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1]
        self.arm_waist_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


    def set_mode(self):
        self.start_time = time.time()

    def register_observations(self):
        """
        Register observations for this controller.
        """
        from state_manager.observations import current_time

        # Register observations using the mode-specific obs_manager
        self.obs_manager.register(
            "time",
            ObsTerm(
                current_time,
            ),
        )

    def compute_lowlevelcmd(self, state):
        super().compute_lowlevelcmd(state)

        cmd = {}
        for i in range(len(self.leg_joint2motor_idx)):
            motor_idx = self.leg_joint2motor_idx[i]
            cmd[f"motor_{motor_idx}"] = {
                "q": self.default_angles[i],
                "kp": self.leg_kps[i],
                "dq": 0.0,
                "kd": self.leg_kds[i],
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


class G1LowLevelController(ControllerBase):
    """
    The Low Level Controller is used to control the robot at the low level.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any]):
        super().__init__(robot=robot, configs=configs)

        self.name = "G1LowLevelController"
        self.total_time = 3.0  # 2 seconds
        self.num_steps = int(self.total_time / self.control_dt)

        self.leg_joint2motor_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.leg_kps = [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40]
        self.leg_kds = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2]
        self.default_angles = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0]
        self.arm_waist_joint2motor_idx = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

        self.arm_waist_kps = [300, 300, 300, 100, 100, 50, 50, 20, 20, 20, 100, 100, 50, 50, 20, 20, 20]
        self.arm_waist_kds = [3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1]
        self.arm_waist_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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
        from state_manager.observations import current_time

        # Register observations using the mode-specific obs_manager
        self.obs_manager.register(
            "time",
            ObsTerm(
                current_time,
            ),
        )

    def compute_lowlevelcmd(self, state):
        super().compute_lowlevelcmd(state)
        obs = self.obs_manager.compute(state)
        time = obs["time"] - self.start_time

        Kp = [
            60,
            60,
            60,
            100,
            40,
            40,  # legs
            60,
            60,
            60,
            100,
            40,
            40,  # legs
            60,
            40,
            40,  # waist
            40,
            40,
            40,
            40,
            40,
            40,
            40,  # arms
            40,
            40,
            40,
            40,
            40,
            40,
            40,  # arms
        ]

        Kd = [
            1,
            1,
            1,
            2,
            1,
            1,  # legs
            1,
            1,
            1,
            2,
            1,
            1,  # legs
            1,
            1,
            1,  # waist
            1,
            1,
            1,
            1,
            1,
            1,
            1,  # arms
            1,
            1,
            1,
            1,
            1,
            1,
            1,  # arms
        ]

        G1_NUM_MOTOR = 29

        class G1JointIndex:
            LeftHipPitch = 0
            LeftHipRoll = 1
            LeftHipYaw = 2
            LeftKnee = 3
            LeftAnklePitch = 4
            LeftAnkleB = 4
            LeftAnkleRoll = 5
            LeftAnkleA = 5
            RightHipPitch = 6
            RightHipRoll = 7
            RightHipYaw = 8
            RightKnee = 9
            RightAnklePitch = 10
            RightAnkleB = 10
            RightAnkleRoll = 11
            RightAnkleA = 11
            WaistYaw = 12
            WaistRoll = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
            WaistA = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
            WaistPitch = 14  # NOTE: INVALID for g1 23dof/29dof with waist locked
            WaistB = 14  # NOTE: INVALID for g1 23dof/29dof with waist locked
            LeftShoulderPitch = 15
            LeftShoulderRoll = 16
            LeftShoulderYaw = 17
            LeftElbow = 18
            LeftWristRoll = 19
            LeftWristPitch = 20  # NOTE: INVALID for g1 23dof
            LeftWristYaw = 21  # NOTE: INVALID for g1 23dof
            RightShoulderPitch = 22
            RightShoulderRoll = 23
            RightShoulderYaw = 24
            RightElbow = 25
            RightWristRoll = 26
            RightWristPitch = 27  # NOTE: INVALID for g1 23dof
            RightWristYaw = 28  # NOTE: INVALID for g1 23dof

        class Mode:
            PR = 0  # Series Control for Pitch/Roll Joints
            AB = 1  # Parallel Control for A/B Joints

        cmd = {f"motor_{i}": {"q": 0.0, "kp": 0.0, "dq": 0.0, "kd": 0.0, "tau": 0.0} for i in range(G1_NUM_MOTOR)}
        cmd["mode_pr"] = Mode.PR
        if time < self.total_time:
            # [Stage 1]: set robot to zero posture
            for i in range(G1_NUM_MOTOR):
                ratio = np.clip(time / self.total_time, 0.0, 1.0)
                cmd["mode_pr"] = Mode.PR
                cmd[f"motor_{i}"] = {
                    "q": (1.0 - ratio) * state["robot/joint_pos"][self.dof_idx[i]],
                    "kp": Kp[i],
                    "dq": 0.0,
                    "kd": Kd[i],
                    "tau": 0.0,
                    "mode": 1,
                }

        elif time < self.total_time * 2:
            # [Stage 2]: swing ankle using PR mode
            max_P = np.pi * 30.0 / 180.0
            max_R = np.pi * 10.0 / 180.0
            t = time - self.total_time
            L_P_des = max_P * np.sin(2.0 * np.pi * t)
            L_R_des = max_R * np.sin(2.0 * np.pi * t)
            R_P_des = max_P * np.sin(2.0 * np.pi * t)
            R_R_des = -max_R * np.sin(2.0 * np.pi * t)

            cmd["mode_pr"] = Mode.PR
            cmd[f"motor_{G1JointIndex.LeftAnklePitch}"] = {
                "q": L_P_des,
                "kp": Kp[G1JointIndex.LeftAnklePitch],
                "dq": 0.0,
                "kd": Kd[G1JointIndex.LeftAnklePitch],
                "tau": 0.0,
            }
            cmd[f"motor_{G1JointIndex.LeftAnkleRoll}"] = {
                "q": L_R_des,
                "kp": Kp[G1JointIndex.LeftAnkleRoll],
                "dq": 0.0,
                "kd": Kd[G1JointIndex.LeftAnkleRoll],
                "tau": 0.0,
            }
            cmd[f"motor_{G1JointIndex.RightAnklePitch}"] = {
                "q": R_P_des,
                "kp": Kp[G1JointIndex.RightAnklePitch],
                "dq": 0.0,
                "kd": Kd[G1JointIndex.RightAnklePitch],
                "tau": 0.0,
            }
            cmd[f"motor_{G1JointIndex.RightAnkleRoll}"] = {
                "q": R_R_des,
                "kp": Kp[G1JointIndex.RightAnkleRoll],
                "dq": 0.0,
                "kd": Kd[G1JointIndex.RightAnkleRoll],
                "tau": 0.0,
            }

        else:
            # [Stage 3]: swing ankle using AB mode
            max_A = np.pi * 30.0 / 180.0
            max_B = np.pi * 10.0 / 180.0
            t = time - self.total_time * 2
            L_A_des = max_A * np.sin(2.0 * np.pi * t)
            L_B_des = max_B * np.sin(2.0 * np.pi * t + np.pi)
            R_A_des = -max_A * np.sin(2.0 * np.pi * t)
            R_B_des = -max_B * np.sin(2.0 * np.pi * t + np.pi)

            cmd[f"motor_{G1JointIndex.LeftAnkleA}"] = {
                "q": L_A_des,
                "kp": Kp[G1JointIndex.LeftAnkleA],
                "dq": 0.0,
                "kd": Kd[G1JointIndex.LeftAnkleA],
                "tau": 0.0,
            }
            cmd[f"motor_{G1JointIndex.LeftAnkleB}"] = {
                "q": L_B_des,
                "kp": Kp[G1JointIndex.LeftAnkleB],
                "dq": 0.0,
                "kd": Kd[G1JointIndex.LeftAnkleB],
                "tau": 0.0,
            }
            cmd[f"motor_{G1JointIndex.RightAnkleA}"] = {
                "q": R_A_des,
                "kp": Kp[G1JointIndex.RightAnkleA],
                "dq": 0.0,
                "kd": Kd[G1JointIndex.RightAnkleA],
                "tau": 0.0,
            }
            cmd[f"motor_{G1JointIndex.RightAnkleB}"] = {
                "q": R_B_des,
                "kp": Kp[G1JointIndex.RightAnkleB],
                "dq": 0.0,
                "kd": Kd[G1JointIndex.RightAnkleB],
                "tau": 0.0,
            }

            max_WristYaw = np.pi * 30.0 / 180.0
            L_WristYaw_des = max_WristYaw * np.sin(2.0 * np.pi * t)
            R_WristYaw_des = max_WristYaw * np.sin(2.0 * np.pi * t)
            cmd[f"motor_{G1JointIndex.LeftWristRoll}"] = {
                "q": L_WristYaw_des,
                "kp": Kp[G1JointIndex.LeftWristRoll],
                "dq": 0.0,
                "kd": Kd[G1JointIndex.LeftWristRoll],
                "tau": 0.0,
            }
            cmd[f"motor_{G1JointIndex.RightWristRoll}"] = {
                "q": R_WristYaw_des,
                "kp": Kp[G1JointIndex.RightWristRoll],
                "dq": 0.0,
                "kd": Kd[G1JointIndex.RightWristRoll],
                "tau": 0.0,
            }

            cmd["mode_pr"] = Mode.AB
            # cmd["mode_machine"] = 1

        return cmd

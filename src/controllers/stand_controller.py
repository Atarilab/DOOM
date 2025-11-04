import time
from typing import TYPE_CHECKING, Any, Dict

import torch

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

        # # When Init Controller is called, set the init frame
        # if self.robot.mj_model is not None:
        #     self.robot.mj_model.set_initial_world_frame(state, caller=self.__class__.__name__)

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
        if self.robot.actuated_joint_indices is not None:
            for i, motor_idx in enumerate(self.robot.actuated_joint_indices):
                cmd[f"motor_{motor_idx}"] = {
                    "q": 0.0,
                    "kp": 0.0,
                    "dq": 0.0,
                    "kd": self.robot.damping_gain,
                    "tau": 0.0,
                }
            for i, motor_idx in enumerate(self.robot.non_actuated_joint_indices):
                cmd[f"motor_{motor_idx}"] = {
                    "q": 0.0,
                    "kp": 0.0,
                    "dq": 0.0,
                    "kd": 0.0,
                    "tau": 0.0,
                }
        else:
            for i in range(self.robot.num_joints):
                cmd[f"motor_{i}"] = {
                    "q": 0.0,
                    "kp": 0.0,
                    "dq": 0.0,
                    "kd": self.robot.damping_gain,
                    "tau": 0.0,
                }
        return cmd


class Go2StandUpController(ControllerBase):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

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
                obs_dim=1,
            ),
        )

    def compute_lowlevelcmd(self, state):
        super().compute_lowlevelcmd(state)
        obs = self.obs_manager.compute(state)

        time = obs["time"] - self.start_time
        phase = torch.tanh(torch.tensor(time / 1.2, device=self.device))

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

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

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
                obs_dim=1,
            ),
        )

    def compute_lowlevelcmd(self, state):
        super().compute_lowlevelcmd(state)

        obs = self.obs_manager.compute(state)

        time = obs["time"] - self.start_time
        phase = torch.tanh(torch.tensor(time / 1.2, device=self.device))
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

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

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
                obs_dim=1,
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

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

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
                obs_dim=1,
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


class G1PhasePDController(ControllerBase):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

        self.name = "G1PhasePDController"
        self.total_time = 5  # 2 seconds
        self.num_steps = int(self.total_time / self.control_dt)

        self.leg_joint2motor_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.leg_kps = [350.0, 200.0, 200.0, 300.0, 300.0, 150.0, 350.0, 200.0, 200.0, 300.0, 300.0, 150.0]
        self.leg_kds = [5.0, 5.0, 5.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 10.0, 5.0, 5.0]
        self.leg_default_angles = [
            -0.312,
            0.0,
            0.0,
            0.669,
            -0.33,
            0.0,
            -0.312,
            0.0,
            0.0,
            0.669,
            -0.33,
            0.0,
        ]

        self.arm_waist_joint2motor_idx = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        self.arm_waist_kps = [300, 300, 300, 100, 100, 50, 50, 20, 20, 20, 100, 100, 50, 50, 20, 20, 20]
        self.arm_waist_kds = [3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1]

        # self.dof_idx = self.leg_joint2motor_idx + self.arm_waist_joint2motor_idx
        self.kps = torch.tensor(self.leg_kps + self.arm_waist_kps, dtype=torch.float32, device=self.device)
        self.kds = torch.tensor(self.leg_kds + self.arm_waist_kds, dtype=torch.float32, device=self.device)

        self.final_pos = torch.zeros(len(self.robot.actuated_joint_indices), dtype=torch.float32, device=self.device)
        self.dof_size = len(self.robot.actuated_joint_indices)
        self.init_dof_pos = torch.zeros(self.dof_size, dtype=torch.float32, device=self.device)

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
                obs_dim=1,
            ),
        )

    def compute_lowlevelcmd(self, state):
        super().compute_lowlevelcmd(state)
        self.obs_manager.compute(state)
        # record the current pos
        for i, motor_idx in enumerate(self.robot.actuated_joint_indices):
            self.init_dof_pos[i] = state["robot/joint_pos"][motor_idx]

        alpha = self.step_counter / self.num_steps

        cmd = {}
        for i, motor_idx in enumerate(self.robot.actuated_joint_indices):
            cmd[f"motor_{motor_idx}"] = {
                "q": alpha * self.final_pos[i] + (1 - alpha) * self.init_dof_pos[i],
                "kp": self.kps[i],
                "dq": 0.0,
                "kd": self.kds[i],
                "tau": 0.0,
            }
        for i, motor_idx in enumerate(self.robot.non_actuated_joint_indices):
            cmd[f"motor_{motor_idx}"] = {
                "q": 0.0,
                "kp": 0.0,
                "dq": 0.0,
                "kd": 0.0,
                "tau": 0.0,
            }

        self.step_counter = min(self.step_counter + 1, self.num_steps)
        return cmd


class G1LateralHandsController(G1PhasePDController):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

        self.name = "G1LateralHandsController"
        self.waist_default_targets = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.arm_default_targets = torch.tensor(
            [0.0, 1.3, 0.0000, 1.3, 0.0000, 0.0000, 0.0000, 0.0, -1.3, 0.0000, 1.3, 0.0000, 0.0000, 0.0000],
            dtype=torch.float32,
            device=self.device,
        )
        self.final_pos = torch.cat(
            (
                torch.tensor(self.leg_default_angles, dtype=torch.float32, device=self.device),
                self.waist_default_targets,
                self.arm_default_targets,
            ),
            dim=0,
        )


class G1DefaultHandsController(G1PhasePDController):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

        self.name = "G1DefaultHandsController"
        self.waist_default_targets = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.arm_default_targets = torch.tensor(
            [
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.final_pos = torch.cat(
            (
                torch.tensor(self.leg_default_angles, dtype=torch.float32, device=self.device),
                self.waist_default_targets,
                self.arm_default_targets,
            ),
            dim=0,
        )


class G1ZeroLegController(G1PhasePDController):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

        self.name = "G1ZeroLegController"
        self.waist_default_targets = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        default_joint_pos = torch.tensor(
            self.configs["controller_config"]["default_joint_pos"], dtype=torch.float32, device=self.device
        )
        if default_joint_pos.shape[0] == 17:
            self.waist_default_targets = default_joint_pos[:3]
            self.arm_default_targets = default_joint_pos[3:]
        else:
            self.waist_default_targets = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
            self.arm_default_targets = default_joint_pos.clone()

        self.kps[self.leg_joint2motor_idx] = 0.0
        self.kds[self.leg_joint2motor_idx] = 0.0

        self.final_pos = torch.cat(
            (
                torch.tensor(self.leg_default_angles, dtype=torch.float32, device=self.device),
                self.waist_default_targets,
                self.arm_default_targets,
            ),
            dim=0,
        )


class G1ManipulationInitHandsController(G1PhasePDController):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

        self.name = "G1ManipulationInitHandsController"
        self.waist_default_targets = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.arm_default_targets = torch.tensor(
            [
                -0.4500,
                0.5000,
                0.0000,
                0.5000,
                0.0000,
                0.0000,
                -0.8000,
                -0.4500,
                -0.5000,
                0.0000,
                0.5000,
                0.0000,
                0.0000,
                0.8000,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.final_pos = torch.cat(
            (
                torch.tensor(self.leg_default_angles, dtype=torch.float32, device=self.device),
                self.waist_default_targets,
                self.arm_default_targets,
            ),
            dim=0,
        )


class G1UpperPhasePDController(G1PhasePDController):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

        self.name = "G1UpperPhasePDController"
        # self.dof_idx = range(self.robot.num_joints)
        # self.dof_idx = self.robot.actuated_joint_indices
        # self.dof_size = len(self.dof_idx)
        # self.init_dof_pos = np.zeros(self.dof_size, dtype=np.float32)
        # self.final_pos = np.zeros(self.dof_size, dtype=np.float32)
        self.step_counter = 0
        # self.kps = np.array([300, 300, 300, 100, 100, 50, 50, 20, 20, 20, 100, 100, 50, 50, 20, 20, 20])
        # self.kds = np.array([3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1])
        self.kps = torch.tensor(
            [100, 100, 50, 50, 20, 20, 20, 100, 100, 50, 50, 20, 20, 20], dtype=torch.float32, device=self.device
        )
        self.kds = torch.tensor([2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1], dtype=torch.float32, device=self.device)


class G1LowerPhasePDController(G1PhasePDController):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

        self.name = "G1LowerPhasePDController"
        # self.dof_idx = range(self.robot.num_joints)
        # self.dof_idx = self.robot.actuated_joint_indices
        # self.dof_size = len(self.dof_idx)
        # self.init_dof_pos = np.zeros(self.dof_size, dtype=np.float32)
        # self.final_pos = np.zeros(self.dof_size, dtype=np.float32)
        self.step_counter = 0
        # self.kps = np.array([300, 300, 300, 100, 100, 50, 50, 20, 20, 20, 100, 100, 50, 50, 20, 20, 20])
        # self.kds = np.array([3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1])
        self.kps = torch.tensor(
            [
                100,
                100,
                100,
                150,
                40,
                40,
                100,
                100,
                100,
                150,
                40,
                40,
                300,
                100,
                100,
                50,
                50,
                20,
                20,
                20,
                100,
                100,
                50,
                50,
                20,
                20,
                20,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.kds = torch.tensor(
            [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 3, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1],
            dtype=torch.float32,
            device=self.device,
        )


class G1StandUpController(G1PhasePDController):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

        self.name = "G1StandUpController"
        default_joint_pos = torch.tensor(
            self.configs["controller_config"]["default_joint_pos"], dtype=torch.float32, device=self.device
        )
        if default_joint_pos.shape[0] == 29:
            self.final_pos = default_joint_pos
        elif default_joint_pos.shape[0] == 11:
            self.waist_default_targets = default_joint_pos[:3]
            self.arm_default_targets = torch.cat(
                [
                    default_joint_pos[3:7],
                    torch.tensor([0.0, 0.0, -1.449], dtype=torch.float32, device=self.device),
                    default_joint_pos[7:],
                    torch.tensor([0.0, 0.0, 1.449], dtype=torch.float32, device=self.device),
                ],
                dim=0,
            )
            self.final_pos = torch.cat(
                (
                    torch.tensor(self.leg_default_angles, dtype=torch.float32, device=self.device),
                    self.waist_default_targets,
                    self.arm_default_targets,
                ),
                dim=0,
            )
        elif default_joint_pos.shape[0] == 17:
            self.waist_default_targets = default_joint_pos[:3]
            self.arm_default_targets = default_joint_pos[3:]
            self.final_pos = torch.cat(
                (
                    torch.tensor(self.leg_default_angles, dtype=torch.float32, device=self.device),
                    self.waist_default_targets,
                    self.arm_default_targets,
                ),
                dim=0,
            )
        elif default_joint_pos.shape[0] == 12:
            self.waist_default_targets = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
            self.arm_default_targets = torch.tensor(
                [
                    -0.4500,
                    0.4000,
                    0.0000,
                    0.2000,
                    0.0000,
                    0.0000,
                    -1.449,
                    -0.4500,
                    -0.4000,
                    0.0000,
                    0.2000,
                    0.0000,
                    0.0000,
                    1.449,
                ],
                dtype=torch.float32,
                device=self.device,
            )
            self.final_pos = torch.cat(
                (
                    torch.tensor(default_joint_pos.clone(), dtype=torch.float32, device=self.device),
                    self.waist_default_targets,
                    self.arm_default_targets,
                ),
                dim=0,
            )
        else:
            self.waist_default_targets = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
            self.arm_default_targets = default_joint_pos.clone()
            self.final_pos = torch.cat(
                (
                    torch.tensor(self.leg_default_angles, dtype=torch.float32, device=self.device),
                    self.waist_default_targets,
                    self.arm_default_targets,
                ),
                dim=0,
            )


class G1UpperDefaultPosController(G1UpperPhasePDController):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

        self.name = "G1UpperDefaultPosController"
        self.final_pos = torch.tensor(
            [
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            dtype=torch.float32,
            device=self.device,
        )


class G1UpperExtendLateralController(G1UpperPhasePDController):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

        self.name = "G1UpperExtendLateralController"
        self.final_pos = torch.tensor(
            [0.0, 1.3, 0.0000, 1.3, 0.0000, 0.0000, 0.0000, 0.0, -1.3, 0.0000, 1.3, 0.0000, 0.0000, 0.0000],
            dtype=torch.float32,
            device=self.device,
        )


class G1UpperHomePosController(G1UpperPhasePDController):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

        self.name = "G1UpperHomePosController"
        self.final_pos = self.final_pos = torch.tensor(
            configs["controller_config"]["default_joint_pos"], dtype=torch.float32, device=self.device
        )


class G1LowerStandUpController(G1LowerPhasePDController):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

        self.name = "G1LowerStandUpController"
        self.final_pos = torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=torch.float32,
            device=self.device,
        )


class G1StayUpController(ControllerBase):
    """
    The Stay Up Controller is used to stay up close the ground, to prepare to get up.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

        self.name = "G1StayUpController"

        self.leg_joint2motor_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.leg_kps = [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40]
        self.leg_kds = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2]
        self.leg_default_angles = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0]

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
                obs_dim=1,
            ),
        )

    def compute_lowlevelcmd(self, state):
        super().compute_lowlevelcmd(state)

        cmd = {}
        for i in range(len(self.leg_joint2motor_idx)):
            motor_idx = self.leg_joint2motor_idx[i]
            cmd[f"motor_{motor_idx}"] = {
                "q": self.leg_default_angles[i],
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

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

        self.name = "G1LowLevelController"
        self.total_time = 3.0  # 2 seconds
        self.num_steps = int(self.total_time / self.control_dt)

        self.leg_joint2motor_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.leg_kps = [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40]
        self.leg_kds = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2]
        self.leg_default_angles = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0]
        self.arm_waist_joint2motor_idx = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

        self.arm_waist_kps = [300, 300, 300, 100, 100, 50, 50, 20, 20, 20, 100, 100, 50, 50, 20, 20, 20]
        self.arm_waist_kds = [3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1]
        self.arm_waist_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.dof_idx = self.leg_joint2motor_idx + self.arm_waist_joint2motor_idx
        self.kps = self.leg_kps + self.arm_waist_kps
        self.kds = self.leg_kds + self.arm_waist_kds

        self.default_pos = torch.cat(
            (
                torch.tensor(self.leg_default_angles, dtype=torch.float32, device=self.device),
                torch.tensor(self.arm_waist_target, dtype=torch.float32, device=self.device),
            ),
            dim=0,
        )
        self.dof_size = len(self.dof_idx)
        self.init_dof_pos = torch.zeros(self.dof_size, dtype=torch.float32, device=self.device)

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
                obs_dim=1,
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
                ratio = torch.clamp(torch.tensor(time / self.total_time, device=self.device), 0.0, 1.0)
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
            max_P = torch.pi * 30.0 / 180.0
            max_R = torch.pi * 10.0 / 180.0
            t = time - self.total_time
            L_P_des = max_P * torch.sin(2.0 * torch.pi * t)
            L_R_des = max_R * torch.sin(2.0 * torch.pi * t)
            R_P_des = max_P * torch.sin(2.0 * torch.pi * t)
            R_R_des = -max_R * torch.sin(2.0 * torch.pi * t)

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
            max_A = torch.pi * 30.0 / 180.0
            max_B = torch.pi * 10.0 / 180.0
            t = time - self.total_time * 2
            L_A_des = max_A * torch.sin(2.0 * torch.pi * t)
            L_B_des = max_B * torch.sin(2.0 * torch.pi * t + torch.pi)
            R_A_des = -max_A * torch.sin(2.0 * torch.pi * t)
            R_B_des = -max_B * torch.sin(2.0 * torch.pi * t + torch.pi)

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

            max_WristYaw = torch.pi * 30.0 / 180.0
            L_WristYaw_des = max_WristYaw * torch.sin(2.0 * torch.pi * t)
            R_WristYaw_des = max_WristYaw * torch.sin(2.0 * torch.pi * t)
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

            cmd["mode_pr"] = Mode.PR
            # cmd["mode_machine"] = 1

        return cmd

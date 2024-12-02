# controllers/rl_controller.py
import os
import torch
import numpy as np
from controllers.controller_base import ControllerBase
from pprint import pprint  # debugging only, removable

from utils.math import quat_rotate_inverse
from utils.obs_history_storage import ObservationHistoryStorage

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class RLInitPosController(ControllerBase):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot_config):
        self.joint_action_isaac_to_unitree_mapping = torch.tensor(
            robot_config["JOINT_ACTION_ISAAC_LAB_TO_UNITREE_MAPPING"]
        )
        isaac_default_joint_pos = torch.tensor(
            robot_config["ISAAC_LAB_DEFAULT_JOINT_POS"]
        )

        self.rl_init_joint_pos = (
            isaac_default_joint_pos[self.joint_action_isaac_to_unitree_mapping]
            .detach()
            .numpy()
        )

        self.stand_down_joint_pos = robot_config["STAND_DOWN_JOINT_POS"]

        self.start_time = 0.0

    def compute_torques(self, state, desired_goal=None):
        elapsed_time = state["elapsed_time"] - self.start_time
        phase = np.tanh(elapsed_time / 1.2)
        cmd = {}
        for i in range(12):
            cmd[f"motor_{i}"] = {
                "q": phase * self.rl_init_joint_pos[i]
                + (1 - phase) * self.stand_down_joint_pos[i],
                "kp": phase * 50.0 + (1 - phase) * 20.0,
                "dq": 0.0,
                "kd": 3.5,
                "tau": 0.0,
            }
        return cmd


class RLController(ControllerBase):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """

    def __init__(self, robot_config, policy_path: str):
        self.policy = torch.jit.load(
            os.path.join(FILE_DIR, policy_path)
        ).cpu()  # stay on cpu

        # states
        self.base_lin_vel = torch.zeros(3)
        self.base_ang_vel = torch.zeros(3)
        self.projected_gravity = torch.zeros(3)
        self.velocity_commands = torch.zeros(3)  # x-y vel [0,1], and yaw rate [2]
        self.joint_pos = torch.zeros(12)
        self.joint_vel = torch.zeros(12)
        self.last_actions = torch.zeros(12)

        # helpers
        self.gravity_dir = torch.tensor(
            [0.0, 0.0, -1.0]
        )  # definition from IsaacLab (and physical one)

        self.joint_obs_unitree_to_isaac_mapping = torch.tensor(
            robot_config["JOINT_OBSERVATION_UNITREE_TO_ISAAC_LAB_MAPPING"]
        )
        self.default_joint_pos = torch.tensor(
            robot_config["ISAAC_LAB_DEFAULT_JOINT_POS"]
        )
        self.scale = torch.tensor(robot_config["ISAAC_LAB_ACTION_SCALE"])
        self.offset = self.default_joint_pos
        self.KP = robot_config["ISAAC_KP"]
        self.KD = robot_config["ISAAC_KD"]

        self.actions_isaac_to_unitree_mapping = torch.tensor(
            robot_config["JOINT_ACTION_ISAAC_LAB_TO_UNITREE_MAPPING"]
        )

        self.obs_history_storage = ObservationHistoryStorage(
            num_envs=1,
            num_obs=48,
            max_length=5,
            device="cpu",
        )

    def compute_torques(self, state, desired_goal=None):
        # NOTE: IMU is (0-w, 1-x, 2-y, 3-z)
        # NOTE: rpy is (0-roll, 1-pitch, 2-yaw) in rad

        if "sport_state_sim" in state:  # simulation
            self.base_lin_vel = torch.tensor(state["sport_state_sim"]["velocity"])
        elif "vicon_state" in state:  # real robot; using vicon
            raise NotImplementedError("Vicon state not implemented yet")

        self.base_ang_vel = torch.tensor(state["low_state"]["imu/gyroscope"])

        _quat = torch.tensor(state["low_state"]["imu/quat"])
        self.projected_gravity = quat_rotate_inverse(
            _quat, self.gravity_dir
        ) # seems right conversion after checking values
        self.velocity_commands = torch.tensor([0.5, 0.0, 0.0])

        unitree_joint_pos = torch.tensor(state["low_state"]["motor/joint_pos"])
        self.joint_pos = (
            unitree_joint_pos[self.joint_obs_unitree_to_isaac_mapping]
            - self.default_joint_pos
        )  # reorder and relative positions

        unitree_joint_vel = torch.tensor(state["low_state"]["motor/joint_vel"])
        self.joint_vel = unitree_joint_vel[
            self.joint_obs_unitree_to_isaac_mapping
        ]  # reorder (relative values are 0)

        state = torch.cat(
            [
                self.base_lin_vel,
                self.base_ang_vel,
                self.projected_gravity,
                self.velocity_commands,
                self.joint_pos,
                self.joint_vel,
                self.last_actions,
            ]
        ).unsqueeze(0)
        self.obs_history_storage.add(state)
        obs_history = self.obs_history_storage.get()

        command_raw = self.policy(obs_history)
        self.last_actions = command_raw[0]
        command_processed = (
            command_raw * self.scale + self.offset
        )  # rescaling, offsetting
        command_processed = (
            command_processed[0][self.actions_isaac_to_unitree_mapping].detach().numpy()
        )  # re-ordering

        command = {}
        for i in range(12):
            command[f"motor_{i}"] = {
                "q": command_processed[i],
                "kp":  self.KP,
                "dq": 0.0,
                "kd":  self.KD,
                "tau": 0.0,
            }
        return command

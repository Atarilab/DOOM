import time
from typing import TYPE_CHECKING, Any, Dict

import torch

from controllers.rl_controller_base import RLControllerBase
from state_manager.obs_manager import ObsTerm
from utils.frequency_tracker import FrequencyTracker

if TYPE_CHECKING:
    from robots.robot_base import RobotBase

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 9.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


class RLHumanoidBalanceController(RLControllerBase):
    """
    Humanoid Waist Policy for the waist.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):

        super().__init__(robot=robot, configs=configs, debug=debug)

        self.policy_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ]

        self.default_activated_indices = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.int32, device=self.device
        )
        # self.default_activated_indices = torch.tensor([0,1,2,3,7,8,9,10], dtype=torch.int32, device=self.device)

        self.policy_stiffness = torch.tensor(
            [
                STIFFNESS_7520_14,
                STIFFNESS_7520_22,
                STIFFNESS_7520_14,
                STIFFNESS_7520_22,
                2.0 * STIFFNESS_5020,
                2.0 * STIFFNESS_5020,
                STIFFNESS_7520_14,
                STIFFNESS_7520_22,
                STIFFNESS_7520_14,
                STIFFNESS_7520_22,
                2.0 * STIFFNESS_5020,
                2.0 * STIFFNESS_5020,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.policy_damping = torch.tensor(
            [
                DAMPING_7520_14,
                DAMPING_7520_22,
                DAMPING_7520_14,
                DAMPING_7520_22,
                2.0 * DAMPING_5020,
                2.0 * DAMPING_5020,
                DAMPING_7520_14,
                DAMPING_7520_22,
                DAMPING_7520_14,
                DAMPING_7520_22,
                2.0 * DAMPING_5020,
                2.0 * DAMPING_5020,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.policy_joint_indices = [
            self.robot.actuated_joint_names.index(joint_name) for joint_name in self.policy_joint_names
        ]

        self.non_policy_joint_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ]
        self.non_policy_joint_indices = [
            self.robot.actuated_joint_names.index(joint_name)
            for joint_name in self.non_policy_joint_names
            if joint_name in self.robot.actuated_joint_names + self.robot.non_actuated_joint_names
        ]
        self.non_policy_joint_stiffness = torch.tensor(
            [
                STIFFNESS_7520_14,
                2.0 * STIFFNESS_5020,
                2.0 * STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_4010,
                STIFFNESS_4010,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_4010,
                STIFFNESS_4010,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.non_policy_joint_damping = torch.tensor(
            [
                DAMPING_7520_14,
                6.0 * DAMPING_5020,
                2.0 * DAMPING_5020,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_4010,
                DAMPING_4010,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_4010,
                DAMPING_4010,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.effort_limit = torch.tensor(self.robot.effort_limit, dtype=torch.float32, device=self.device)
        self.non_policy_default_angles = [
            0.0000,
            0.0000,
            0.0000,
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
        ]
        self.deactivate_joint_names = []
        self.deactivate_joint_indices = [
            self.robot.actuated_joint_names.index(joint_name)
            for joint_name in self.deactivate_joint_names
            if joint_name in self.robot.actuated_joint_names + self.robot.non_actuated_joint_names
        ]

        # Pre-compute the combined joint indices to avoid repeated set operations
        self.combined_joint_indices = list(set(self.policy_joint_indices + self.deactivate_joint_indices))
        self.action_scale = torch.zeros(len(self.combined_joint_indices), dtype=torch.float32, device=self.device)
        self.actions_mapping = torch.arange(self.action_dim, dtype=torch.int32, device=self.device)
        # self.actions_mapping = torch.arange(len(self.policy_joint_indices), dtype=torch.int32, device=self.device)
        # self.action_scale = 0.0

        # Frequency tracking (logger will be set later in set_cmd_manager)
        self._frequency_tracker = FrequencyTracker(
            name="compute_lowlevelcmd",
            log_interval=2.0,  # Reduced for easier testing
            logger=self.logger,  # Will be updated when logger is available
        )

    def register_observations(self):
        """
        Register observations for balance policy.
        """
        from state_manager.observations import (
            joint_pos_rel,
            joint_vel,
            last_action,
        )

        # - Joint observations
        self.obs_manager.register(
            "joint_pos",
            ObsTerm(
                joint_pos_rel,
                params={
                    "default_joint_pos": self.default_joint_pos[self.default_activated_indices],
                    "mapping": self.policy_joint_indices,
                },
                obs_dim=len(self.policy_joint_names),
                device=self.device,
            ),
        )
        self.obs_manager.register(
            "joint_vel",
            ObsTerm(
                joint_vel,
                params={
                    "mapping": self.policy_joint_indices,
                    # "scale": 0.2,
                },
                obs_dim=len(self.policy_joint_names),
                device=self.device,
            ),
        )
        # - Action observation
        self.obs_manager.register(
            "last_action",
            ObsTerm(
                last_action,
                params={"last_action": lambda: self.raw_action},
                obs_dim=len(self.combined_joint_indices),
                device=self.device,
            ),
        )

    def set_mode(self):
        """Runs when the mode is changed in the UI.
        Generates the future feet positions in the init frame for horizon planning.
        """
        # Call the base class set_mode to activate the controller
        super().set_mode()

    def compute_lowlevelcmd(self, state):
        """
        Compute motor commands using the learned policy.

        :param state: Current robot state
        :return: Motor commands dictionary
        """
        # Frequency tracking
        # frequency = self._frequency_tracker.tick()
        # self.logger.debug(f"compute_lowlevelcmd frequency: {self._frequency_tracker.get_statistics()['current_frequency']:.2f} Hz")
        # self.logger.debug(f"stiffness: {self.policy_stiffness}")
        # self.logger.debug(f"damping: {self.policy_damping}")
        if self.robot.mj_model is not None:
            self.robot.mj_model.update(state)

        start_time = time.perf_counter()

        # Update the latest state for the observation processing thread
        with self._lock:
            self.latest_state = state

        # If we do not use threading, we need to compute the obs first, pass it to the policy, and then compute the joint pos targets
        # else we can compute the joint pos targets directly from the obs tensor
        if self.counter % self.decimation == 0:
            if not self.use_threading:
                obs_tensor = self.obs_manager.compute_full_tensor(self.latest_state, batch_idx=0)
                self.joint_pos_targets[self.combined_joint_indices] = self.compute_joint_pos_targets_from_policy(
                    obs_tensor
                )
            else:
                self.joint_pos_targets[self.combined_joint_indices] = self.compute_joint_pos_targets()

        # # Clip the joint pos targets for safety
        if hasattr(self, "soft_dof_pos_limit"):
            self.joint_pos_targets = self._clip_dof_pos(self.joint_pos_targets)

        for idx, joint_idx in enumerate(self.combined_joint_indices):
            self.cmd[f"motor_{joint_idx}"] = {
                "q": self.joint_pos_targets[joint_idx].item(),
                # "q": self.default_joint_pos_np[idx],
                "kp": self.policy_stiffness[idx],
                "dq": 0.0,
                "kd": self.policy_damping[idx],
                "tau": 0.0,
            }
        for idx, joint_idx in enumerate(self.non_policy_joint_indices):
            self.cmd[f"motor_{joint_idx}"] = {
                "q": self.non_policy_default_angles[idx],
                "kp": self.non_policy_joint_stiffness[idx],
                "dq": 0.0,
                "kd": self.non_policy_joint_damping[idx],
                "tau": 0.0,
            }

        # Track command preparation time
        self.cmd_preparation_time = time.perf_counter() - start_time

        self.cmd["mode_pr"] = self.robot.MotorMode.PR
        self.cmd["mode_machine"] = state["mode_machine"]

        return self.cmd

    """
    Joystick mappings and callbacks        
    """

    def get_joystick_mappings(self):
        """
        Define joystick button mappings for balance control.

        Returns:
            Dict mapping button names to callback functions.
        """
        return {
            # Toggle action scale
            "L1-R1": lambda: self._handle_action_scale_change(),
        }

    """
    Function handlers for changing the task for the rl-reach-controller
    - task, command
    """

    def _handle_action_scale_change(self):
        """
        Handle action scale changes.
        """
        if self.action_scale.sum() == 0.0:

            # Compute base action scale for all policy joints
            self.action_scale = 0.25 * self.effort_limit[self.combined_joint_indices] / self.policy_stiffness

            # Indices in the global actuated joint list
            deactivated_joint_indices = [
                self.robot.actuated_joint_names.index(joint_name)
                for joint_name in self.deactivate_joint_names
                if joint_name in self.robot.actuated_joint_names + self.robot.non_actuated_joint_names
            ]
            # Build a mask over policy joints for those that are deactivated
            if len(deactivated_joint_indices) > 0:
                deactivated_set = set(deactivated_joint_indices)
                policy_deactivated_mask = torch.tensor(
                    [idx in deactivated_set for idx in self.combined_joint_indices],
                    dtype=torch.bool,
                    device=self.device,
                )
                self.action_scale[policy_deactivated_mask] = 0.0
                if self.robot.interface == "real":
                    self.policy_damping[policy_deactivated_mask] = 0.0
                    self.policy_stiffness[policy_deactivated_mask] = 0.0
            self.action_term.action_scale = self.action_scale
        else:
            self.action_scale = torch.zeros(len(self.combined_joint_indices), dtype=torch.float32, device=self.device)

        if self.logger is not None:
            self.logger.debug(f"Action scale changed to: {self.action_scale}")

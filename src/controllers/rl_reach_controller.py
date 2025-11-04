import threading
import time
from typing import TYPE_CHECKING, Any, Dict

from std_msgs.msg import ColorRGBA
import torch

# ROS2 imports for visualization
from visualization_msgs.msg import Marker, MarkerArray

from controllers.action_terms import JointPositionAction
from controllers.rl_controller_base import RLControllerBase
from state_manager.obs_manager import ObsTerm
from utils.frequency_tracker import FrequencyTracker
from utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from robots.robot_base import RobotBase

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 8.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


class RLHumanoidReachController(RLControllerBase):
    """
    Humanoid Reaching Policy for the hands.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):

        super().__init__(robot=robot, configs=configs, debug=debug)

        # Initialize publishers for visualization
        self._init_publishers(configs)

        self.policy_joint_names = [
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
        ]

        self.policy_stiffness = torch.tensor(
            [
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
        self.policy_damping = torch.tensor(
            [
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
        self.policy_joint_indices = [
            self.robot.actuated_joint_names.index(joint_name) for joint_name in self.policy_joint_names
        ]
        self.non_policy_leg_joint_names = [
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
        self.non_policy_waist_joint_names = [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ]
        self.non_policy_leg_joint_indices = [
            self.robot.actuated_joint_names.index(joint_name)
            for joint_name in self.non_policy_leg_joint_names
            if joint_name in self.robot.actuated_joint_names + self.robot.non_actuated_joint_names
        ]
        self.non_policy_waist_joint_indices = [
            self.robot.actuated_joint_names.index(joint_name)
            for joint_name in self.non_policy_waist_joint_names
            if joint_name in self.robot.actuated_joint_names + self.robot.non_actuated_joint_names
        ]
        self.non_policy_joint_stiffness = torch.tensor(
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
                STIFFNESS_7520_14,
                2.0 * STIFFNESS_5020,
                2.0 * STIFFNESS_5020,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.non_policy_leg_joint_stiffness = torch.tensor(
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
        self.non_policy_waist_joint_stiffness = torch.tensor(
            [STIFFNESS_7520_14, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020], dtype=torch.float32, device=self.device
        )
        self.non_policy_leg_joint_damping = torch.tensor(
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
        self.non_policy_waist_joint_damping = torch.tensor(
            [DAMPING_7520_14, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020], dtype=torch.float32, device=self.device
        )
        self.non_policy_leg_joint_stiffness[self.non_policy_leg_joint_indices] = 0.0
        self.non_policy_leg_joint_damping[self.non_policy_leg_joint_indices] = 0.0

        self.effort_limit = torch.tensor(self.robot.effort_limit, dtype=torch.float32, device=self.device)
        self.non_policy_default_angles = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0]
        self.action_scale = torch.zeros(len(self.policy_joint_indices), dtype=torch.float32, device=self.device)
        self.actions_mapping = torch.arange(len(self.policy_joint_indices), dtype=torch.int32, device=self.device)

        self.action_term = JointPositionAction(
            configs=configs,
            action_scale=self.action_scale,
            default_joint_pos=self.default_joint_pos,
            actions_mapping=self.actions_mapping,
        )
        # self.action_scale = 0.0

        # -- Reach command parameters
        self.reach_commands_w = torch.zeros(2, 7, dtype=torch.float32, device=self.device)
        self.reach_commands_b = torch.zeros(2, 7, dtype=torch.float32, device=self.device)
        self.init_reach_commands_b = torch.tensor(
            [[0.2, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0], [0.2, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=self.device,
        )
        self.reach_commands_b = self.init_reach_commands_b.clone()

        self.command_duration = 1.3  # Duration of each contact plan in seconds
        self.command_start_time = time.time()  # When the current plan started
        self._command_lock = threading.RLock()  # Reentrant lock for command changes

        self.time_left = self.command_duration

        # Pending pose updates from joystick
        self.pending_reach_update = None  # Store pending pose update
        self.reach_update_pending = False  # Flag to indicate pending pose update

        # Frequency tracking (logger will be set later in set_cmd_manager)
        self._frequency_tracker = FrequencyTracker(
            name="compute_lowlevelcmd",
            log_interval=2.0,  # Reduced for easier testing
            logger=self.logger,  # Will be updated when logger is available
        )

    def register_observations(self):
        """
        Register observations for reaching policy.
        """
        from state_manager.observations import (
            joint_pos_rel,
            joint_vel,
            last_action,
            reach_commands,
        )

        # - Joint observations
        self.obs_manager.register(
            "joint_pos",
            ObsTerm(
                joint_pos_rel,
                params={
                    "default_joint_pos": self.default_joint_pos,
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
        self.obs_manager.register(
            "reach_commands_b",
            ObsTerm(
                reach_commands, params={"reach_commands": lambda: self.reach_commands_b}, obs_dim=14, device=self.device
            ),
        )
        # - Action observation
        self.obs_manager.register(
            "last_action",
            ObsTerm(
                last_action,
                params={"last_action": lambda: self.action_term.raw_action},
                obs_dim=len(self.policy_joint_indices),
                device=self.device,
            ),
        )

    def set_mode(self):
        """Runs when the mode is changed in the UI.
        Generates the future feet positions in the init frame for horizon planning.
        """
        # Call the base class set_mode to activate the controller
        super().set_mode()

        # Set default task when switching back to RL controller
        with self._command_lock:
            self.reach_commands_b = self.init_reach_commands_b.clone()

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

        # Update contact plan timing with thread safety
        with self._command_lock:
            current_time = time.time()
            elapsed = current_time - self.command_start_time
            self.time_left = max(0, self.command_duration - elapsed)
            # self.time_left = 0.5
            if elapsed >= self.command_duration:
                self.command_start_time = current_time
                # self._resample_commands()

        # Compute contact poses
        self._update_commands()

        # Publish visualizations to RViz
        self.pub_all_visualizations()

        # If we do not use threading, we need to compute the obs first, pass it to the policy, and then compute the joint pos targets
        # else we can compute the joint pos targets directly from the obs tensor
        if self.counter % self.decimation == 0:
            if not self.use_threading:
                obs_tensor = self.obs_manager.compute_full_tensor(self.latest_state, batch_idx=0)
                self.joint_pos_targets[self.policy_joint_indices] = self.compute_joint_pos_targets_from_policy(
                    obs_tensor
                )
            else:
                self.joint_pos_targets[self.policy_joint_indices] = self.compute_joint_pos_targets()

        # # # Clip the joint pos targets for safety
        # if hasattr(self, "soft_dof_pos_limit"):
        #     self.joint_pos_targets = self._clip_dof_pos(self.joint_pos_targets)

        # Clip the joint pos targets for safety based on effort limits
        self.joint_pos_targets[self.policy_joint_indices] = self._clip_joint_pos_by_effort_limit(
            joint_pos_targets=self.joint_pos_targets[self.policy_joint_indices],
            joint_pos=state["robot/joint_pos"][self.policy_joint_indices],
            joint_vel=state["robot/joint_vel"][self.policy_joint_indices],
        )

        # Log each observation with key and value from
        # if self.debug:
        # for name, obs_term in self.obs_manager.obs_terms.items():
        #     self.logger.debug(f"{name}: {obs_term(self.latest_state)}")
        # self.logger.debug("--------------------------------")
        # self.logger.debug(f"Joint pos targets: {self.joint_pos_targets[self.robot.actuated_joint_indices]}")

        for idx, joint_idx in enumerate(self.policy_joint_indices):
            self.cmd[f"motor_{joint_idx}"] = {
                "q": self.joint_pos_targets[joint_idx].item(),
                # "q": self.default_joint_pos_np[idx],
                "kp": self.policy_stiffness[idx],
                "dq": 0.0,
                "kd": self.policy_damping[idx],
                "tau": 0.0,
            }
        for idx, joint_idx in enumerate(self.non_policy_leg_joint_indices):
            self.cmd[f"motor_{joint_idx}"] = {
                "q": self.non_policy_default_angles[idx],
                "kp": self.non_policy_leg_joint_stiffness[idx],
                "dq": 0.0,
                "kd": self.non_policy_leg_joint_damping[idx],
                "tau": 0.0,
            }
        for idx, joint_idx in enumerate(self.non_policy_waist_joint_indices):
            self.cmd[f"motor_{joint_idx}"] = {
                "q": self.non_policy_default_angles[idx],
                "kp": self.non_policy_waist_joint_stiffness[idx],
                "dq": 0.0,
                "kd": self.non_policy_waist_joint_damping[idx],
                "tau": 0.0,
            }

        # Track command preparation time
        self.cmd_preparation_time = time.perf_counter() - start_time

        self.cmd["mode_pr"] = self.robot.MotorMode.PR
        self.cmd["mode_machine"] = state["mode_machine"]

        return self.cmd

    def _update_commands(self):
        """Update reach commands in world and base frames."""
        try:

            # Contact poses in the world frame
            self.reach_commands_w[:, :3], self.reach_commands_w[:, 3:7] = combine_frame_transforms(
                t01=self.latest_state["robot/base_pos_w"].unsqueeze(0).expand(2, -1),
                q01=self.latest_state["robot/base_quat"].unsqueeze(0).expand(2, -1),
                t12=self.reach_commands_b[:, :3],
                q12=self.reach_commands_b[:, 3:7],
            )

        except Exception as e:
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.error(f"Error updating reach commands: {e}")
            else:
                print(f"Error updating reach commands: {e}")

    """
    Joystick mappings and callbacks        
    """

    def get_joystick_mappings(self):
        """
        Define joystick button mappings for task changes and control.

        Returns:
            Dict mapping button names to callback functions.
        """
        seperate_reach_commands = {
            # Reach commands position control
            "L1-up": lambda: self._update_reach_commands_position("x", "left", "increase"),
            "L1-down": lambda: self._update_reach_commands_position("x", "left", "decrease"),
            "L1-left": lambda: self._update_reach_commands_position("y", "left", "increase"),
            "L1-right": lambda: self._update_reach_commands_position("y", "left", "decrease"),
            "L2-up": lambda: self._update_reach_commands_position("z", "left", "increase"),
            "L2-down": lambda: self._update_reach_commands_position("z", "left", "decrease"),
            "R1-up": lambda: self._update_reach_commands_position("x", "right", "increase"),
            "R1-down": lambda: self._update_reach_commands_position("x", "right", "decrease"),
            "R1-left": lambda: self._update_reach_commands_position("y", "right", "increase"),
            "R1-right": lambda: self._update_reach_commands_position("y", "right", "decrease"),
            "R2-up": lambda: self._update_reach_commands_position("z", "right", "increase"),
            "R2-down": lambda: self._update_reach_commands_position("z", "right", "decrease"),
            "L2-R2": lambda: self._handle_reset(),
        }

        coupled_reach_commands = {
            # Coupled reach commands - both hands move together
            "X-up": lambda: self._update_coupled_reach_commands_position("x", "both", "increase"),
            "X-down": lambda: self._update_coupled_reach_commands_position("x", "both", "decrease"),
            "Y-up": lambda: self._update_coupled_reach_commands_position("y", "opposite", "increase"),
            "Y-down": lambda: self._update_coupled_reach_commands_position("y", "opposite", "decrease"),
            "up": lambda: self._update_coupled_reach_commands_position("z", "both", "increase"),
            "down": lambda: self._update_coupled_reach_commands_position("z", "both", "decrease"),
            # Toggle action scale
            "L1-R1": lambda: self._handle_action_scale_change(),
            "L2-R2": lambda: self._handle_reset(),
        }

        return {**seperate_reach_commands, **coupled_reach_commands}

    """
    Function handlers for changing the task for the rl-reach-controller
    - task, command
    """

    def _handle_reset(self):
        """Reset command duration and offset values to default."""
        with self._command_lock:
            # Reset command duration
            self.pending_command_duration = 1.3
            self.command_duration_change_pending = True

            self.reach_commands_b[:3] = self.init_reach_commands_b.clone()
            self.reach_commands_w[:, :3], self.reach_commands_w[:, 3:7] = combine_frame_transforms(
                t01=self.latest_state["robot/base_pos_w"].unsqueeze(0).expand(2, -1),
                q01=self.latest_state["robot/base_quat"].unsqueeze(0).expand(2, -1),
                t12=self.reach_commands_b[:, :3],
                q12=self.reach_commands_b[:, 3:7],
            )

            if self.logger is not None:
                self.logger.debug("Reset command duration and reach commands to default")

    def _handle_action_scale_change(self):
        """
        Handle action scale changes.
        """
        if self.action_scale.sum() == 0.0:
            deactivated_joints = [
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "left_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
                "right_wrist_roll_joint",
            ]
            # Indices in the global actuated joint list
            deactivated_joint_indices = [
                self.robot.actuated_joint_names.index(joint_name)
                for joint_name in deactivated_joints
                if joint_name in self.robot.actuated_joint_names + self.robot.non_actuated_joint_names
            ]
            # Compute base action scale for all policy joints
            self.action_scale = 0.25 * self.effort_limit[self.policy_joint_indices] / self.policy_stiffness

            # Build a mask over policy joints for those that are deactivated
            if len(deactivated_joint_indices) > 0:
                deactivated_set = set(deactivated_joint_indices)
                policy_deactivated_mask = torch.tensor(
                    [idx in deactivated_set for idx in self.policy_joint_indices],
                    dtype=torch.bool,
                    device=self.device,
                )
                self.action_scale[policy_deactivated_mask] = 0.0
            self.action_term.action_scale = self.action_scale
        else:
            self.action_scale = torch.zeros(len(self.policy_joint_indices), dtype=torch.float32, device=self.device)
            self.action_term.action_scale = self.action_scale
        if self.logger is not None:
            self.logger.debug(f"Action scale changed to: {self.action_scale}")

    def _update_reach_commands_position(self, axis, hand, direction):
        """Update reach commands position based on axis and hand."""
        with self._command_lock:
            if axis == "x":
                idx = 0
            elif axis == "y":
                idx = 1
            elif axis == "z":
                idx = 2
            else:
                return
            if hand == "left":
                self.reach_commands_b[0, idx] += 0.03 if direction == "increase" else -0.03
            else:
                self.reach_commands_b[1, idx] += 0.03 if direction == "increase" else -0.03

            if self.logger is not None:
                hand_idx = 0 if hand == "left" else 1
                self.logger.debug(
                    f"Updated reach commands position {axis} for {hand} to {self.reach_commands_b[hand_idx, idx]:.3f}"
                )

    def _update_coupled_reach_commands_position(self, axis, mode, direction):
        """Update reach commands position for coupled movement of both hands."""
        with self._command_lock:
            if axis == "x":
                idx = 0
            elif axis == "y":
                idx = 1
            elif axis == "z":
                idx = 2
            else:
                return

            delta = 0.2 if direction == "increase" else -0.2

            if mode == "both":
                # Both hands move in the same direction
                self.reach_commands_b[0, idx] += delta  # left hand
                self.reach_commands_b[1, idx] += delta  # right hand
            elif mode == "opposite":
                # Hands move in opposite directions
                self.reach_commands_b[0, idx] += delta  # left hand
                self.reach_commands_b[1, idx] -= delta  # right hand (opposite direction)

            if self.logger is not None:
                self.logger.debug(
                    f"Updated coupled reach commands position {axis} (mode: {mode}) - left: {self.reach_commands_b[0, idx]:.3f}, right: {self.reach_commands_b[1, idx]:.3f}"
                )

    """
    Helper functions to initialize publishers for visualization.
    """

    def _init_publishers(self, configs):
        """
        Initialize publishers for visualization.
        """
        # Initialize visualization flags from configs
        self.visualize = {
            key: configs["controller_config"]["visualize"][key] for key in configs["controller_config"]["visualize"]
        }

        if self.visualize.get("reach_commands", False):
            # Add marker publisher for reach commands
            self.reach_commands_markers_pub = self.create_publisher(MarkerArray, "reach_commands_markers", 10)

    def pub_reach_commands(self):
        """
        Publish the reach commands as sphere markers in RViz.
        """
        try:
            # Check if publisher is initialized
            if not hasattr(self, "reach_commands_markers_pub"):
                return

            # Create marker array for reach commands
            marker_array = MarkerArray()

            for i in range(2):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "reach_commands"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                # Set position from reach_commands
                marker.pose.position.x = float(self.reach_commands_w[i, 0])
                marker.pose.position.y = float(self.reach_commands_w[i, 1])
                marker.pose.position.z = float(self.reach_commands_w[i, 2])

                # Set orientation (identity quaternion)
                marker.pose.orientation.w = 1.0
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0

                # Set scale (sphere radius)
                marker.scale.x = 0.03
                marker.scale.y = 0.03
                marker.scale.z = 0.03

                # Set color based on contact plan
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
                marker_array.markers.append(marker)

            # Publish marker array
            self.reach_commands_markers_pub.publish(marker_array)

        except Exception as e:
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.error(f"Failed to publish reach commands: {e}")

    def pub_all_visualizations(self):
        """
        Publish all visualization data to RViz.
        """
        try:
            if hasattr(self, "visualize"):
                if self.visualize.get("reach_commands", False):
                    self.pub_reach_commands()

        except Exception as e:
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.error(f"Failed to publish visualizations: {e}")

    def _clip_joint_pos_by_effort_limit(
        self, joint_pos_targets: torch.Tensor, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> torch.Tensor:
        """
        Clip joint position targets based on 0.9 * effort_limit for PD-like controller.

        This method clips joint positions to ensure the resulting torques from the PD controller
        don't exceed 90% of the motor effort limits, providing safety margins.

        :param joint_pos_targets: Desired joint positions for policy joints
        :return: Clipped joint positions constrained by effort limits
        """
        # Get current joint positions for the policy joints
        current_joint_pos = joint_pos

        # Calculate position error
        pos_error = joint_pos_targets - current_joint_pos

        # Calculate maximum allowed position error based on effort limit
        # For a PD controller: tau = kp * pos_error + kd * vel_error
        # We want: |kp * pos_error + kd * vel| <= 0.9 * effort_limit

        # Calculate current damping torque component (can be positive or negative)
        damping_torque = self.policy_damping * joint_vel

        # Compute available positive and negative torque budgets separately
        available_torque_pos = 0.8 * self.effort_limit[self.policy_joint_indices] - (-damping_torque)
        available_torque_neg = 0.8 * self.effort_limit[self.policy_joint_indices] - (damping_torque)

        # Clamp to nonnegative values
        available_torque_pos = torch.clamp(available_torque_pos, min=0.0)
        available_torque_neg = torch.clamp(available_torque_neg, min=0.0)

        # Compute max error for positive and negative torque directions
        max_pos_error = available_torque_pos / self.policy_stiffness
        max_neg_error = available_torque_neg / self.policy_stiffness

        # Clip accordingly
        clipped_pos_error = torch.max(torch.min(pos_error, max_pos_error), -max_neg_error)

        # Return clipped joint positions
        return current_joint_pos + clipped_pos_error

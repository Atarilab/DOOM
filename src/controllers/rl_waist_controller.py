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


class RLHumanoidWaistController(RLControllerBase):
    """
    Humanoid Waist Policy for the waist.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):

        super().__init__(robot=robot, configs=configs, debug=debug)

        # Initialize publishers for visualization
        self._init_publishers(configs)

        self.policy_joint_names = [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            # "left_wrist_roll_joint",
            # "left_wrist_pitch_joint",
            # "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            # "right_wrist_roll_joint",
            # "right_wrist_pitch_joint",
            # "right_wrist_yaw_joint",
        ]

        self.default_activated_indices = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13], dtype=torch.int32, device=self.device
        )
        # self.default_activated_indices = torch.tensor([0,1,2,3,7,8,9,10], dtype=torch.int32, device=self.device)

        self.policy_stiffness = torch.tensor(
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
        self.policy_damping = torch.tensor(
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
        self.policy_joint_indices = [
            self.robot.actuated_joint_names.index(joint_name) for joint_name in self.policy_joint_names
        ]

        self.non_policy_joint_names = [
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
            # "waist_yaw_joint",
            # "waist_roll_joint",
            # "waist_pitch_joint",
        ]
        self.non_policy_joint_indices = [
            self.robot.actuated_joint_names.index(joint_name)
            for joint_name in self.non_policy_joint_names
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
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.non_policy_joint_damping = (
            torch.tensor(
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
            * 0.5
        )
        self.effort_limit = torch.tensor(self.robot.effort_limit, dtype=torch.float32, device=self.device)
        self.non_policy_default_angles = [
            -0.1,
            0.0,
            0.0,
            0.3,
            -0.2,
            0.0,
            -0.1,
            0.0,
            0.0,
            0.3,
            -0.2,
            0.0,
            # 0.0, 0.0, 0.0,
        ]
        self.deactivate_joint_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
        ]
        self.deactivate_joint_indices = [
            self.robot.actuated_joint_names.index(joint_name)
            for joint_name in self.deactivate_joint_names
            if joint_name in self.robot.actuated_joint_names + self.robot.non_actuated_joint_names
        ]

        # Pre-compute the combined joint indices to avoid repeated set operations
        self.combined_joint_indices = list(set(self.policy_joint_indices + self.deactivate_joint_indices))

        self.action_scale = torch.zeros(len(self.combined_joint_indices), dtype=torch.float32, device=self.device)
        self.actions_mapping = torch.arange(len(self.combined_joint_indices), dtype=torch.int32, device=self.device)

        # self.actions_mapping = torch.arange(len(self.policy_joint_indices), dtype=torch.int32, device=self.device)
        # self.action_scale = 0.0

        # -- Waist command parameters
        self.waist_commands = torch.zeros(3, dtype=torch.float32, device=self.device)
        self.init_waist_commands = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.waist_commands = self.init_waist_commands.clone()
        self.command_duration = 1.3  # Duration of each contact plan in seconds
        self.command_start_time = time.time()  # When the current plan started
        self._command_lock = threading.RLock()  # Reentrant lock for command changes

        self.time_left = self.command_duration

        # Pending pose updates from joystick
        self.pending_waist_update = None  # Store pending pose update
        self.waist_update_pending = False  # Flag to indicate pending pose update

        # Frequency tracking (logger will be set later in set_cmd_manager)
        self._frequency_tracker = FrequencyTracker(
            name="compute_lowlevelcmd",
            log_interval=2.0,  # Reduced for easier testing
            logger=self.logger,  # Will be updated when logger is available
        )
        self.action_term = JointPositionAction(
            configs=configs,
            action_scale=self.action_scale,
            default_joint_pos=self.default_joint_pos,
            actions_mapping=self.actions_mapping,
        )

    def register_observations(self):
        """
        Register observations for waist policy.
        """
        from state_manager.observations import (
            joint_pos_rel,
            joint_vel,
            last_action,
            waist_commands,
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
        self.obs_manager.register(
            "waist_commands",
            ObsTerm(
                waist_commands, params={"waist_commands": lambda: self.waist_commands}, obs_dim=3, device=self.device
            ),
        )
        # - Action observation
        self.obs_manager.register(
            "last_action",
            ObsTerm(
                last_action,
                params={"last_action": lambda: self.action_term.raw_action},
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

        # Set default task when switching back to RL controller
        with self._command_lock:
            self.waist_commands = self.init_waist_commands.clone()

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

        # Publish visualizations to RViz
        self.pub_all_visualizations()

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

        # Log each observation with key and value from
        # if self.debug:
        # for name, obs_term in self.obs_manager.obs_terms.items():
        #     self.logger.debug(f"{name}: {obs_term(self.latest_state)}")
        # self.logger.debug("--------------------------------")
        # self.logger.debug(f"Joint pos targets: {self.joint_pos_targets[self.robot.actuated_joint_indices]}")

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
        Define joystick button mappings for waist control.

        Returns:
            Dict mapping button names to callback functions.
        """
        return {
            # Waist yaw control (left/right rotation)
            "left": lambda: self._update_waist_commands("yaw", "decrease"),
            "right": lambda: self._update_waist_commands("yaw", "increase"),
            # Waist roll control (side-to-side tilt)
            "L1-left": lambda: self._update_waist_commands("roll", "decrease"),
            "L1-right": lambda: self._update_waist_commands("roll", "increase"),
            # Waist pitch control (forward/backward tilt)
            "R1-left": lambda: self._update_waist_commands("pitch", "increase"),
            "R1-right": lambda: self._update_waist_commands("pitch", "decrease"),
            # Toggle action scale
            "L1-R1": lambda: self._handle_action_scale_change(),
            # Reset waist commands
            "L2-R2": lambda: self._handle_reset(),
        }

    """
    Function handlers for changing the task for the rl-reach-controller
    - task, command
    """

    def _handle_reset(self):
        """Reset waist commands to default."""
        with self._command_lock:
            self.waist_commands = self.init_waist_commands.clone()

            if self.logger is not None:
                self.logger.debug("Reset waist commands to default")

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
            self.action_term.action_scale = self.action_scale
        if self.logger is not None:
            self.logger.debug(f"Action scale changed to: {self.action_scale}")

    def _update_waist_commands(self, axis, direction):
        """Update waist commands based on axis and direction.

        Args:
            axis: 'yaw', 'roll', or 'pitch'
            direction: 'increase' or 'decrease'
        """
        with self._command_lock:
            if axis == "yaw":
                idx = 0  # waist_yaw
            elif axis == "roll":
                idx = 1  # waist_roll
            elif axis == "pitch":
                idx = 2  # waist_pitch
            else:
                return

            delta = 0.2 if direction == "increase" else -0.2
            self.waist_commands[idx] += delta

            # Clamp values to reasonable range
            self.waist_commands[idx] = torch.clamp(self.waist_commands[idx], -1.5, 1.5)

            if self.logger is not None:
                self.logger.debug(f"Updated waist {axis} to {self.waist_commands[idx]:.3f}")

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

        if self.visualize.get("waist_commands", False):
            # Add marker publisher for waist commands
            self.waist_commands_markers_pub = self.create_publisher(MarkerArray, "waist_commands_markers", 10)

    def pub_waist_commands(self):
        """
        Publish the waist commands as a cuboid marker in RViz.
        """
        try:
            # Check if publisher is initialized
            if not hasattr(self, "waist_commands_markers_pub"):
                return

            # Create marker array for waist commands
            marker_array = MarkerArray()

            # Create cuboid marker representing waist orientation
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waist_commands"
            marker.id = 0
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Set position (above robot)
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 1.0

            # Convert waist commands (yaw, roll, pitch) to quaternion
            from utils.math import euler_to_quaternion

            yaw, roll, pitch = (
                self.waist_commands[0].item(),
                self.waist_commands[1].item(),
                self.waist_commands[2].item(),
            )
            quat = euler_to_quaternion(roll, pitch, yaw)

            # Set orientation from waist commands
            marker.pose.orientation.w = float(quat[0])
            marker.pose.orientation.x = float(quat[1])
            marker.pose.orientation.y = float(quat[2])
            marker.pose.orientation.z = float(quat[3])

            # Set cuboid dimensions (representing waist)
            marker.scale.x = 0.3  # width
            marker.scale.y = 0.2  # depth
            marker.scale.z = 0.4  # height

            # Set color (blue with transparency)
            marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.7)
            marker_array.markers.append(marker)

            # Create text marker showing waist command values
            text_marker = Marker()
            text_marker.header.frame_id = "world"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "waist_commands"
            text_marker.id = 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            # Set position (above the cuboid)
            text_marker.pose.position.x = 0.0
            text_marker.pose.position.y = 0.0
            text_marker.pose.position.z = 1.3

            # Set orientation (identity quaternion)
            text_marker.pose.orientation.w = 1.0
            text_marker.pose.orientation.x = 0.0
            text_marker.pose.orientation.y = 0.0
            text_marker.pose.orientation.z = 0.0

            # Set text content
            text_marker.text = f"Waist Commands:\nYaw: {yaw:.3f}\nRoll: {roll:.3f}\nPitch: {pitch:.3f}"

            # Set scale
            text_marker.scale.z = 0.08

            # Set color
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
            marker_array.markers.append(text_marker)

            # Publish marker array
            self.waist_commands_markers_pub.publish(marker_array)

        except Exception as e:
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.error(f"Failed to publish waist commands: {e}")

    def pub_all_visualizations(self):
        """
        Publish all visualization data to RViz.
        """
        try:
            if hasattr(self, "visualize"):
                if self.visualize.get("waist_commands", False):
                    self.pub_waist_commands()

        except Exception as e:
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.error(f"Failed to publish visualizations: {e}")

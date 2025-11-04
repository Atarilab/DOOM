import logging
import time
from typing import TYPE_CHECKING, Optional

from geometry_msgs.msg import TransformStamped, TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
import torch

# Unitree DDS
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from utils.joystick_interface import JoystickManager

if TYPE_CHECKING:
    from robots.robot_base import RobotBase
    from state_manager.state_manager import StateManager
    from utils.mode_manager import ModeManager


class LowLevelCmdPublisher(Node):
    """Manages low-level robot command publishing."""

    def __init__(
        self,
        dt: float,
        robot: "RobotBase",
        mode_manager: "ModeManager",
        state_manager: "StateManager",
        logger: Optional[logging.Logger] = None,
        debug: bool = False,
    ):
        super().__init__("low_level_cmd")
        self.robot = robot
        self.mode_manager = mode_manager
        self.state_manager = state_manager
        self.logger = logger or logging.getLogger(__name__)

        # Control parameters
        self.dt = dt
        self.running_time = 0.0

        # Timing monitoring
        self.last_callback_time = time.time()
        self.callback_count = 0

        # Initialize command message
        self.dds_cmd = self.robot.low_cmd_msg()
        self.crc = CRC()

        # Initialize command message with robot-specific defaults
        self.robot.init_low_cmd(self.dds_cmd)

        # Mode initialization tracking
        self.mode_initialization_complete = False

        # DDS Publisher setup
        self.dds_pub = ChannelPublisher("rt/lowcmd", self.robot.low_cmd_msg_type)
        self.dds_pub.Init()

        # Setup ROS publishers for visualization
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 1)
        self.unsmoothed_joint_state_pub = self.create_publisher(JointState, "/unsmoothed_joint_states", 1)
        self.joint_torque_pub = self.create_publisher(JointState, "/joint_torques", 1)
        self.unsmoothed_joint_torque_pub = self.create_publisher(JointState, "/unsmoothed_joint_torques", 1)
        self.computed_joint_torque_pub = self.create_publisher(JointState, "/computed_joint_torques", 1)

        self.object_velocity_pub = self.create_publisher(TwistStamped, "/object_velocity", 1)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize joystick manager with debug mode support
        self.joystick_manager = JoystickManager(
            mode_manager=self.mode_manager, robot=self.robot.name, logger=self.logger, debug=debug
        )

        # Create timer for periodic command publishing
        # self.timer = self.create_timer(self.dt, self.low_level_cmd_callback, clock=self.get_clock())
        self.timer = RecurrentThread(interval=self.dt, target=self.low_level_cmd_callback, name="control")
        self.timer.Start()

        # self.last_callback_time = self.get_clock().now().nanoseconds / 1e9

    def low_level_cmd_callback(self):
        """Periodic callback to compute and send motor commands."""
        start_time = time.time()
        self.running_time += self.dt

        # Get active controller and compute torques
        active_controller = self.mode_manager.get_active_controller()
        # self.mode_manager.get_active_obs_manager()

        try:
            # current_time = self.get_clock().now().nanoseconds / 1e9

            # Update joystick state and handle mode switching
            self.joystick_manager.update()

            # Retrieve states from state manager
            combined_state = self.state_manager.get_combined_state()

            # Check mode initialization only if not already complete
            if not self.mode_initialization_complete:
                self.mode_initialization_complete = self.robot.get_mode_initialization_state(combined_state)
                if self.mode_initialization_complete:
                    self.logger.info(f"{self.robot.name}: Receiving robot states, initialization complete")
                else:
                    self.logger.info(f"{self.robot.name}: Waiting for robot states")
                    return

            # Update controller state
            active_controller.update_state(combined_state)

            # Compute motor commands
            try:
                motor_commands = active_controller.compute_lowlevelcmd(combined_state)
            except Exception as e:
                self.logger.error(f"Error computing motor commands: {e}")
                return

            try:
                # Update low-level motor command to the robot
                for i in range(self.robot.num_joints):
                    motor = motor_commands[f"motor_{i}"]
                    self.robot.update_motor_command(self.dds_cmd, i, motor)

                self.robot.update_command_modes(self.dds_cmd, motor_commands)

            except Exception as e:
                self.logger.error(f"Error updating motor commands: {e}")
                return

            # Publish the command
            self.dds_cmd.crc = self.crc.Crc(self.dds_cmd)
            self.dds_pub.Write(self.dds_cmd)

            if combined_state.get("robot/base_pos_w", None) is not None:
                # Publish robot state for visualization
                self.publish_robot_state()

            # Publish object state for visualization if available
            if combined_state.get("object/base_pos_w", None) is not None:
                self.publish_object_state()

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in low level callback computation: {e}")

        # # self.last_callback_time = current_time

        # # Log execution time and actual control frequency
        # execution_time = time.time() - start_time
        # current_time = time.time()
        # actual_interval = current_time - self.last_callback_time
        # self.callback_count += 1

        # # Log every 100 callbacks to avoid spam
        # if self.callback_count % 10 == 0:
        #     actual_freq = 1.0 / actual_interval if actual_interval > 0 else 0
        #     expected_freq = 1.0 / self.dt
        #     self.logger.debug(f"Control frequency: {actual_freq:.1f} Hz (expected: {expected_freq:.1f} Hz), execution: {execution_time*1000:.2f} ms")

        # self.last_callback_time = current_time

    def publish_robot_state(self):
        """Publish robot state for visualization in RViz."""
        current_time = self.get_clock().now()

        combined_state = self.state_manager.get_combined_state()
        # Publish smoothed joint states
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = current_time.to_msg()
        joint_state_msg.name = self.robot.get_joint_names

        # Convert numpy arrays to Python lists of floats
        joint_pos = combined_state.get("robot/joint_pos", [0.0] * self.robot.num_joints)
        joint_vel = combined_state.get("robot/joint_vel", [0.0] * self.robot.num_joints)

        joint_state_msg.position = [float(x) for x in joint_pos]
        joint_state_msg.velocity = [float(x) for x in joint_vel]

        self.joint_state_pub.publish(joint_state_msg)

        # Publish unsmoothed joint states
        unsmoothed_joint_state_msg = JointState()
        unsmoothed_joint_state_msg.header.stamp = current_time.to_msg()
        unsmoothed_joint_state_msg.name = self.robot.get_joint_names

        # Convert unsmoothed numpy arrays to Python lists of floats
        unsmoothed_joint_pos = combined_state.get("robot/unsmoothed_joint_pos", [0.0] * self.robot.num_joints)
        unsmoothed_joint_vel = combined_state.get("robot/unsmoothed_joint_vel", [0.0] * self.robot.num_joints)

        unsmoothed_joint_state_msg.position = [float(x) for x in unsmoothed_joint_pos]
        unsmoothed_joint_state_msg.velocity = [float(x) for x in unsmoothed_joint_vel]

        self.unsmoothed_joint_state_pub.publish(unsmoothed_joint_state_msg)

        # Publish joint torques
        joint_torque_msg = JointState()
        joint_torque_msg.header.stamp = current_time.to_msg()
        joint_torque_msg.name = self.robot.get_joint_names

        # Convert torque numpy arrays to Python lists of floats
        joint_torques = combined_state.get("robot/joint_tau_est", [0.0] * self.robot.num_joints)
        joint_torque_msg.effort = [float(x) for x in joint_torques]

        self.joint_torque_pub.publish(joint_torque_msg)

        # Publish computed joint torques
        computed_joint_torque_msg = JointState()
        computed_joint_torque_msg.header.stamp = current_time.to_msg()
        computed_joint_torque_msg.name = self.robot.get_joint_names

        # Compute torques from joint_pos_targets using policy_stiffness and policy_damping if available
        computed_joint_torques = self._compute_torques_from_targets(combined_state)
        computed_joint_torque_msg.effort = [float(x) for x in computed_joint_torques]

        self.computed_joint_torque_pub.publish(computed_joint_torque_msg)

        # Publish unsmoothed joint torques
        unsmoothed_joint_torque_msg = JointState()
        unsmoothed_joint_torque_msg.header.stamp = current_time.to_msg()
        unsmoothed_joint_torque_msg.name = self.robot.get_joint_names

        # Convert unsmoothed torque numpy arrays to Python lists of floats
        unsmoothed_joint_torques = combined_state.get("robot/unsmoothed_joint_tau_est", [0.0] * self.robot.num_joints)
        unsmoothed_joint_torque_msg.effort = [float(x) for x in unsmoothed_joint_torques]

        self.unsmoothed_joint_torque_pub.publish(unsmoothed_joint_torque_msg)

        # Publish base transform
        transform = TransformStamped()
        transform.header.stamp = current_time.to_msg()
        transform.header.frame_id = "world"
        transform.child_frame_id = self.robot.base_link

        # Set translation from base_pos_w
        transform.transform.translation.x = float(combined_state["robot/base_pos_w"][0])
        transform.transform.translation.y = float(combined_state["robot/base_pos_w"][1])
        transform.transform.translation.z = float(combined_state["robot/base_pos_w"][2])

        # Set rotation from base_quat
        transform.transform.rotation.x = float(combined_state["robot/base_quat"][1])
        transform.transform.rotation.y = float(combined_state["robot/base_quat"][2])
        transform.transform.rotation.z = float(combined_state["robot/base_quat"][3])
        transform.transform.rotation.w = float(combined_state["robot/base_quat"][0])

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform)

    def publish_object_state(self):
        """Publish object state for visualization in RViz."""
        current_time = self.get_clock().now()
        combined_state = self.state_manager.get_combined_state()

        # Publish object transform
        object_transform = TransformStamped()
        object_transform.header.stamp = current_time.to_msg()
        object_transform.header.frame_id = "world"
        object_transform.child_frame_id = "object"

        # Set translation from object/base_pos_w
        object_pos = combined_state.get("object/base_pos_w", [0.0, 0.0, 0.0])
        object_transform.transform.translation.x = float(object_pos[0])
        object_transform.transform.translation.y = float(object_pos[1])
        object_transform.transform.translation.z = float(object_pos[2])

        # Set rotation from object/base_quat
        object_quat = combined_state.get("object/base_quat", [1.0, 0.0, 0.0, 0.0])
        object_transform.transform.rotation.x = float(object_quat[1])
        object_transform.transform.rotation.y = float(object_quat[2])
        object_transform.transform.rotation.z = float(object_quat[3])
        object_transform.transform.rotation.w = float(object_quat[0])

        # Broadcast the object transform
        self.tf_broadcaster.sendTransform(object_transform)

        # Publish object velocity
        object_velocity_msg = TwistStamped()
        object_velocity_msg.header.stamp = current_time.to_msg()
        object_velocity_msg.header.frame_id = "world"

        # Set linear velocity from object/lin_vel_w
        object_lin_vel = combined_state.get("object/lin_vel_w", [0.0, 0.0, 0.0])
        object_velocity_msg.twist.linear.x = float(object_lin_vel[0])
        object_velocity_msg.twist.linear.y = float(object_lin_vel[1])
        object_velocity_msg.twist.linear.z = float(object_lin_vel[2])

        # Set angular velocity from object/ang_vel_w
        object_ang_vel = combined_state.get("object/ang_vel_w", [0.0, 0.0, 0.0])
        object_velocity_msg.twist.angular.x = float(object_ang_vel[0])
        object_velocity_msg.twist.angular.y = float(object_ang_vel[1])
        object_velocity_msg.twist.angular.z = float(object_ang_vel[2])

        # Publish the velocity message
        self.object_velocity_pub.publish(object_velocity_msg)

    def _compute_torques_from_targets(self, combined_state):
        """
        Compute torques from joint_pos_targets using policy_stiffness and policy_damping if available.

        Args:
            combined_state: Combined state dictionary containing robot state

        Returns:
            List of computed torques for all joints
        """
        # Initialize with zeros
        computed_torques = [0.0] * self.robot.num_joints

        # Get active controller
        active_controller = self.mode_manager.get_active_controller()

        # Check if controller has the required attributes
        if not hasattr(active_controller, "joint_pos_targets"):
            return computed_torques

        # Get current joint positions and velocities
        joint_pos = combined_state.get("robot/joint_pos", [0.0] * self.robot.num_joints)
        joint_vel = combined_state.get("robot/joint_vel", [0.0] * self.robot.num_joints)

        # Convert to torch tensors if they aren't already
        if not isinstance(joint_pos, torch.Tensor):
            joint_pos = torch.tensor(joint_pos, dtype=torch.float32)
        if not isinstance(joint_vel, torch.Tensor):
            joint_vel = torch.tensor(joint_vel, dtype=torch.float32)

        # Get joint position targets from controller
        joint_pos_targets = active_controller.joint_pos_targets

        # Check if controller has policy stiffness and damping
        if hasattr(active_controller, "policy_stiffness") and hasattr(active_controller, "policy_damping"):
            # Get policy joint indices
            if hasattr(active_controller, "policy_joint_indices"):
                policy_indices = active_controller.policy_joint_indices
                policy_stiffness = active_controller.policy_stiffness
                policy_damping = active_controller.policy_damping

                # Compute PD torques for policy joints
                for i, joint_idx in enumerate(policy_indices):
                    if joint_idx < len(joint_pos) and i < len(policy_stiffness):
                        pos_error = joint_pos_targets[joint_idx] - joint_pos[joint_idx]
                        vel_error = 0.0 - joint_vel[joint_idx]  # Target velocity is 0
                        torque = policy_stiffness[i] * pos_error + policy_damping[i] * vel_error
                        computed_torques[joint_idx] = float(torque)

        return computed_torques

        # def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "joystick_manager"):
            self.joystick_manager.cleanup()

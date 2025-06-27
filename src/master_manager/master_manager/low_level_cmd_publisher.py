import logging
from typing import Optional

from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

# Unitree DDS
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.utils.crc import CRC
from utils.joystick_interface import JoystickManager


class LowLevelCmdPublisher(Node):
    """Manages low-level robot command publishing."""

    def __init__(
        self,
        dt: float,
        robot: "RobotBase",
        mode_manager: "ModeManager",
        state_manager: "StateManager",
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__("low_level_cmd")

        self.robot = robot
        self.mode_manager = mode_manager
        self.state_manager = state_manager
        self.logger = logger or logging.getLogger(__name__)

        # Control parameters
        self.dt = dt
        self.running_time = 0.0

        # Initialize command message
        self.dds_cmd = self.robot.low_cmd_msg()
        self.crc = CRC()

        if self.robot.name == "UnitreeG1":
            self.motor_mode = MotorMode.pr
            self.mode_machine_ = 0
            self._init_cmd_g1(self.mode_machine_, self.motor_mode)

        elif self.robot.name == "UnitreeGo2":
            self._init_cmd_go2()
        # DDS Publisher setup
        self.dds_pub = ChannelPublisher("rt/lowcmd", self.robot.low_cmd_msg_type)
        self.dds_pub.Init()

        # Setup ROS publishers for visualization
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize joystick manager
        self.joystick_manager = JoystickManager(mode_manager=self.mode_manager, logger=self.logger)

        # Create timer for periodic command publishing
        self.timer = self.create_timer(self.dt, self.low_level_cmd_callback, clock=self.get_clock())

        self.last_callback_time = self.get_clock().now().nanoseconds / 1e9

    def _init_cmd_go2(self):
        """Initialize command message with default values for go2."""
        self.dds_cmd.head[0] = 0xFE
        self.dds_cmd.head[1] = 0xEF
        self.dds_cmd.level_flag = 0xFF
        self.dds_cmd.gpio = 0

        for i in range(len(self.dds_cmd.motor_cmd)):
            self.dds_cmd.motor_cmd[i].mode = 0x01  # PMSM mode
            self.dds_cmd.motor_cmd[i].q = self.dds_cmd.motor_cmd[i].kp = self.dds_cmd.motor_cmd[
                i
            ].dq = self.dds_cmd.motor_cmd[i].kd = self.dds_cmd.motor_cmd[i].tau = 0.0

    def _init_cmd_g1(self, mode_machine, mode_pr):
        """Initialize command message with default values for g1."""

        self.dds_cmd.mode_machine = mode_machine
        self.dds_cmd.mode_pr = mode_pr

        for i in range(len(self.dds_cmd.motor_cmd)):
            self.dds_cmd.motor_cmd[i].mode = 1
            self.dds_cmd.motor_cmd[i].q = self.dds_cmd.motor_cmd[i].kp = self.dds_cmd.motor_cmd[
                i
            ].dq = self.dds_cmd.motor_cmd[i].kd = self.dds_cmd.motor_cmd[i].tau = 0.0

    def low_level_cmd_callback(self):
        """Periodic callback to compute and send motor commands."""
        self.running_time += self.dt

        # Get active controller and compute torques
        active_controller = self.mode_manager.get_active_controller()
        self.mode_manager.get_active_obs_manager()

        try:
            current_time = self.get_clock().now().nanoseconds / 1e9

            # Calculate actual time since last callback

            # Update joystick state and handle mode switching
            self.joystick_manager.update()

            # Retrieve states from state manager
            try:
                combined_state = self.state_manager.get_combined_state()
            except Exception as e:
                self.logger.error(f"Error getting combined state: {e}")
                return

            try:
                active_controller.update_state(combined_state)
            except Exception as e:
                self.logger.error(f"Error updating controller state: {e}")
                return

            # Compute motor commands
            try:
                motor_commands = active_controller.compute_torques(combined_state, {})
            except Exception as e:
                self.logger.error(f"Error computing motor commands: {e}")
                return

            try:
                # Update low-level command to the robot
                num_joints = self.robot.get_num_joints()
                for i in range(num_joints):
                    motor = motor_commands[f"motor_{i}"]
                    for attr in ["q", "kp", "dq", "kd", "tau"]:
                        setattr(self.dds_cmd.motor_cmd[i], attr, motor[attr])
                    if motor_commands.get("mode", None) is not None:
                        self.dds_cmd.motor_cmd[i].mode = motor["mode"]
                if motor_commands.get("mode_pr", None) is not None:
                    self.dds_cmd.mode_pr = motor_commands["mode_pr"]
                if motor_commands.get("mode_machine", None) is not None:
                    self.dds_cmd.mode_machine = motor_commands["mode_machine"]

            except Exception as e:
                self.logger.error(f"Error updating motor commands: {e}")
                return

            if combined_state.get("base_pos_w", None) is not None:
                # Publish robot state for visualization
                self.publish_robot_state()

            # Publish the command
            self.dds_cmd.crc = self.crc.Crc(self.dds_cmd)
            self.dds_pub.Write(self.dds_cmd)

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in low level callback computation: {e}")

        self.last_callback_time = current_time

    def publish_robot_state(self):
        """Publish robot state for visualization in RViz."""
        current_time = self.get_clock().now()

        combined_state = self.state_manager.get_combined_state()
        # Publish joint states
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = current_time.to_msg()
        joint_state_msg.name = self.robot.get_joint_names()

        # Convert numpy arrays to Python lists of floats
        joint_pos = combined_state.get("joint_pos", [0.0] * self.robot.get_num_joints())
        joint_vel = combined_state.get("joint_vel", [0.0] * self.robot.get_num_joints())

        joint_state_msg.position = [float(x) for x in joint_pos]
        joint_state_msg.velocity = [float(x) for x in joint_vel]

        self.joint_state_pub.publish(joint_state_msg)

        # Publish base transform
        transform = TransformStamped()
        transform.header.stamp = current_time.to_msg()
        transform.header.frame_id = "world"
        transform.child_frame_id = "base_link"

        # Set translation from base_pos_w
        transform.transform.translation.x = float(combined_state["base_pos_w"][0])
        transform.transform.translation.y = float(combined_state["base_pos_w"][1])
        transform.transform.translation.z = float(combined_state["base_pos_w"][2])

        # Set rotation from base_quat
        transform.transform.rotation.x = float(combined_state["base_quat"][1])
        transform.transform.rotation.y = float(combined_state["base_quat"][2])
        transform.transform.rotation.z = float(combined_state["base_quat"][3])
        transform.transform.rotation.w = float(combined_state["base_quat"][0])

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform)

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'joystick_manager'):
            self.joystick_manager.cleanup()


class MotorMode:
    pr = 0  # Series Control for Pitch/Roll Joints
    ab = 1  # Parallel Control for A/B Joints
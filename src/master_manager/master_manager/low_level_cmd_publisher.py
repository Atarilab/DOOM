import time
import logging
from typing import TYPE_CHECKING, Optional

from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

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
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize joystick manager with debug mode support
        self.joystick_manager = JoystickManager(mode_manager=self.mode_manager, robot=self.robot.name, logger=self.logger, debug=debug)

        # Create timer for periodic command publishing
        # self.timer = self.create_timer(self.dt, self.low_level_cmd_callback, clock=self.get_clock())
        self.timer = RecurrentThread(
            interval=self.dt, target=self.low_level_cmd_callback, name="control"
        )
        self.timer.Start()

        self.last_callback_time = self.get_clock().now().nanoseconds / 1e9

    def low_level_cmd_callback(self):
        """Periodic callback to compute and send motor commands."""
        start_time = time.time()
        self.running_time += self.dt

        # Get active controller and compute torques
        active_controller = self.mode_manager.get_active_controller()
        self.mode_manager.get_active_obs_manager()

        try:
            current_time = self.get_clock().now().nanoseconds / 1e9

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
            self.logger.debug(f"Sending dds_cmd: {motor_commands}")
            self.dds_pub.Write(self.dds_cmd)

            if combined_state.get("robot/base_pos_w", None) is not None:
                # Publish robot state for visualization
                self.publish_robot_state()
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in low level callback computation: {e}")

        self.last_callback_time = current_time
        
        # Log execution time
        # execution_time = time.time() - start_time
        # self.logger.debug(f"low_level_cmd_callback execution time: {execution_time:.6f} seconds")

    def publish_robot_state(self):
        """Publish robot state for visualization in RViz."""
        current_time = self.get_clock().now()

        combined_state = self.state_manager.get_combined_state()
        # Publish joint states
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = current_time.to_msg()
        joint_state_msg.name = self.robot.get_joint_names

        # Convert numpy arrays to Python lists of floats
        joint_pos = combined_state.get("robot/joint_pos", [0.0] * self.robot.num_joints)
        joint_vel = combined_state.get("robot/joint_vel", [0.0] * self.robot.num_joints)

        joint_state_msg.position = [float(x) for x in joint_pos]
        joint_state_msg.velocity = [float(x) for x in joint_vel]

        self.joint_state_pub.publish(joint_state_msg)

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

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "joystick_manager"):
            self.joystick_manager.cleanup()
            
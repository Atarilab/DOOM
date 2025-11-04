"""
Specialized robot state publisher for publishing robot states to ROS2.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

if TYPE_CHECKING:
    from robots.robot_base import RobotBase


class RobotStatePublisher:
    """Specialized publisher for robot state visualization in RViz."""

    def __init__(
        self,
        robot: "RobotBase",
        node: Node,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize robot state publisher.

        :param robot: Robot instance
        :param node: ROS2 node for creating publishers
        :param logger: Optional logger for debugging
        """
        self.robot = robot
        self.node = node
        self.logger = logger or logging.getLogger(__name__)

        # Create publishers
        self.joint_state_pub = self.node.create_publisher(JointState, "/joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self.node)

    def publish_robot_state(self, combined_state: Dict[str, Any]):
        """
        Publish robot state for visualization in RViz.

        :param combined_state: Combined state dictionary from StateManager
        """
        try:
            current_time = self.node.get_clock().now()

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

            # Publish base transform if available
            if "robot/base_pos_w" in combined_state and "robot/base_quat" in combined_state:
                transform = TransformStamped()
                transform.header.stamp = current_time.to_msg()
                transform.header.frame_id = "world"
                transform.child_frame_id = self.robot.base_link

                # Set translation from base_pos_w
                base_pos = combined_state["robot/base_pos_w"]
                transform.transform.translation.x = float(base_pos[0])
                transform.transform.translation.y = float(base_pos[1])
                transform.transform.translation.z = float(base_pos[2])

                # Set rotation from base_quat
                base_quat = combined_state["robot/base_quat"]
                transform.transform.rotation.x = float(base_quat[1])
                transform.transform.rotation.y = float(base_quat[2])
                transform.transform.rotation.z = float(base_quat[3])
                transform.transform.rotation.w = float(base_quat[0])

                # Broadcast the transform
                self.tf_broadcaster.sendTransform(transform)

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error publishing robot state: {e}")

    def cleanup(self):
        """Clean up resources."""
        # Publishers are automatically cleaned up when the node is destroyed
        pass

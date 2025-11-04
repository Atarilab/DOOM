"""
Example usage of StateManager with publishers for robot state publishing.
"""

import logging

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from state_manager.robot_state_publisher import RobotStatePublisher
from state_manager.state_manager import RobotStateROS2Publisher, ROS2StatePublisher, StateManager


def example_usage():
    """Example of how to use StateManager with publishers."""

    # Setup logging
    logger = logging.getLogger(__name__)

    # Create state manager
    state_manager = StateManager(logger=logger)

    # Example 1: Add a simple ROS2 publisher for joint states
    joint_state_publisher = ROS2StatePublisher(
        topic="/joint_states", node_name="joint_state_publisher", msg_type=JointState, logger=logger
    )
    state_manager.add_publisher("joint_states", joint_state_publisher)

    # Example 2: Add a specialized robot state publisher
    # (This would need a robot instance and ROS2 node)
    # robot_state_publisher = RobotStateROS2Publisher(
    #     topic="/robot_state",
    #     node_name="robot_state_publisher",
    #     msg_type=PoseStamped,
    #     robot=robot_instance,
    #     logger=logger
    # )
    # state_manager.add_publisher("robot_state", robot_state_publisher)

    # Example 3: Using the specialized RobotStatePublisher
    # (This would need a robot instance and ROS2 node)
    # robot_viz_publisher = RobotStatePublisher(
    #     robot=robot_instance,
    #     node=ros2_node,
    #     logger=logger
    # )

    # Simulate getting combined state (normally this comes from subscribers)
    combined_state = {
        "robot/joint_pos": [0.1, 0.2, 0.3, 0.4],
        "robot/joint_vel": [0.01, 0.02, 0.03, 0.04],
        "robot/base_pos_w": [1.0, 2.0, 0.5],
        "robot/base_quat": [1.0, 0.0, 0.0, 0.0],
    }

    # Publish state using specific publisher
    try:
        state_manager.publish_state("joint_states", combined_state)
        logger.info("Published joint states")
    except Exception as e:
        logger.error(f"Error publishing joint states: {e}")

    # Publish to all publishers
    try:
        state_manager.publish_all_states(combined_state)
        logger.info("Published to all publishers")
    except Exception as e:
        logger.error(f"Error publishing to all publishers: {e}")

    # Spin publishers (normally done in main loop)
    state_manager.spin_publishers()

    # Cleanup
    state_manager.destroy_all()


def example_integration_with_master_manager():
    """
    Example of how to integrate robot state publishing into the master manager.
    This shows how the LowLevelCmdPublisher could be modified to use StateManager publishers.
    """

    # In the LowLevelCmdPublisher.__init__ method, you could add:
    #
    # # Add robot state publisher to state manager
    # if hasattr(self, 'node'):  # If we have a ROS2 node
    #     robot_state_publisher = RobotStatePublisher(
    #         robot=self.robot,
    #         node=self.node,
    #         logger=self.logger
    #     )
    #     self.state_manager.add_publisher("robot_state_viz", robot_state_publisher)
    #
    # # In the low_level_cmd_callback method, you could add:
    # # Publish robot state for visualization
    # if combined_state.get("robot/base_pos_w", None) is not None:
    #     self.state_manager.publish_combined_state("robot_state_viz")

    pass


if __name__ == "__main__":
    example_usage()

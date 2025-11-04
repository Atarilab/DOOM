"""
Simple usage example showing how to use the enhanced StateManager with publishers.
"""

import logging

from sensor_msgs.msg import JointState

from state_manager.state_manager import ROS2StatePublisher, StateManager


def demonstrate_state_manager_publishers():
    """Demonstrate how to use StateManager with publishers."""

    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Create state manager
    state_manager = StateManager(logger=logger)

    print("StateManager created successfully!")
    print(f"StateManager has add_publisher method: {hasattr(state_manager, 'add_publisher')}")
    print(f"StateManager has publish_combined_state method: {hasattr(state_manager, 'publish_combined_state')}")

    # Example of adding a publisher (this would work with the updated StateManager)
    try:
        # Create a simple ROS2 publisher
        joint_state_publisher = ROS2StatePublisher(
            topic="/joint_states", node_name="joint_state_publisher", msg_type=JointState, logger=logger
        )

        # Add publisher to state manager
        state_manager.add_publisher("joint_states", joint_state_publisher)
        print("Successfully added joint state publisher!")

        # Simulate some robot state
        robot_state = {
            "robot/joint_pos": [0.1, 0.2, 0.3, 0.4],
            "robot/joint_vel": [0.01, 0.02, 0.03, 0.04],
            "robot/base_pos_w": [1.0, 2.0, 0.5],
            "robot/base_quat": [1.0, 0.0, 0.0, 0.0],
        }

        # Publish the state
        state_manager.publish_state("joint_states", robot_state)
        print("Successfully published robot state!")

        # Publish combined state
        state_manager.publish_combined_state("joint_states")
        print("Successfully published combined state!")

    except Exception as e:
        print(f"Error: {e}")
        print("This is expected if the StateManager hasn't been updated yet.")

    # Cleanup
    try:
        state_manager.destroy_all()
        print("Successfully cleaned up StateManager!")
    except Exception as e:
        print(f"Cleanup error: {e}")


if __name__ == "__main__":
    demonstrate_state_manager_publishers()

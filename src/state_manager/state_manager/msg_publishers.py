import time
from typing import Any, Dict, List, Optional

from geometry_msgs.msg import Twist
import torch

from utils.logger import logging


def object_velocity_publisher(
    combined_state: Dict[str, Any], state_manager, logger: Optional[logging.Logger] = None
) -> None:
    """
    Publish object velocity data to the /object_velocity topic.
    Optimized version with reduced computational overhead.

    Args:
        combined_state: Combined state dictionary containing object velocity data
        state_manager: StateManager instance to get the publisher
        logger: Logger for debugging
    """
    try:
        # Check if object velocity data is available
        if "object/lin_vel_w" not in combined_state or "object/ang_vel_w" not in combined_state:
            return  # Skip debug logging to reduce overhead

        # Get object velocities from state
        object_lin_vel_w = combined_state["object/lin_vel_w"]
        object_ang_vel_w = combined_state["object/ang_vel_w"]

        # Optimized tensor conversion - only convert if necessary
        if isinstance(object_lin_vel_w, torch.Tensor):
            object_lin_vel_w = object_lin_vel_w.detach().cpu().numpy()
        if isinstance(object_ang_vel_w, torch.Tensor):
            object_ang_vel_w = object_ang_vel_w.detach().cpu().numpy()

        # Create Twist message with direct assignment
        twist_msg = Twist()
        twist_msg.linear.x = float(object_lin_vel_w[0])
        twist_msg.linear.y = float(object_lin_vel_w[1])
        twist_msg.linear.z = float(object_lin_vel_w[2])
        twist_msg.angular.x = float(object_ang_vel_w[0])
        twist_msg.angular.y = float(object_ang_vel_w[1])
        twist_msg.angular.z = float(object_ang_vel_w[2])

        # Get the publisher and publish the message
        publisher = state_manager.get_publisher("object_velocity")
        publisher.publish(twist_msg)

    except Exception as e:
        if logger:
            logger.error(f"Failed to publish object velocity: {e}")


def tf_broadcast_publisher(
    combined_state: Dict[str, Any], state_manager, logger: Optional[logging.Logger] = None
) -> None:
    """
    Publish robot base transform for TF broadcasting.
    Optimized version with reduced computational overhead.

    Args:
        combined_state: Combined state dictionary containing robot state data
        state_manager: StateManager instance
        logger: Logger for debugging
    """
    try:
        from geometry_msgs.msg import TransformStamped
        from tf2_ros import TransformBroadcaster

        # Get robot from state_manager
        robot = getattr(state_manager, "robot", None)
        if robot is None:
            return  # Skip debug logging to reduce overhead

        # Publish base transform
        if "robot/base_pos_w" in combined_state and "robot/base_quat" in combined_state:
            transform = TransformStamped()
            transform.header.stamp = state_manager.get_clock().now().to_msg()
            transform.header.frame_id = "world"
            transform.child_frame_id = robot.base_link

            # Set translation and rotation with direct assignment
            base_pos = combined_state["robot/base_pos_w"]
            base_quat = combined_state["robot/base_quat"]

            transform.transform.translation.x = float(base_pos[0])
            transform.transform.translation.y = float(base_pos[1])
            transform.transform.translation.z = float(base_pos[2])
            transform.transform.rotation.x = float(base_quat[1])
            transform.transform.rotation.y = float(base_quat[2])
            transform.transform.rotation.z = float(base_quat[3])
            transform.transform.rotation.w = float(base_quat[0])

            # Broadcast the transform using tf2_ros (TF broadcasting doesn't use the ROS2Publisher)
            tf_broadcaster = TransformBroadcaster(state_manager)
            tf_broadcaster.sendTransform(transform)

    except Exception as e:
        if logger:
            logger.error(f"Failed to publish TF broadcast: {e}")


def joint_state_publisher(
    combined_state: Dict[str, Any], state_manager, logger: Optional[logging.Logger] = None
) -> None:
    """
    Publish joint state data to the /joint_states topic.
    Optimized version with reduced computational overhead.

    Args:
        combined_state: Combined state dictionary containing joint state data
        state_manager: StateManager instance to get the publisher
        logger: Logger for debugging
    """
    try:
        from sensor_msgs.msg import JointState

        # Check if joint state data is available
        if "robot/joint_pos" not in combined_state:
            return  # Skip debug logging to reduce overhead

        # Get joint positions and velocities
        joint_pos = combined_state["robot/joint_pos"]
        joint_vel = combined_state.get("robot/joint_vel", [0.0] * len(joint_pos))

        # Optimized tensor conversion - only convert if necessary
        if isinstance(joint_pos, torch.Tensor):
            joint_pos = joint_pos.detach().cpu().numpy()
        if isinstance(joint_vel, torch.Tensor):
            joint_vel = joint_vel.detach().cpu().numpy()

        # Create JointState message
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = state_manager.get_clock().now().to_msg()

        # Set joint names from robot (cache this to avoid repeated lookups)
        robot = getattr(state_manager, "robot", None)
        if robot is not None:
            joint_state_msg.name = robot.actuated_joint_names
        else:
            # Fallback to generic names if robot not available
            joint_state_msg.name = [f"joint_{i}" for i in range(len(joint_pos))]

        # Set joint positions and velocities with list comprehension for efficiency
        joint_state_msg.position = [float(pos) for pos in joint_pos]
        joint_state_msg.velocity = [float(vel) for vel in joint_vel]

        # Get the publisher and publish the message
        publisher = state_manager.get_publisher("joint_states")
        publisher.publish(joint_state_msg)

    except Exception as e:
        if logger:
            logger.error(f"Failed to publish joint states: {e}")

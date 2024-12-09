#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import threading
from threading import Lock

from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState

class RobotStatePublisher(Node):
    def __init__(self):
        # Initialize ROS 2 node
        super().__init__('robot_state_publisher')
        
        # Create publishers for different state components
        self.base_pose_pub = self.create_publisher(PoseStamped, 'robot/base/pose', 10)
        self.base_twist_pub = self.create_publisher(TwistStamped, 'robot/base/twist', 10)
        
        # Thread-safe lock for accessing latest_state
        self._state_lock = Lock()
        
        # Latest state variable 
        self.latest_state = None
        
        # Create a timer to publish at 100 Hz
        self.timer = self.create_timer(0.01, self.publish_state)  # 0.01 seconds = 100 Hz
    
    def update_latest_state(self, state):
        """
        Update the latest state thread-safely.
        
        :param state: Dictionary containing robot state
        """
        with self._state_lock:
            self.latest_state = state
    
    def publish_state(self):
        """
        Publish robot state as ROS 2 messages.
        """
        with self._state_lock:
            if self.latest_state is None:
                return
            
            current_state = self.latest_state
            
            # Create and publish base pose
            base_pose_msg = PoseStamped()
            base_pose_msg.header.stamp = self.get_clock().now().to_msg()
            base_pose_msg.header.frame_id = 'world'
            if 'base_pos_w' in current_state.keys():
                base_pose_msg.pose.position.x = float(current_state['base_pos_w'][0])
                base_pose_msg.pose.position.y = float(current_state['base_pos_w'][1])
                base_pose_msg.pose.position.z = float(current_state['base_pos_w'][2])
                base_pose_msg.pose.orientation.x = float(current_state['base_quat'][1])
                base_pose_msg.pose.orientation.y = float(current_state['base_quat'][2])
                base_pose_msg.pose.orientation.z = float(current_state['base_quat'][3])
                base_pose_msg.pose.orientation.w = float(current_state['base_quat'][0])
            else:
                base_pose_msg.pose.position.y = 0.0
                base_pose_msg.pose.position.x = 0.0
                base_pose_msg.pose.position.z = 0.0
                base_pose_msg.pose.orientation.x = 0.0
                base_pose_msg.pose.orientation.y = 0.0
                base_pose_msg.pose.orientation.z = 0.0
                base_pose_msg.pose.orientation.w = 0.0
                
            
            # Create and publish base twist
            base_twist_msg = TwistStamped()
            base_twist_msg.header.stamp = self.get_clock().now().to_msg()
            base_twist_msg.header.frame_id = 'base_link'
            base_twist_msg.twist.linear.x = float(current_state.get('lin_vel_b', [0,0,0])[0]) if 'lin_vel_b' in current_state.keys() else 0.0
            base_twist_msg.twist.linear.y = float(current_state.get('lin_vel_b', [0,0,0])[1]) if 'lin_vel_b' in current_state.keys() else 0.0
            base_twist_msg.twist.linear.z = float(current_state.get('lin_vel_b', [0,0,0])[2]) if 'lin_vel_b' in current_state.keys() else 0.0
            base_twist_msg.twist.angular.x = float(current_state.get('ang_vel_b', [0,0,0])[0]) if 'ang_vel_b' in current_state.keys() else 0.0
            base_twist_msg.twist.angular.y = float(current_state.get('ang_vel_b', [0,0,0])[1]) if 'ang_vel_b' in current_state.keys() else 0.0
            base_twist_msg.twist.angular.z = float(current_state.get('ang_vel_b', [0,0,0])[2]) if 'ang_vel_b' in current_state.keys() else 0.0
            
            
            # Publish messages
            self.base_pose_pub.publish(base_pose_msg)
            self.base_twist_pub.publish(base_twist_msg)

def main(args=None):
    # Initialize ROS 2 communication
    rclpy.init(args=args)
    
    # Create the node
    state_publisher = RobotStatePublisher()
    
    try:
        # Spin the node
        rclpy.spin(state_publisher)
    except KeyboardInterrupt:
        state_publisher.get_logger().info('Interrupted. Shutting down.')
    finally:
        # Clean up the node
        state_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
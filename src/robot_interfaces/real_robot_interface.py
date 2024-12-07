import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from robot_interfaces.robot_interface_base import RobotInterfaceBase


class RealRobotInterface(RobotInterfaceBase):
    def __init__(self, node_name="unitree_robot_node"):
        rclpy.init()
        self.node = Node(node_name)

        # Publishers and subscribers
        self.command_pub = self.node.create_publisher(
            Float64MultiArray, "/robot/command", 10
        )
        self.state_sub = self.node.create_subscription(
            Float64MultiArray, "/robot/state", self.state_callback, 10
        )
        self.current_state = {}

    def state_callback(self, msg):
        self.current_state = {"observation": msg.data}

    def send_command(self, command):
        msg = Float64MultiArray()
        msg.data = command["command"]
        self.command_pub.publish(msg)

    def receive_state(self):
        rclpy.spin_once(self.node, timeout_sec=0.1)
        return self.current_state

import rclpy
from rclpy.node import Node
from vicon_receiver.msg import Position

#####################################################################################
# Script for testing the Vicon Subscriber ROS Node. In reality, this is not used..
#####################################################################################


class ViconSubscriber(Node):
    def __init__(self):
        super().__init__("vicon_debug_subscriber")
        self.subscription = self.create_subscription(
            Position, "/vicon/Go2/Go2", self.listener_callback, 10
        )
        print("Subscriber initialized!")

    def listener_callback(self, msg):
        print("VICON MESSAGE RECEIVED!")
        print(f"Full message: {msg}")
        # Print individual fields if possible
        try:
            for field in msg.get_fields_and_field_types().keys():
                print(f"{field}: {getattr(msg, field)}")
        except Exception as e:
            print(f"Error accessing fields: {e}")


def main(args=None):
    rclpy.init(args=args)
    vicon_subscriber = ViconSubscriber()

    try:
        print("Spinning...")
        rclpy.spin(vicon_subscriber)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        vicon_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

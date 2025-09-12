import argparse
import asyncio
import logging
import os
import time
from typing import Optional

import rclpy
from controllers.rl_controller import RLLocomotionVelocityController,RLLocomotionVelocitySineController, GlobalRLLocomotionVelocityControllerBox, RLLocomotionVelocityControllerTorque
from controllers.rl_contact_controller import RLLocomotionContactController
from controllers.stand_controller import (
    IdleController,
    StanceController,
    StandDownController,
    StandUpController,
    StayDownController,
)
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from state_manager.msg_handlers import low_state_handler, sport_mode_state_handler, vicon_handler, vicon_object_handler
from state_manager.state_manager import DDSStateSubscriber, ROS2StateSubscriber, StateManager
from tf2_ros import TransformBroadcaster

# Unitree DDS
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from unitree_sdk2py.utils.crc import CRC
from utils.initialization import initialize_channel, initialize_robot_controller
from utils.logger import get_logger
from utils.mj_wrapper import MjQuadRobotWrapper

# DOOM Imports
from utils.ui_interface import ModeManager, RobotControlUI
from utils.joystick_interface import JoystickManager
# ROS Messages
# from unitree_go.msg._low_state import LowState
# from unitree_go.msg._sport_mode_state import SportModeState
from vicon_receiver.msg import Position


class LowLevelCmdPublisher(Node):
    """Manages low-level robot command publishing."""

    def __init__(
        self,
        dt: float,
        mode_manager: ModeManager,
        state_manager: StateManager,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__("low_level_cmd")

        self.mode_manager = mode_manager
        self.state_manager = state_manager
        self.logger = logger or logging.getLogger(__name__)

        # Control parameters
        self.dt = dt  # This should be 0.005 for 200Hz control
        self.running_time = 0.0

        # DDS Publisher setup
        self.dds_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.dds_pub.Init()

        # Setup ROS publishers for visualization
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize joystick manager
        self.joystick_manager = JoystickManager(mode_manager=self.mode_manager, logger=self.logger)

        # Define joint names in the correct order
        self.joint_names = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]

        # Create timer for periodic command publishing
        self.timer = self.create_timer(dt, self.low_level_cmd_callback, clock=self.get_clock())

        # Initialize command message
        self.dds_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()
        self._init_cmd()

        self.last_callback_time = self.get_clock().now().nanoseconds / 1e9

    def _init_cmd(self):
        """Initialize command message with default values."""
        self.dds_cmd.head[0] = 0xFE
        self.dds_cmd.head[1] = 0xEF
        self.dds_cmd.level_flag = 0xFF
        self.dds_cmd.gpio = 0

        for i in range(20):
            motor_cmd = self.dds_cmd.motor_cmd[i]
            motor_cmd.mode = 0x01  # PMSM mode
            motor_cmd.q = motor_cmd.kp = motor_cmd.dq = motor_cmd.kd = motor_cmd.tau = 0.0

    def low_level_cmd_callback(self):
        """Periodic callback to compute and send motor commands."""
        self.running_time += self.dt

        # Get active controller and compute torques
        active_controller = self.mode_manager.get_active_controller()
        active_obs_manager = self.mode_manager.get_active_obs_manager()

        try:
            current_time = self.get_clock().now().nanoseconds / 1e9

            # Calculate actual time since last callback
            time_since_last_callback = current_time - self.last_callback_time
            # self.logger.debug(f"Time since last low_level_cmd_callback: {time_since_last_callback}")
            
            # self.logger.error(f"Here3")
            

            # Update joystick state and handle mode switching
            joystick_state = self.joystick_manager.update()

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
            
            # self.logger.error(f"Here2")


            # Compute motor commands
            try:
                # self.logger.debug(f"here 1")
                motor_commands = active_controller.compute_torques(combined_state, {})
                self.logger.debug(f"motor_commands after calling active_controller.compute_torques: {motor_commands}")
                
            except Exception as e:
                self.logger.error(f"Error computing motor commands: {e}")
                return

            try:
                # Update command structure
                # self.logger.debug(f"here 5")
                
                for i in range(12):
                    motor = motor_commands[f"motor_{i}"]
                    for attr in ["q", "kp", "dq", "kd", "tau"]:
                        setattr(self.dds_cmd.motor_cmd[i], attr, motor[attr])
            except Exception as e:
                self.logger.error(f"Error updating motor commands: {e}")
                return

            # Publish robot state for visualization
            # self.logger.debug(f"here 6")
            
            self.publish_robot_state()
            # self.logger.debug(f"here 7")

            # Publish the command
            self.dds_cmd.crc = self.crc.Crc(self.dds_cmd)
            self.dds_pub.Write(self.dds_cmd)

        except Exception as e:
            # self.logger.error("here4")
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
        joint_state_msg.name = self.joint_names

        # self.logger.error("here8")
        
        # Convert numpy arrays to Python lists of floats
        joint_pos = combined_state.get("joint_pos", [0.0] * 12)
        joint_vel = combined_state.get("joint_vel", [0.0] * 12)

        joint_state_msg.position = [float(x) for x in joint_pos]
        joint_state_msg.velocity = [float(x) for x in joint_vel]
        
        # self.logger.error("here9")
        

        self.joint_state_pub.publish(joint_state_msg)

        # Publish base transform
        transform = TransformStamped()
        transform.header.stamp = current_time.to_msg()
        transform.header.frame_id = "world"
        transform.child_frame_id = "base_link"
         # self.logger.error(combined_state)
        

        # Set translation from base_pos_w
        transform.transform.translation.x = float(combined_state["base_pos_w"][0])
        transform.transform.translation.y = float(combined_state["base_pos_w"][1])
        transform.transform.translation.z = float(combined_state["base_pos_w"][2])
        # self.logger.error("here11")
        

        # Set rotation from base_quat
        transform.transform.rotation.x = float(combined_state["base_quat"][1])
        transform.transform.rotation.y = float(combined_state["base_quat"][2])
        transform.transform.rotation.z = float(combined_state["base_quat"][3])
        transform.transform.rotation.w = float(combined_state["base_quat"][0])
        # self.logger.error("here12")
        

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform)

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'joystick_manager'):
            self.joystick_manager.cleanup()


node = None
state_manager = None


async def main_async(args=None):
    """Main asynchronous entry point for robot controller."""
    global node
    global state_manager
    rclpy.init(args=args)

    # Parse arguments
    parser = argparse.ArgumentParser(description="ATARI DOOM Robot Controller")
    parser.add_argument("--task", type=str, default="rl-velocity-sim-go2", help="Task name to run")
    parser.add_argument("--log", type=str, default="test", help="Experiment name to log information")
    parser.add_argument("--debug", action="store_true", help="Show debug logs")

    args = parser.parse_args()

    # Setup logger
    log_file = os.path.join("logs", args.log, f"{args.task}_robot_controller.log")
    logger = get_logger(f"{args.task}_robot_controller", log_file, debug=args.debug)

    try:
        logger.info(f"Starting robot controller for task: {args.task}")

        # Load configurations
        configs = await initialize_robot_controller(args.task, logger)

        # Initialize communication channel
        await initialize_channel(args.task, configs["robot_interface_config"], logger)

        state_manager = StateManager(logger=logger)

        dds_low_state_sub = DDSStateSubscriber(
            topic="rt/lowstate",
            msg_type=LowState_,
            handler_func=low_state_handler,
            logger=logger,
        )

        # ros2_low_state_sub = ROS2StateSubscriber(
        #     topic="/lowstate",
        #     node_name="low_state",
        #     msg_type=LowState,
        #     handler_func=low_state_handler
        # )
        # state_manager.add_subscriber("low_state", ros2_low_state_sub)

        state_manager.add_subscriber("low_state", dds_low_state_sub)

        if "sim" in args.task:
            #     ros2_sportsmode_state_sub = ROS2StateSubscriber(
            #         topic="/sportmodestate",
            #         node_name="sportmodestate",
            #         msg_type=SportModeState,
            #         handler_func=sport_mode_state_handler
            #     )
            #     state_manager.add_subscriber("sports_mode_state", ros2_sportsmode_state_sub)

            dds_sportsmode_state_sub = DDSStateSubscriber(
                topic="rt/sportmodestate",
                msg_type=SportModeState_,
                handler_func=sport_mode_state_handler,
                logger=logger,
            )
            state_manager.add_subscriber("sports_mode_state", dds_sportsmode_state_sub)
            
            from state_manager.msg_handlers import object_state_handler
            dds_object_state_sub = DDSStateSubscriber(
                    topic="rt/objectstate",
                    msg_type=SportModeState_,
                    handler_func=object_state_handler,
                    logger=logger,
            )
            state_manager.add_subscriber("object_state", dds_object_state_sub)
        else:
            ros2_vicon_sub = ROS2StateSubscriber(
                # topic="/vicon/Go2with6markers/Go2with6markers",
                topic="/vicon/Go2/Go2",
                node_name="vicon_state",
                msg_type=Position,
                handler_func=vicon_handler,
                logger=logger,
            )
            state_manager.add_subscriber("vicon_state", ros2_vicon_sub)
            
            # ros2_vicon_sub_obj = ROS2StateSubscriber(
            #     topic="/vicon/Step/Step",
            #     node_name="vicon_state_obj",
            #     msg_type=Position,
            #     handler_func=vicon_object_handler,
            #     logger=logger,
            # )
            # state_manager.add_subscriber("vicon_state_obj", ros2_vicon_sub_obj)

        mj_model_wrapper = MjQuadRobotWrapper(configs["robot_config"]["xml_path"])  # Using same URDF for now

        # Create mode manager and register controllers
        mode_manager = ModeManager(logger=logger)
        mode_manager.register_mode("IDLE", {"default": IdleController(mj_model_wrapper, configs)})

        mode_manager.register_mode(
            "STANDING",
            {
                "STAY_DOWN": StayDownController(mj_model_wrapper, configs),
                "STAND_UP": StandUpController(mj_model_wrapper, configs),
                "STAND_DOWN": StandDownController(mj_model_wrapper, configs),
            },
        )

        mode_manager.register_mode("STANCE", {"STANCE": StanceController(mj_model_wrapper, configs)})
        if "rl-contact" in args.task:
            mode_manager.register_mode(
                "RL-CONTACT",
                {
                    "STANCE": StanceController(mj_model_wrapper, configs),
                    "RL-CONTACT": RLLocomotionContactController(mj_model_wrapper=mj_model_wrapper, configs=configs, interface=interface),
                },
            )
        if "rl-velocity" in args.task:
            mode_manager.register_mode(
                "RL-VELOCITY",
                {
                    "STANCE": StanceController(mj_model_wrapper, configs),
                    "RL-VELOCITY": RLLocomotionVelocityController(mj_model_wrapper=mj_model_wrapper, configs=configs),
                    "RL-VELOCITYTORQUE": RLLocomotionVelocityControllerTorque(mj_model_wrapper=mj_model_wrapper, configs=configs),
                    "RL-SINE-VELOCITY": RLLocomotionVelocitySineController(mj_model_wrapper=mj_model_wrapper, configs=configs),
                    "RL-GLOBAL-VELOCITY-BOX": GlobalRLLocomotionVelocityControllerBox(mj_model_wrapper=mj_model_wrapper, configs=configs),
                },
            )
        mode_manager.set_mode("IDLE")

        node = LowLevelCmdPublisher(
            dt=configs["controller_config"]["control_dt"],
            mode_manager=mode_manager,
            state_manager=state_manager,
            logger=logger,
        )

        async def run():
            logger.info("Starting concurrent tasks")

            app_task = asyncio.create_task(RobotControlUI(mode_manager).run_async())

            # Use rclpy.spin_once() in a loop to ensure callbacks are processed
            def spin_node():
                spin_dt = 0.005
                last_spin_time = time.time()

                while rclpy.ok():
                    current_time = time.time()
                    elapsed = current_time - last_spin_time

                    if elapsed >= spin_dt:
                        rclpy.spin_once(node)
                        state_manager.spin_subscribers()
                        last_spin_time = current_time
                    else:
                        # Sleep for a short time to avoid busy waiting
                        time.sleep(max(0, spin_dt - elapsed))

            node_task = asyncio.create_task(asyncio.to_thread(spin_node))
            try:
                await asyncio.gather(app_task, node_task)
            except asyncio.CancelledError:
                logger.info("Tasks were cancelled")
                raise

        await run()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt. Shutting down...")

        # Set controller to idle mode before shutting down
        mode_manager.set_mode("IDLE")
        logger.info("Set controller to IDLE mode before shutdown")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise
    finally:
        if node:
            node.cleanup()
            node.destroy_node()
        if state_manager:
            state_manager.destroy_subscribers()
        rclpy.shutdown()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

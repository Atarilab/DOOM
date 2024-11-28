# New Version

import os
import time
import asyncio
import logging
import argparse
from typing import Optional
import unitree_legged_const as go2

import rclpy
from rclpy.node import Node

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from unitree_go.msg._low_state import LowState

from utils.ui_interface import ModeManager, RobotControlUI
from utils.logger import get_logger 
from utils.initialization import initialize_channel, initialize_robot_controller

from controllers.stand_controller import (
    IdleController, 
    StayDownController, 
    StandUpController, 
    StandDownController, 
    StanceController
)
from state_manager.state_manager import (
    StateManager, 
    DDSStateSubscriber, 
    ROS2StateSubscriber
)

from vicon_receiver.msg import Position

class LowLevelCmdPublisher(Node):
    """Manages low-level robot command publishing."""
    
    def __init__(self, 
                 frequency: float,
                 mode_manager: ModeManager, 
                 state_manager: StateManager,
                 logger: Optional[logging.Logger] = None):
        super().__init__('low_level_cmd')

        self.mode_manager = mode_manager
        self.state_manager = state_manager
        self.logger = logger or logging.getLogger(__name__)

        # DDS Publisher setup
        self.dds_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.dds_pub.Init()

        # Control parameters
        self.dt = 0.002
        self.running_time = 0.0

        # Create timer for periodic command publishing
        self.timer = self.create_timer(self.dt, self.low_level_cmd_callback)

        # Initialize command message
        self.dds_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()
        self._init_cmd()

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
        try:
            # Retrieve states from state manager
            combined_state = {
                "elapsed_time": time.time(),
                "low_state": self.state_manager.get_state("low_state"),
                "vicon_state": self.state_manager.get_state("vicon")
            }

            # Compute motor commands
            motor_commands = active_controller.compute_torques(combined_state, {})
            
            # Update command structure
            for i in range(12):
                motor = motor_commands[f'motor_{i}']
                for attr in ['q', 'kp', 'dq', 'kd', 'tau']:
                    setattr(self.dds_cmd.motor_cmd[i], attr, motor[attr])
            
            # Publish the command
            self.dds_cmd.crc = self.crc.Crc(self.dds_cmd)
            self.dds_pub.Write(self.dds_cmd)
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in command computation: {e}")
                


async def main_async(args=None):
    """Main asynchronous entry point for robot controller."""
    rclpy.init(args=args)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="ATARI DOOM Robot Controller")
    parser.add_argument("--task", type=str, required=True, help="Task name to run")
    parser.add_argument("--log", type=str, required=True, help="Experiment name to log information")
    
    args = parser.parse_args()
    
    # Setup logger
    log_file = os.path.join('logs', f"{args.task}_robot_controller.log")
    logger = get_logger(f"{args.task}_robot_controller", log_file)
    
    try:
        logger.info(f"Starting robot controller for task: {args.task}")

        # Load configurations
        configs = await initialize_robot_controller(args.task, logger)
        
        # Initialize communication channel
        await initialize_channel(args.task, configs['robot_interface_config'], logger)
        
        state_manager = StateManager(logger=logger)
        #######################################################
        # TODO: Move Individual State Management Inside its class and handle them properly
        #######################################################
        # Add DDS Low State Subscriver
        def low_state_handler(msg):
            logger.debug(f"Received low state at {time.time()}")
            print("FR_0 motor state: ", msg.motor_state[go2.LegID["FR_0"]].q)
            try:
                # Log detailed message inspection
                logger.debug(f"Low State Message Details: {vars(msg)}")
            except Exception as e:
                logger.error(f"Error processing low state: {e}")

        # dds_low_state_sub = DDSStateSubscriber(
        #     topic="rt/lowstate", 
        #     msg_type=LowState_, 
        #     handler_func=low_state_handler
        # )
        # state_manager.add_subscriber("low_state", dds_low_state_sub)
        
        ros2_low_state_sub = ROS2StateSubscriber(
            topic="/lowstate", 
            node_name="low_state",
            msg_type=LowState, 
            handler_func=low_state_handler
        )
        state_manager.add_subscriber("low_state", ros2_low_state_sub)

        # Add ROS2 Vicon subscriber
        def vicon_handler(msg):
            logger.debug(f"Received Vicon data at {time.time()}")
            try:
                # Log detailed message inspection
                logger.debug(f"Vicon Message Details: {vars(msg)}")
            except Exception as e:
                logger.error(f"Error processing Vicon data: {e}")
                
        ros2_vicon_sub = ROS2StateSubscriber(
            topic="/vicon/Go2/Go2", 
            node_name="vicon",
            msg_type=Position, 
            handler_func=vicon_handler
        )
        state_manager.add_subscriber("vicon", ros2_vicon_sub)

        # Create mode manager and register controllers
        mode_manager = ModeManager()
        mode_manager.register_mode('IDLE', {
            'default': IdleController()
        })
        
        mode_manager.register_mode('STANDING', {
            'STAY_DOWN': StayDownController(configs['robot_config']),
            'STAND_UP': StandUpController(configs['robot_config']),
            'STAND_DOWN': StandDownController(configs['robot_config']),
        })
        
        mode_manager.register_mode('STANCE', {
            'STANCE': StanceController(configs['robot_config'])
        })
        
        mode_manager.set_mode('IDLE')
        
        node = LowLevelCmdPublisher(frequency=configs['controller_config']['control_dt'],
                                    mode_manager=mode_manager,
                                    state_manager=state_manager,
                                    logger=logger)
        
        # Modify the concurrent running
        async def run_app_and_node():
            logger.info("Starting concurrent tasks")
            
            app_task = asyncio.create_task(RobotControlUI(mode_manager).run_async())
            # Use rclpy.spin_once() in a loop to ensure callbacks are processed
            def spin_node():
                while rclpy.ok():
                    rclpy.spin_once(node)
            
            node_task = asyncio.create_task(asyncio.to_thread(spin_node))
            await asyncio.gather(app_task, node_task)

        await run_app_and_node()

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
def main():
    asyncio.run(main_async())
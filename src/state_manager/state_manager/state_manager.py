import time
import asyncio
import argparse
import os
import logging
from typing import Optional

import rclpy
from rclpy.node import Node

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


from tasks.task_configs import TASK_CONFIG
from utils.ui_interface import ModeManager, RobotControlUI
from utils.config_loader import load_config
from utils.logger import get_logger 
from controllers.stand_controller import *

class LowLevelCmdSender(Node):
    def __init__(self, mode_manager: ModeManager, logger: Optional[logging.Logger] = None):
        super().__init__('state_manager')

        self.mode_manager = mode_manager
         # Create a publisher to publish the data defined in UserData class
        self.dds_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.dds_pub.Init()

        # Parameters
        self.dt = 0.002
        self.running_time = 0.0

        # Create a timer to call the timer_callback method
        self.timer = self.create_timer(self.dt, self.timer_callback)

        # Initialize command message
        self.dds_cmd = unitree_go_msg_dds__LowCmd_()
        
        self.crc = CRC()
        self.init_cmd()
        
        self.logger = logger or logging.getLogger(__name__)      

    def init_cmd(self):

        self.dds_cmd.head[0] = 0xFE
        self.dds_cmd.head[1] = 0xEF
        self.dds_cmd.level_flag = 0xFF
        self.dds_cmd.gpio = 0
        
        for i in range(20):
            self.dds_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.dds_cmd.motor_cmd[i].q = 0.0
            self.dds_cmd.motor_cmd[i].kp = 0.0
            self.dds_cmd.motor_cmd[i].dq = 0.0
            self.dds_cmd.motor_cmd[i].kd = 0.0
            self.dds_cmd.motor_cmd[i].tau = 0.0

    def timer_callback(self):
         
        # Increment running time
        self.running_time += self.dt

        # Get active controller dynamically
        active_controller = self.mode_manager.get_active_controller()
        
        # Compute torques
        state = {
            "elapsed_time": time.time()
        }
        motor_commands = active_controller.compute_torques(state, {})
        
        # Update command structure
        for i in range(12):
            motor = motor_commands[f'motor_{i}']
            for attr in ['q', 'kp', 'dq', 'kd', 'tau']:
                setattr(self.dds_cmd.motor_cmd[i], attr, motor[attr])
        
        # Publish the command
        self.dds_cmd.crc = self.crc.Crc(self.dds_cmd)
        self.dds_pub.Write(self.dds_cmd)

async def main_async(args=None):
    rclpy.init(args=args)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="ATARI DOOM Robot Controller")
    parser.add_argument("--task", type=str, required=True, help="Task name to run")
    parser.add_argument("--log", type=str, required=True, help="Experiment name to log information")
    
    args = parser.parse_args()
    
    # Setup logger with a more specific log file path
    log_file = os.path.join('logs', f"{args.task}_robot_controller.log")
    logger = get_logger(f"{args.task}_robot_controller", log_file)
    
    try:
        # Log the start of the application
        logger.info(f"Starting robot controller for task: {args.task}")

        # Load configurations
        if args.task not in TASK_CONFIG:
            logger.error(f"Unknown task: {args.task}")
            raise ValueError(f"Unknown task: {args.task}")
        
        task_configs = TASK_CONFIG[args.task]
        logger.info("Configurations loaded successfully")

        # Rest of the existing main function code...
        controller_config = load_config(task_configs["controller"])
        robot_interface_config = load_config(task_configs["robot_interface"])
        robot_config = load_config(task_configs["robot"])

        # Log network initialization
        if "sim" in args.task:
            logger.info(f"Initializing channel with Domain ID: {robot_interface_config['DOMAIN_ID']}")
            ChannelFactoryInitialize(
                robot_interface_config["DOMAIN_ID"],
                robot_interface_config["INTERFACE"]
            )
        else:
            network_interface = os.environ.get('NETWORK_INTERFACE')
            logger.info(f"Initializing channel with network interface: {network_interface}")
            ChannelFactoryInitialize(0, network_interface)


        # Create mode manager and register controllers
        mode_manager = ModeManager()
        
        # Register modes (existing code)
        mode_manager.register_mode('IDLE', {
            'default': IdleController()
        })
        
        mode_manager.register_mode('STANDING', {
            'STAY_DOWN': StayDownController(robot_config),
            'STAND_UP': StandUpController(robot_config),
            'STAND_DOWN': StandDownController(robot_config),
        })
        
        mode_manager.register_mode('STANCE', {
            'STANCE': StanceController(robot_config)
        })
        
        # Set the mode upon initialization
        mode_manager.set_mode('IDLE')
        
        # Create node and UI
        node = LowLevelCmdSender(mode_manager)
        # Run Textual App and ROS2 node concurrently
        async def run_app_and_node():
            app_task = asyncio.create_task(RobotControlUI(mode_manager).run_async())
            node_task = asyncio.create_task(asyncio.to_thread(rclpy.spin, node))
            await asyncio.gather(app_task, node_task)

        # Run the concurrent tasks
        await run_app_and_node()

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()

def main():
    asyncio.run(main_async())
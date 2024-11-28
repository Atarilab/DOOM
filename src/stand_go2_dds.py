import asyncio
import os
import argparse
import logging
from typing import Dict, Any, Optional

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_
from unitree_sdk2py.utils.crc import CRC

from tasks.task_configs import TASK_CONFIG
from utils.ui_interface import ModeManager, RobotControlUI
from utils.config_loader import load_config
from utils.logger import get_logger 
from controllers.stand_controller import *


NUM_MOTORS = 12
UPDATE_INTERVAL = 0.002  # 2ms/500Hz control loop
        
class RobotController:
    """
    Generic Robot Controller that reads robot data from Unitree 
    """
    def __init__(self, mode_manager: ModeManager, logger: Optional[logging.Logger] = None):
        self.mode_manager = mode_manager
        self.mode_change_event = asyncio.Event()
        self.crc = CRC()
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.logger = logger or logging.getLogger(__name__)        
        self._init_command()
    
    def _init_command(self):
        """Initialize command structure."""
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0
        
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0
    
    async def run_control_loop(self, publisher: ChannelPublisher): 
        """
        Generic control loop that uses the mode manager.
        
        :param publisher: Channel publisher for sending commands
        """
        while True:
            loop_start = asyncio.get_event_loop().time()
            
            # Wait for mode change if needed
            if self.mode_change_event.is_set():
                # Perform any necessary transition logic
                current_mode_info = self.mode_manager.get_current_mode_info()
                self.logger.info(f"Mode changed to: {current_mode_info}")
                self.mode_change_event.clear()
                
            # Get active controller dynamically
            active_controller = self.mode_manager.get_active_controller()
            
            # Compute torques
            state = {
                "elapsed_time": asyncio.get_event_loop().time()
            }
            motor_commands = active_controller.compute_torques(state, {})
            
            # Update command structure
            for i in range(NUM_MOTORS):
                motor = motor_commands[f'motor_{i}']
                for attr in ['q', 'kp', 'dq', 'kd', 'tau']:
                    setattr(self.cmd.motor_cmd[i], attr, motor[attr])
            
            self.cmd.crc = self.crc.Crc(self.cmd)
            publisher.Write(self.cmd)
            
            # Maintain control frequency
            elapsed = asyncio.get_event_loop().time() - loop_start
            if elapsed < UPDATE_INTERVAL:
                await asyncio.sleep(UPDATE_INTERVAL - elapsed)

async def main():
    
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

        # Initialize communication channels
        publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        publisher.Init()
        
        subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        subscriber.Init(None, 10)

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

        # Initialize robot controller and UI
        robot_controller = RobotController(mode_manager, {}, logger)
        app = RobotControlUI(mode_manager)
        
        logger.info("Starting robot controller and UI")
        # Run concurrently
        await asyncio.gather(
            robot_controller.run_control_loop(publisher),
            app.run_async()
        )

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
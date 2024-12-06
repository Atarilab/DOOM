import os
import asyncio
import logging
import argparse
import numpy as np
from typing import Optional


import rclpy
from rclpy.node import Node

# Unitree DDS
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from unitree_sdk2py.utils.crc import CRC

# ROS Messages
from unitree_go.msg._low_state import LowState
from unitree_go.msg._sport_mode_state import SportModeState
from vicon_receiver.msg import Position

# DOOM Imports
from utils.ui_interface import ModeManager, RobotControlUI
from utils.logger import get_logger 
from utils.initialization import initialize_channel, initialize_robot_controller
from utils.mj_pin_wrapper.pin_robot import PinQuadRobotWrapper

from controllers.stand_controller import (
    IdleController, 
    StayDownController, 
    StandUpController, 
    StandDownController, 
    StanceController
)
from controllers.rl_controller import RLLocomotionVelocityController
from state_manager.state_manager import (
    StateManager, 
    DDSStateSubscriber,
    ROS2StateSubscriber
)
from state_manager.msg_handlers import *


class LowLevelCmdPublisher(Node):
    """Manages low-level robot command publishing."""
    
    def __init__(self, 
                 dt: float,
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
        self.dt = dt
        self.running_time = 0.0

        # Create timer for periodic command publishing
        self.timer = self.create_timer(dt, self.low_level_cmd_callback)

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
        active_obs_manager = self.mode_manager.get_active_obs_manager()

        try:
            # Retrieve states from state manager
            combined_state = self.state_manager.get_combined_state()    
            # self.logger.debug(combined_state['feet_pos'])     
            # observations = active_obs_manager.compute_observations(combined_state)
            # if observations != {}:
            #     self.logger.debug(observations)
            
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
    parser.add_argument("--task", type=str, default="rl-velocity-sim-go2", help="Task name to run")
    parser.add_argument("--log", type=str, default="test", help="Experiment name to log information")
    parser.add_argument("--debug", action="store_true", help="Show debug logs")
    
    args = parser.parse_args()
    
    # Setup logger
    log_file = os.path.join('logs', args.log, f"{args.task}_robot_controller.log")
    logger = get_logger(f"{args.task}_robot_controller", log_file, debug=args.debug)
    
    try:
        logger.info(f"Starting robot controller for task: {args.task}")

        # Load configurations
        configs = await initialize_robot_controller(args.task, logger)

        # Initialize communication channel
        await initialize_channel(args.task, configs['robot_interface_config'], logger)
        
        state_manager = StateManager(logger=logger)
        
        dds_low_state_sub = DDSStateSubscriber(
            topic="rt/lowstate", 
            msg_type=LowState_, 
            handler_func=low_state_handler,
            # handler_args={"pin_model_wrapper": pin_model_wrapper, "joint_mappings": configs['robot_config']['joint_mappings']},
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
        else:  
            ros2_vicon_sub = ROS2StateSubscriber(
                topic="/vicon/Go2with6markers/Go2with6markers", 
                node_name="vicon_state",
                msg_type=Position, 
                handler_func=vicon_handler,
                logger=logger
            )
            state_manager.add_subscriber("vicon_state", ros2_vicon_sub)
        

        pin_model_wrapper = PinQuadRobotWrapper(configs['robot_config']['pinocchio_urdf'])
        
        # Create mode manager and register controllers
        mode_manager = ModeManager(logger=logger)
        mode_manager.register_mode('IDLE', {
            'default': IdleController(pin_model_wrapper, configs)
        })
        
        mode_manager.register_mode('STANDING', {
            'STAY_DOWN': StayDownController(pin_model_wrapper, configs),
            'STAND_UP': StandUpController(pin_model_wrapper, configs),
            'STAND_DOWN': StandDownController(pin_model_wrapper, configs),
        })
        
        mode_manager.register_mode('STANCE', {
            'STANCE': StanceController(pin_model_wrapper, configs)
        })
        
        mode_manager.register_mode('RL-VELOCITY', {
            'STANCE': StanceController(pin_model_wrapper, configs),
            'RL-VELOCITY': RLLocomotionVelocityController(pin_model_wrapper, configs)
        })
        
        mode_manager.set_mode('IDLE')
        
        node = LowLevelCmdPublisher(dt=configs['controller_config']['control_dt'],
                                    mode_manager=mode_manager,
                                    state_manager=state_manager,
                                    logger=logger)
        
        async def run():
            logger.info("Starting concurrent tasks")
            
            app_task = asyncio.create_task(RobotControlUI(mode_manager).run_async())
            # Use rclpy.spin_once() in a loop to ensure callbacks are processed
            def spin_node():
                while rclpy.ok():
                    rclpy.spin_once(node)
                    state_manager.spin_subscribers()
            
            node_task = asyncio.create_task(asyncio.to_thread(spin_node))
            await asyncio.gather(app_task, node_task)

        await run()

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise
    finally:
        node.destroy_node()
        state_manager.stop_subscription()
        rclpy.shutdown()
        
def main():
    asyncio.run(main_async())
    
main() # For Debug
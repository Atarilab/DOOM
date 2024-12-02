import os
import time
import asyncio
import logging
import argparse
from typing import Optional
import torch

import rclpy
from rclpy.node import Node

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from unitree_sdk2py.utils.crc import CRC

from vicon_receiver.msg import Position

from utils.ui_interface import ModeManager, RobotControlUI
from utils.logger import get_logger 
from utils.initialization import initialize_channel, initialize_robot_controller

from controllers.stand_controller import (
    IdleController, 
    StayDownController, 
    StandUpController, 
    StandDownController, 
    StanceController,
)
from controllers.rl_controller import RLController, RLInitPosController


# I require this try-catch to run in debugging and deployment mode.
try:
    from state_manager.state_manager import (
        StateManager, 
        DDSStateSubscriber, 
        ROS2StateSubscriber
    )
    from state_manager.handlers import *
except:
    from state_manager.state_manager.state_manager import (
        StateManager, 
        DDSStateSubscriber, 
        ROS2StateSubscriber
    )
    from state_manager.state_manager.handlers import *
    


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
        self.logger.debug(active_controller)
        try:
            # Retrieve states from state manager
            combined_state = self.state_manager.get_combined_state()
            combined_state["elapsed_time"] = time.time()
            
            # Compute motor commands
            with torch.no_grad():
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
    parser.add_argument("--task",  type=str, default="rl-velocity-sim-go2",  help="Task name to run")
    parser.add_argument("--log",  default="test", type=str,  help="Experiment name to log information")
    parser.add_argument("--debug", action="store_true", help="Show debug logs")
    parser.add_argument("--sim", default=True, action="store_true", help="Run in simulation mode")
    
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
        #######################################################
        # TODO: Move Individual State Handlers Inside its class and handle them properly
        #######################################################

        dds_low_state_sub = DDSStateSubscriber(
            topic="rt/lowstate", 
            msg_type=LowState_, 
            handler_func=low_state_handler,
            logger=logger
        )
        state_manager.add_subscriber("low_state", dds_low_state_sub)
        
        # ros2_low_state_sub = ROS2StateSubscriber(
        #     topic="/lowstate", 
        #     node_name="low_state",
        #     msg_type=LowState, 
        #     handler_func=low_state_handler
        # )
        # state_manager.add_subscriber("low_state", ros2_low_state_sub)

        # comment out when not connected to vicon
        if args.sim:
            # use sports state
            dds_low_state_sub = DDSStateSubscriber(
                topic="rt/sportmodestate", 
                msg_type=SportModeState_, 
                handler_func=sport_states_handler,
                logger=logger
            )
            state_manager.add_subscriber("sport_state_sim", dds_low_state_sub)
        else:
            # use vicon
            ros2_vicon_sub = ROS2StateSubscriber(
                topic="/vicon/Go2/Go2", 
                node_name="vicon",
                msg_type=Position, 
                handler_func=vicon_handler,
                logger=logger
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
        
        mode_manager.register_mode('RL', {
            'RL': RLController(configs['robot_config'], policy_path="policies/policy.pt"),
            'RLINITPOS': RLInitPosController(configs['robot_config']),
            
        })

        # NOTE There is a bug where lower-case submodes are not recognized due to https://github.com/Atarilab/DOOM/blob/9be6e3a0151ba4f6c634c1473940a9ebc19aa116/src/utils/ui_interface.py#L314
        
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
        print("Shutting down node...")
        node.destroy_node()
        state_manager.stop_subscription()
        rclpy.shutdown()
        
def main():
    print("Starting node")
    asyncio.run(main_async())
    
main() # required for debugging
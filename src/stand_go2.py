import asyncio
import os
import argparse
import logging
import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_
from unitree_sdk2py.utils.crc import CRC

from tasks.task_configs import TASK_CONFIG
from utils.config_loader import load_config
from utils.ui_interface import *
from controllers.stand_controller import *
from controllers.rl_controller import RLController
from controllers.mpc_controller import ModelPredictiveController

# Constants
UPDATE_INTERVAL = 0.002  # 2ms control loop
MAX_MOTORS = 12

class RobotState:
    def __init__(self):
        self.mode: RobotMode = RobotMode.IDLE
        self.standing_state: StandingState = StandingState.IDLE
        self.rl_gait: RLGait = RLGait.TROT
        # self.mpc_gait: MPCGait = MPCGait.TROT
        self.start_time: float = asyncio.get_event_loop().time()
        self.motor_commands: Dict[int, MotorCommand] = {
            i: MotorCommand() for i in range(MAX_MOTORS)
        }
        self.active_controller = None

class RobotController:
    def __init__(self, config: dict):
        self.state = RobotState()
        self.crc = CRC()
        
        # Initialize controllers with new RL and MPC controllers
        self.controllers = {
            RobotMode.IDLE: IdleController(),
            RobotMode.STANDING: {
                StandingState.IDLE: IdleController(),
                StandingState.STAND_UP: StandUpController(config),
                StandingState.STAND_DOWN: StandDownController(config),
                StandingState.STAY_DOWN: StayDownController(config),
            },
            RobotMode.STANCE: StanceController(config),
            RobotMode.RL: {
                RLGait.TROT: RLController(),
                RLGait.WALK: RLController(),
                RLGait.GALLOP: RLController(),
                RLGait.BOUND: RLController(),
            },
        }
        
        self.state.active_controller = self.controllers[RobotMode.IDLE]
        self.cmd = unitree_go_msg_dds__LowCmd_()
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
        

    def set_mode(self, mode: RobotMode):
        """Set robot mode and update active controller."""
        self.state.mode = mode
        self.state.start_time = asyncio.get_event_loop().time()
        
        if mode == RobotMode.RL:
            self.state.active_controller = self.controllers[mode][self.state.rl_gait]
        # elif mode == RobotMode.MPC:
        #     self.state.active_controller = self.controllers[mode][self.state.mpc_gait]
        elif mode != RobotMode.STANDING:
            self.state.active_controller = self.controllers[mode]

    def set_rl_gait(self, gait: RLGait):
        """Set RL gait and update controller if in RL mode."""
        self.state.rl_gait = gait
        if self.state.mode == RobotMode.RL:
            self.state.active_controller = self.controllers[RobotMode.RL][gait]
            self.state.start_time = asyncio.get_event_loop().time()

        
    def set_standing_state(self, standing_state: StandingState):
        """Set standing state and update active controller."""
        self.state.standing_state = standing_state
        self.state.start_time = asyncio.get_event_loop().time()
        self.state.active_controller = self.controllers[RobotMode.STANDING][standing_state]

    async def run_control_loop(self, publisher: ChannelPublisher):
        """Main control loop."""
        while True:
            loop_start = asyncio.get_event_loop().time()
            
            # Use the active controller directly instead of selecting based on mode
            state = {
                "elapsed_time": asyncio.get_event_loop().time() - self.state.start_time
            }
            motor_commands = self.state.active_controller.compute_torques(state, {})
            
            # Update command structure
            for i in range(MAX_MOTORS):
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
    parser = argparse.ArgumentParser(description="Robot Controller")
    parser.add_argument("--task", type=str, required=True, help="Task name to run")
    args = parser.parse_args()

    # Load configurations
    if args.task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {args.task}")
    
    task_configs = TASK_CONFIG[args.task]
    controller_config = load_config("controller", task_configs["controller_config"])
    robot_interface_config = load_config("robot_interfaces", task_configs["robot_interface_config"])
    robot_config = load_config("robot", task_configs["robot_config"])

    # Initialize robot network interface
    if "sim" in args.task:
        ChannelFactoryInitialize(
            robot_interface_config["DOMAIN_ID"],
            robot_interface_config["INTERFACE"]
        )
    else:
        ChannelFactoryInitialize(0, os.environ.get('NETWORK_INTERFACE'))

    # Initialize communication channels
    publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    publisher.Init()
    
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(None, 10)

    # Initialize controller
    robot_controller = RobotController(robot_config)
    
    # Create and run UI
    app = RobotControlUI(robot_controller)
    
    try:
        # Run control loop and UI concurrently
        await asyncio.gather(
            robot_controller.run_control_loop(publisher),
            app.run_async()
        )
    finally:
        logging.info("Shutting down gracefully...")

        for i in range(20):
            robot_controller.cmd.motor_cmd[i].mode = 0x01
            robot_controller.cmd.motor_cmd[i].q = 0.0
            robot_controller.cmd.motor_cmd[i].kp = 0.0
            robot_controller.cmd.motor_cmd[i].dq = 0.0
            robot_controller.cmd.motor_cmd[i].kd = 0.0
            robot_controller.cmd.motor_cmd[i].tau = 0.0

        robot_controller.cmd.crc = robot_controller.crc.Crc(robot_controller.cmd)
        await robot_controller.run_control_loop(publisher)
        
        time.sleep(1.0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutting down gracefully...")
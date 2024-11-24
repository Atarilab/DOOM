import time
import os
import sys
import argparse
import threading

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC

from tasks.task_configs import TASK_CONFIG
from utils.config_loader import load_config

from controllers.stand_controller import *

COMMANDS = {
    0: "IDLE",
    1: "STAY_DOWN",  # Example of another mode
    2: "STAND_UP",
    3: "STAND_DOWN",
    9: "STANCE"
}

dt = 0.002
crc = CRC()


# Global variables for user command and timing
command = 1  # Default to IDLE
command_start_time = time.perf_counter()

def handle_user_input():
    global command, command_start_time
    while True:
        print("\nAvailable commands:")
        for cmd_id, cmd_name in COMMANDS.items():
            print(f"{cmd_id}: {cmd_name}")
        try:
            key_command = int(input("Enter your command: ").strip())
            if key_command in COMMANDS:
                command = key_command
                command_start_time = time.perf_counter()
                print(f"Switched to controller: {COMMANDS[command]}")
            else:
                print("Invalid command. Try again.")
        except ValueError:
            print("Please enter a valid number.")


if __name__ == '__main__':

    input("Press enter to start")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="ATARI DOOM Controller")
    parser.add_argument("--task", type=str, required=True, help="Task name to run (e.g., rl-sim, mpc-real).")
    args = parser.parse_args()

    # Load task-specific configurations
    if args.task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {args.task}. Check tasks/task_configs.py for available tasks.")
    
    task_configs = TASK_CONFIG[args.task]

    # Load individual configurations
    controller_config = load_config("controller", task_configs["controller_config"])
    robot_interface_config = load_config("robot_interfaces", task_configs["robot_interface_config"])
    robot_config = load_config("robot", task_configs["robot_config"])

    # Initialize controllers
    controllers = {
        0: IdleController(),
        1: StayDownController(robot_config),
        2: StandUpController(robot_config),
        3: StandDownController(robot_config),
        9: StanceController(robot_config)
    }

    # Robot Network Interface Initialization
    if "sim" in args.task:
        ChannelFactoryInitialize(robot_interface_config["DOMAIN_ID"], robot_interface_config["INTERFACE"])
    else:
        ChannelFactoryInitialize(0, os.environ.get('NETWORK_INTERFACE'))

    input_thread = threading.Thread(target=handle_user_input, daemon=True)
    input_thread.start()

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    lowstate_subscriber.Init(None, 10)

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0

    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

    try:
        # Control loop
        while True:
            step_start = time.perf_counter()

            # State and desired goal (add actual values if needed)
            state = {"elapsed_time": time.perf_counter() - command_start_time}
            desired_goal = {}

            # Get the controller
            current_controller = controllers[command]

            # Compute commands using the controller
            motor_commands = current_controller.compute_torques(state, desired_goal)

            # Update cmd structure
            for i in range(12):
                motor = motor_commands[f'motor_{i}']
                cmd.motor_cmd[i].q = motor['q']
                cmd.motor_cmd[i].kp = motor['kp']
                cmd.motor_cmd[i].dq = motor['dq']
                cmd.motor_cmd[i].kd = motor['kd']
                cmd.motor_cmd[i].tau = motor['tau']

            cmd.crc = crc.Crc(cmd)
            pub.Write(cmd)

            time_until_next_step = dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        # Default to IdleController
        idle_controller = controllers[0]
        state = {}  # Provide necessary state if needed
        desired_goal = {}

        # Compute idle commands and send them
        motor_commands = idle_controller.compute_torques(state, desired_goal)
        for i in range(12):
            motor = motor_commands[f'motor_{i}']
            cmd.motor_cmd[i].q = motor['q']
            cmd.motor_cmd[i].kp = motor['kp']
            cmd.motor_cmd[i].dq = motor['dq']
            cmd.motor_cmd[i].kd = motor['kd']
            cmd.motor_cmd[i].tau = motor['tau']

        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)


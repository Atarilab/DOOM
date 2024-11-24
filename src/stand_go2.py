import time
import sys
import threading

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC

from controllers.stand_controller import IdleController, StandUpController, StandDownController, StayDownController

COMMANDS = {
    0: "IDLE",
    1: "STAY_DOWN",  # Example of another mode
    2: "STAND_UP",
    3: "STAND_DOWN",
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
            command = int(input("Enter your command: ").strip())
            if command in COMMANDS:
                command = command
                command_start_time = time.perf_counter()
                print(f"Switched to controller: {COMMANDS[command]}")
            else:
                print("Invalid command. Try again.")
        except ValueError:
            print("Please enter a valid number.")


if __name__ == '__main__':

    input("Press enter to start")

    # Initialize controllers
    controllers = {
        0: IdleController(),
        1: StayDownController(),
        2: StandUpController(),
        3: StandDownController(),
    }

    # Robot Network Interface Initialization
    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

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
            motor_commands = current_controller.compute_command(state, desired_goal)

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
        motor_commands = idle_controller.compute_command(state, desired_goal)
        for i in range(12):
            motor = motor_commands[f'motor_{i}']
            cmd.motor_cmd[i].q = motor['q']
            cmd.motor_cmd[i].kp = motor['kp']
            cmd.motor_cmd[i].dq = motor['dq']
            cmd.motor_cmd[i].kd = motor['kd']
            cmd.motor_cmd[i].tau = motor['tau']

        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)


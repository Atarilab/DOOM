import argparse
import asyncio
import os
import time

import rclpy
import gc
gc.set_threshold(10**9, 10**9, 10**9)

# DOOM Imports
from controllers.stand_controller import DampingController, ZeroTorqueController
from master_manager.low_level_cmd_publisher import LowLevelCmdPublisher
from robots import resolve_robot
from state_manager.state_manager import StateManager
from utils.initialization import initialize_channel, initialize_robot_controller
from utils.logger import get_logger
from utils.mode_manager import ModeManager
from utils.ui_interface import RobotControlUI

node = None
state_manager = None

import gc

gc.set_threshold(10**9, 10**9, 10**9)


async def main_async(args=None):
    """
    Asynchronous entry point for initializing and running the DOOM robot controller system.

    - Parses command-line arguments for task selection, logging directory, and debug options.
    - Sets up logging, communication channels (sim/real), and configurations for the controller.
    - Instantiates the robot model which has subscribers and controllers desired for the specified task.
    - Registers controllers desired for the specified task with the mode manager.
    - Registers subscribers desired for the specified task with the state manager.
    - Launches the low-level command publisher and the user interface concurrently.

    Args:
        args: Optional command-line arguments to override defaults.
    """
    global node
    global state_manager
    rclpy.init(args=args)

    # Parse arguments
    parser = argparse.ArgumentParser(description="ATARI DOOM Robot Controller")
    parser.add_argument("--task", type=str, default="rl-velocity-sim-go2", help="Task name to run")
    parser.add_argument("--log", type=str, default="test", help="Experiment name to log information")
    parser.add_argument("--debug", action="store_true", help="Show debug logs")
    parser.add_argument("--ui", action="store_true", help="Enable the Robot Control UI")

    args = parser.parse_args()

    # Setup logger
    log_file = os.path.join("logs", args.log, f"{args.task}_robot_controller.log")
    logger = get_logger(f"{args.task}_robot_controller", log_file, debug=args.debug)

    try:
        logger.info("Starting robot controller for task: %s", args.task)

        # Load configurations
        configs = await initialize_robot_controller(args.task, logger)

        # Add debug flag to configs
        configs["debug"] = args.debug

        # Initialize communication channel
        await initialize_channel(args.task, configs["robot_interface_config"], logger)

        # Initialize state manager - responsible for handling ROS/DDS raw messages
        state_manager = StateManager(logger=logger)

        # Automatically resolve robot class from task name
        robot = resolve_robot(args.task, logger, device=configs["controller_config"]["device"])

        # Add subscribers desired for the specified task to state manager from robot model
        for name, subscriber in robot.subscribers.items():
            state_manager.add_subscriber(name, subscriber)

        # Create mode manager and register idle (damping) controller
        mode_manager = ModeManager(logger=logger, device=configs["controller_config"]["device"])
        mode_manager.register_mode("ZERO", {"default": ZeroTorqueController(robot, configs)})
        mode_manager.register_mode("DAMPING", {"default": DampingController(robot, configs)})

        # Register controllers available for the robot
        for controller_type, controllers in robot.available_controllers.items():
            controller_dict = {
                controller_name: controller_class(robot=robot, configs=configs, debug=args.debug)
                for (controller_name, controller_class) in controllers.items()
            }
            mode_manager.register_mode(controller_type, controller_dict)

        # Set ZERO mode by default
        mode_manager.set_mode("ZERO")

        # Create low level command publisher node
        node = LowLevelCmdPublisher(
            dt=configs["controller_config"]["control_dt"],
            robot=robot,
            mode_manager=mode_manager,
            state_manager=state_manager,
            logger=logger,
            debug=args.debug,
        )

        logger.info("Starting concurrent tasks")

        # Spin state manager at twice the control frequency (sensor frequency)
        def spin_node():
            sensor_dt = 0.5 * configs["controller_config"]["control_dt"]
            next_time = time.time() + sensor_dt

            while rclpy.ok():
                state_manager.spin_subscribers()

                current_time = time.time()
                sleep_time = next_time - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                next_time += sensor_dt

        # Run node in separate thread
        node_task = asyncio.create_task(asyncio.to_thread(spin_node))

        # Create robot control UI if not disabled (non-blocking)
        try:
            if args.ui:
                app_task = asyncio.create_task(RobotControlUI(mode_manager, task_name=args.task).run_async())
                app_task.add_done_callback(lambda t: t.result() if not t.cancelled() else None)

            # Only wait for the node task
            await node_task

        except asyncio.CancelledError:
            mode_manager.set_mode("DAMPING")
            logger.info("Cancelling all tasks in master node before shutdown..")
            raise

    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt. Shutting down...")

        # Set controller to DAMPING mode before shutting down
        mode_manager.set_mode("DAMPING")
        logger.info("Set controller to DAMPING mode before shutdown")

    except Exception as e:
        logger.exception("An error occurred: %s", e)
        raise
    finally:
        if node:
            node.cleanup()
        if state_manager:
            state_manager.destroy_subscribers()
        rclpy.shutdown()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

import argparse
import asyncio
import os
import time

import rclpy

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
    parser.add_argument("--enable-ui", action="store_true", help="Enable the Robot Control UI")

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
        robot = resolve_robot(args.task, logger)

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
                controller_name: controller_class(robot, configs)
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

        async def run():
            logger.info("Starting concurrent tasks")

            # Create robot control UI if not disabled
            app_task = (
                asyncio.create_task(RobotControlUI(mode_manager, task_name=args.task).run_async())
                if args.enable_ui
                else None
            )

            # Use rclpy.spin_once() in a loop to ensure callbacks are processed
            def spin_node():
                spin_dt = 0.00025  # you could increase this value to reduce the frequency of the callbacks (less CPU utilization)
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
                if app_task is not None:
                    await asyncio.gather(app_task, node_task)
                else:
                    await node_task
            except asyncio.CancelledError:
                logger.info("Tasks were cancelled")
                raise

        await run()
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
            node.destroy_node()
        if state_manager:
            state_manager.destroy_subscribers()
        rclpy.shutdown()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

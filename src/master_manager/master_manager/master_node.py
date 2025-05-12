import argparse
import asyncio
import os
import time

import rclpy
from controllers.rl_controller import RLLocomotionVelocityController
from controllers.rl_contact_controller import RLLocomotionContactController
from controllers.stand_controller import (
    IdleController,
    StanceController,
    StandDownController,
    StandUpController,
    StayDownController,
)
from state_manager.msg_handlers import low_state_handler, sport_mode_state_handler, vicon_handler
from state_manager.state_manager import DDSStateSubscriber, ROS2StateSubscriber, StateManager

# Unitree DDS
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
from utils.initialization import initialize_channel, initialize_robot_controller
from utils.logger import get_logger
from utils.mj_wrapper import MjQuadRobotWrapper

# DOOM Imports
from utils.ui_interface import ModeManager, RobotControlUI
from master_manager.low_level_cmd_publisher import LowLevelCmdPublisher
from robots.go2.go2 import Go2

node = None
state_manager = None

async def main_async(args=None):
    """Main asynchronous entry point for robot controller."""
    global node
    global state_manager
    rclpy.init(args=args)

    # Parse arguments
    parser = argparse.ArgumentParser(description="ATARI DOOM Robot Controller")
    parser.add_argument("--task", type=str, default="rl-velocity-sim-go2", help="Task name to run")
    parser.add_argument("--log", type=str, default="test", help="Experiment name to log information")
    parser.add_argument("--debug", action="store_true", help="Show debug logs")

    args = parser.parse_args()

    # Setup logger
    log_file = os.path.join("logs", args.log, f"{args.task}_robot_controller.log")
    logger = get_logger(f"{args.task}_robot_controller", log_file, debug=args.debug)

    try:
        logger.info(f"Starting robot controller for task: {args.task}")

        # Load configurations
        configs = await initialize_robot_controller(args.task, logger)

        # Initialize communication channel
        await initialize_channel(args.task, configs["robot_interface_config"], logger)

        state_manager = StateManager(logger=logger)

        dds_low_state_sub = DDSStateSubscriber(
            topic="rt/lowstate",
            msg_type=LowState_,
            handler_func=low_state_handler,
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
            from vicon_receiver.msg import Position

            ros2_vicon_sub = ROS2StateSubscriber(
                # topic="/vicon/Go2with6markers/Go2with6markers",
                topic="/vicon/Go2/Go2",
                node_name="vicon_state",
                msg_type=Position,
                handler_func=vicon_handler,
                logger=logger,
            )
            state_manager.add_subscriber("vicon_state", ros2_vicon_sub)
            
        robot = Go2()

        # Create mode manager and register controllers
        mode_manager = ModeManager(logger=logger)
        mode_manager.register_mode("IDLE", {"default": IdleController(robot, configs)})

        mode_manager.register_mode(
            "STANDING",
            {
                "STAY_DOWN": StayDownController(robot, configs),
                "STAND_UP": StandUpController(robot, configs),
                "STAND_DOWN": StandDownController(robot, configs),
            },
        )

        mode_manager.register_mode("STANCE", {"STANCE": StanceController(robot, configs)})
        if "rl-contact" in args.task:
            mode_manager.register_mode(
                "RL-CONTACT",
                {
                    "STANCE": StanceController(robot, configs),
                    "RL-CONTACT": RLLocomotionContactController(robot=robot, configs=configs),
                },
            )

        if "rl-velocity" in args.task:
            mode_manager.register_mode(
                "RL-VELOCITY",
                {
                    "STANCE": StanceController(robot, configs),
                    "RL-VELOCITY": RLLocomotionVelocityController(robot=robot, configs=configs),
                },
            )
        mode_manager.set_mode("IDLE")

        node = LowLevelCmdPublisher(
            dt=configs["controller_config"]["control_dt"],
            robot=robot,
            mode_manager=mode_manager,
            state_manager=state_manager,
            logger=logger,
        )

        async def run():
            logger.info("Starting concurrent tasks")

            app_task = asyncio.create_task(RobotControlUI(mode_manager).run_async())

            # Use rclpy.spin_once() in a loop to ensure callbacks are processed
            def spin_node():
                spin_dt = 0.0001
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
                await asyncio.gather(app_task, node_task)
            except asyncio.CancelledError:
                logger.info("Tasks were cancelled")
                raise

        await run()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt. Shutting down...")

        # Set controller to idle mode before shutting down
        mode_manager.set_mode("IDLE")
        logger.info("Set controller to IDLE mode before shutdown")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
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

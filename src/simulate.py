import argparse
import time
import os

from tasks.task_configs import TASK_CONFIG
from utils.config_loader import load_config
from utils.logger import get_logger

from robot_interfaces.sim_robot_interface import SimRobotInterface

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="ATARI DOOM Mujoco Simulator")
    parser.add_argument("--task", type=str, default="rl-velocity-sim-go2", help="Task name to run (e.g., rl-sim, mpc-real).")
    parser.add_argument("--log", type=str, default="test", help="Experiment name to log information")
    args = parser.parse_args()

    # Load task-specific configurations
    if args.task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {args.task}. Check tasks/task_configs.py for available tasks.")
    
    task_configs = TASK_CONFIG[args.task]

    # Load individual configurations
    robot_interface_config = load_config(task_configs["robot_interface"])

    # Setup logger with a more specific log file path
    log_file = os.path.join('logs', args.log, f"{args.task}_simulate.log")
    logger = get_logger(f"{args.task}_simulate", log_file)
    logger.info(f"Task Name: {args.task}")
    logger.info(f"Robot Config: {task_configs['robot_interface']}")

    # Initialize robot interface
    robot_interface = SimRobotInterface(robot_interface_config)
    logger.info(f"Starting Simulation.")
    # Execution loop
    try:
        for step in range(10000000):
            time.sleep(robot_interface.simulate_dt)  # Sync with simulation timestep
            
    except KeyboardInterrupt:
        logger.info("Execution interrupted, stopping simulation.")

    finally:
        robot_interface.stop(logger)
        

if __name__ == "__main__":
    main()

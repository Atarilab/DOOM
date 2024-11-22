import argparse
import time
from tasks.task_configs import TASK_CONFIG
from utils.config_loader import load_config
from utils.logger import get_logger

from controllers.rl_controller import RLController
from controllers.mpc_controller import ModelPredictiveController
from robot_interfaces.sim_robot_interface import SimRobotInterface
from robot_interfaces.real_robot_interface import RealRobotInterface

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="ATARI DOOM")
    parser.add_argument("--task", type=str, required=True, help="Task name to run (e.g., rl-sim, mpc-real).")
    parser.add_argument("--log", type=str, required=True, help="Experiment name to log information")
    args = parser.parse_args()

    # Load task-specific configurations
    if args.task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {args.task}. Check doom.tasks/task_configs.py for available tasks.")
    
    task_configs = TASK_CONFIG[args.task]

    # Load individual configurations
    controller_config = load_config("controllers", task_configs["controller_config"].split("/")[-1])
    robot_interface_config = load_config("robot_interfaces", task_configs["robot_interface_config"].split("/")[-1])

    logger = get_logger(args.log)
    logger.info(f"Task Name: {args.task}")
    logger.info(f"Controller Config: {task_configs['controller_config']}")
    logger.info(f"Robot Config: {task_configs['robot_interface_config']}")
    # logger.info(f"Environment Config: {task_configs['environment_config']}")

    # Initialize robot interface
    if "sim" in args.task:
        robot_interface = SimRobotInterface(robot_interface_config)
    elif "real" in args.task:
        robot_interface = RealRobotInterface(robot_interface_config["ros_node_name"])
    else:
        raise ValueError("Invalid robot interface type detected in task configuration.")

    # # Initialize controller
    # if "rl" in args.task:
    #     controller = RLController()
    # elif "mpc" in args.task:
    #     controller = ModelPredictiveController(
    #         model=None, 
    #         horizon=controller_config["horizon"], 
    #         cost_function=lambda x: sum(x**2)  # Example
    #     )
    # else:
    #     raise ValueError("Invalid controller type detected in task configuration.")

    # Execution loop
    try:
        for step in range(10000):
            # state = robot_interface.receive_state()
            # command = controller.compute_command(state, desired_goal={"goal": [0, 0, 1]})
            # robot_interface.send_command(command)
            logger.info(f"Step {step}: Command sent.")
            time.sleep(robot_interface.simulate_dt)  # Sync with simulation timestep
            

    except KeyboardInterrupt:
        logger.info("Execution interrupted, stopping simulation.")
    finally:
        robot_interface.stop(logger)
        

if __name__ == "__main__":
    main()

import argparse
import os
import threading
from threading import Thread
import time

import mujoco
import mujoco.viewer
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from robot_interfaces.sim_robot_interface import ElasticBand, SimRobotInterface
from tasks.task_configs import TASK_CONFIG
from utils.config_loader import load_config
from utils.logger import get_logger
from utils.unitree_sdk2py_bridge import UnitreeSdk2Bridge


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="ATARI DOOM Mujoco Simulator")
    parser.add_argument(
        "--task",
        type=str,
        default="rl-velocity-sim-go2",
        help="Task name to run (e.g., rl-sim, mpc-real).",
    )
    parser.add_argument("--log", type=str, default="test", help="Experiment name to log information")
    args = parser.parse_args()

    # Load task-specific configurations
    if args.task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {args.task}. Check tasks/task_configs.py for available tasks.")

    task_configs = TASK_CONFIG[args.task]

    # Load individual configurations
    robot_interface_config = load_config(task_configs["robot_interface"])

    # Setup logger with a more specific log file path
    log_file = os.path.join("logs", args.log, f"{args.task}_simulate.log")
    logger = get_logger(f"{args.task}_simulate", log_file)
    logger.info(f"Task Name: {args.task}")
    logger.info(f"Robot Config: {robot_interface_config}")

    # Initialize Mujoco model and data
    robot_scene = os.path.join(os.getcwd(), "robots", robot_interface_config["ROBOT"], robot_interface_config["SCENE"])
    mj_model = mujoco.MjModel.from_xml_path(robot_scene)
    mj_data = mujoco.MjData(mj_model)

    # Set timestep
    mj_model.opt.timestep = robot_interface_config["SIMULATION_DT"]

    # Check for object body in the scene
    object_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "object")

    # Elastic band setup
    use_elastic_band = robot_interface_config.get("ELASTIC_BAND", False)
    elastic_band = ElasticBand() if use_elastic_band else None
    band_attached_link = None

    # Initialize viewer
    viewer_kwargs = {
        "show_left_ui": False,
        "show_right_ui": False,
    }
    if use_elastic_band and elastic_band is not None:
        base_link_name = robot_interface_config["BASE_LINK"]
        band_attached_link = mj_model.body(base_link_name).id
        viewer_kwargs["key_callback"] = elastic_band.MujuocoKeyCallback

    viewer = mujoco.viewer.launch_passive(mj_model, mj_data, **viewer_kwargs)

    # Threading lock
    locker = threading.Lock()

    def SimulationThread():
        """Simulation thread function"""
        ChannelFactoryInitialize(robot_interface_config["DOMAIN_ID"], robot_interface_config["INTERFACE"])
        unitree = UnitreeSdk2Bridge(mj_model, mj_data, robot=robot_interface_config["ROBOT"], object=object_id)

        # Set fixed base position if configured
        fixed_base_pos = robot_interface_config.get("FIXED_BASE_POS", None)
        if fixed_base_pos is not None:
            base_link_name = robot_interface_config["BASE_LINK"]
            base_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, base_link_name)
            if base_body_id >= 0:
                mj_data.qpos[:3] = np.array(fixed_base_pos)
                mj_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
                mj_data.qvel[:3] = np.zeros(3)
                mj_data.qvel[3:6] = np.zeros(3)

        if robot_interface_config.get("PRINT_SCENE_INFORMATION", False):
            unitree.PrintSceneInformation()

        while viewer.is_running():
            step_start = time.perf_counter()

            locker.acquire()

            if use_elastic_band and elastic_band is not None and elastic_band.enable:
                force = elastic_band.Advance(mj_data.qpos[:3], mj_data.qvel[:3])
                mj_data.xfrc_applied[band_attached_link][:3] = force

            mujoco.mj_step(mj_model, mj_data)

            locker.release()

            time_until_next_step = mj_model.opt.timestep - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def PhysicsViewerThread():
        """Viewer thread function"""
        # Enable rendering of world frame axes
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD

        while viewer.is_running():
            locker.acquire()

            # Camera tracking (similar to SimRobotInterface)
            base_link_name = robot_interface_config["BASE_LINK"]
            robot_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, base_link_name)
            if robot_body_id >= 0 and robot_interface_config.get("FIXED_BASE_POS", None) is None:
                robot_pos = mj_data.xpos[robot_body_id]
                viewer.cam.lookat[:] = robot_pos
                viewer.cam.distance = 3.0
                viewer.cam.elevation = -20

            viewer.sync()
            locker.release()
            time.sleep(robot_interface_config["VIEWER_DT"])

    # Start threads
    logger.info("Starting Simulation.")
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()


if __name__ == "__main__":
    main()

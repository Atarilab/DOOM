import logging
import os
from typing import Dict

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from tasks.task_configs import TASK_CONFIG
from utils.config_loader import load_config


async def initialize_robot_controller(task: str, logger: logging.Logger) -> Dict:
    """Initialize robot controller configurations."""
    if task not in TASK_CONFIG:
        logger.error(f"Unknown task: {task}")
        raise ValueError(f"Unknown task: {task}")

    task_configs = TASK_CONFIG[task]
    logger.info("Configurations loaded successfully")

    return {
        "controller_config": load_config(task_configs["controller"]),
        "robot_interface_config": load_config(task_configs["robot_interface"]),
    }


async def initialize_channel(task: str, robot_interface_config: Dict, logger: logging.Logger):
    """Initialize communication channel based on task type."""
    if "sim" in task:  # TODO: Might interfere with other text other than simulation
        logger.info(
            f"Initializing channel with Domain ID: {robot_interface_config['DOMAIN_ID']} "
            f"and network interface: {robot_interface_config['INTERFACE']}"
        )
        ChannelFactoryInitialize(robot_interface_config["DOMAIN_ID"], robot_interface_config["INTERFACE"])
    else:
        network_interface = os.environ.get("NETWORK_INTERFACE")
        logger.info(f"Initializing channel with Domain ID: 0 and network interface: {network_interface}")
        ChannelFactoryInitialize(0, network_interface)

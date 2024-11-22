# utils/config_loader.py
import yaml
import os

def load_config(config_type, config_name):
    """
    Load configuration file based on type and name.

    Args:
        config_type (str): Type of configuration (e.g., 'controllers', 'robot_interfaces').
        config_name (str): Name of the configuration file (e.g., 'rl_controller_cfg.yaml').

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    base_path = os.path.join(os.path.dirname(__file__), '..')
    config_path = os.path.join(base_path, config_type, "config", config_name)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

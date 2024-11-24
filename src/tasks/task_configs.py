# task_mapping.py

TASK_CONFIG = {
    "rl-velocity-sim": {
        "controller_config": "controllers/config/rl_velocity_cfg.yaml",
        "robot_interface_config": "robot_interfaces/config/sim_cfg.yaml",
        "robot_config": "robots/go2/config.yaml",
    },
    "mpc-real": {
        "controller_config": "controllers/config/mpc_controller_cfg.yaml",
        "robot_interface_config": "robot_interfaces/config/real_robot_cfg.yaml",
        "environment_config": "environments/config/ros2_env_cfg.yaml",
    },
    # Add more tasks here as needed
}

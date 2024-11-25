# task_mapping.py

TASK_CONFIG = {
    "rl-velocity-sim-go2": {
        "controller_config": "controllers/config/rl_velocity_cfg.yaml",
        "robot_interface_config": "robot_interfaces/config/sim_cfg.yaml",
        "robot_config": "robots/go2/config.yaml",
    },
    "rl-velocity-real-go2": {
        "controller_config": "controllers/config/rl_velocity_cfg.yaml",
        "robot_interface_config": "robot_interfaces/config/real_cfg.yaml",
        "robot_config": "robots/go2/config.yaml",
    },
    "mpc-velocity-real-go2": {
        "controller_config": "controllers/config/mpc_controller_cfg.yaml",
        "robot_interface_config": "robot_interfaces/config/real_robot_cfg.yaml",
        "robot_config": "robots/go2/config.yaml",
    },
    # Add more tasks here as needed
}

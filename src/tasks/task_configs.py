
TASK_CONFIG = {
    "rl-velocity-sim-go2": {
        "controller": "controllers/config/rl_velocity_cfg.yaml",
        "robot_interface": "robot_interfaces/config/sim_cfg.yaml",
        "robot": "robots/go2/config.yaml",
    },
    "rl-velocity-real-go2": {
        "controller": "controllers/config/rl_velocity_cfg.yaml",
        "robot_interface": "robot_interfaces/config/real_cfg.yaml",
        "robot": "robots/go2/config.yaml",
    },
    "mpc-velocity-real-go2": {
        "controller": "controllers/config/mpc_controller_cfg.yaml",
        "robot_interface": "robot_interfaces/config/real_cfg.yaml",
        "robot": "robots/go2/config.yaml",
    },
    # Add more tasks here as needed
}

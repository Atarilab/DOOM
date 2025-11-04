TASK_CONFIG = {
    #########################################################
    # UnitreeGo2
    #########################################################
    "rl-velocity-sim-go2": {
        "controller": "controllers/config/rl_velocity_go2_cfg.yaml",
        "robot_interface": "robot_interfaces/config/sim_go2_cfg.yaml",
    },
    "rl-velocity-real-go2": {
        "controller": "controllers/config/rl_velocity_go2_cfg.yaml",
        "robot_interface": "robot_interfaces/config/real_go2_cfg.yaml",
    },
    "rl-contact-sim-go2": {
        "controller": "controllers/config/rl_contact_go2_cfg.yaml",
        "robot_interface": "robot_interfaces/config/sim_go2_cfg.yaml",
    },
    "rl-contact-real-go2": {
        "controller": "controllers/config/rl_contact_go2_cfg.yaml",
        "robot_interface": "robot_interfaces/config/real_go2_cfg.yaml",
    },
    #########################################################
    # UnitreeG1
    #########################################################
    "rl-contact-sim-g1": {
        "controller": "controllers/config/rl_contact_g1_cfg.yaml",
        "robot_interface": "robot_interfaces/config/sim_box_g1_cfg.yaml",
    },
    "rl-manicont-sim-g1": {
        "controller": "controllers/config/rl_manicont_g1_cfg.yaml",
        "robot_interface": "robot_interfaces/config/sim_box_g1_cfg.yaml",
    },
    "rl-manicont-real-g1": {
        "controller": "controllers/config/rl_manicont_g1_cfg.yaml",
        "robot_interface": "robot_interfaces/config/real_g1_cfg.yaml",
    },
    "rl-manicont-sim-g1_fixed": {
        "controller": "controllers/config/rl_manicont_g1_cfg.yaml",
        "robot_interface": "robot_interfaces/config/sim_box_g1fixed_cfg.yaml",
    },
    # "rl-contact-real-g1": {
    #     "controller": "controllers/config/rl_contact_g1_cfg.yaml",
    #     "robot_interface": "robot_interfaces/config/real_g1_cfg.yaml",
    # },
    "rl-velocity-sim-g1": {
        "controller": "controllers/config/rl_velocity_g1_cfg.yaml",
        "robot_interface": "robot_interfaces/config/sim_flat_g1_cfg.yaml",
    },
    "rl-velocity-real-g1": {
        "controller": "controllers/config/rl_velocity_g1_cfg.yaml",
        "robot_interface": "robot_interfaces/config/real_g1_cfg.yaml",
    },
    "rl-velocity-sim-g1_lower": {
        "controller": "controllers/config/rl_velocity_g1lower_cfg.yaml",
        "robot_interface": "robot_interfaces/config/sim_flat_g1lower_cfg.yaml",
    },
    "rl-unitree-sim-g1": {
        "controller": "controllers/config/rl_unitree_g1_cfg.yaml",
        "robot_interface": "robot_interfaces/config/sim_flat_g1_cfg.yaml",
    },
    "rl-reach-sim-g1": {
        "controller": "controllers/config/rl_reach_g1_cfg.yaml",
        "robot_interface": "robot_interfaces/config/sim_flat_g1_cfg.yaml",
    },
    "rl-reach-real-g1": {
        "controller": "controllers/config/rl_reach_g1_cfg.yaml",
        "robot_interface": "robot_interfaces/config/real_g1_cfg.yaml",
    },
    "rl-waist-sim-g1": {
        "controller": "controllers/config/rl_waist_g1_cfg.yaml",
        "robot_interface": "robot_interfaces/config/sim_flat_g1_cfg.yaml",
    },
    "rl-waist-real-g1": {
        "controller": "controllers/config/rl_waist_g1_cfg.yaml",
        "robot_interface": "robot_interfaces/config/real_g1_cfg.yaml",
    },
    "rl-balance-sim-g1": {
        "controller": "controllers/config/rl_balance_g1_cfg.yaml",
        "robot_interface": "robot_interfaces/config/sim_flat_g1_cfg.yaml",
    },
    "rl-balance-real-g1": {
        "controller": "controllers/config/rl_balance_g1_cfg.yaml",
        "robot_interface": "robot_interfaces/config/real_g1_cfg.yaml",
    },
    "gain-tuning-sim-g1": {
        "controller": "controllers/config/g1_gain_tuning_cfg.yaml",
        "robot_interface": "robot_interfaces/config/sim_flat_g1_cfg.yaml",
    },
    "gain-tuning-real-g1": {
        "controller": "controllers/config/g1_gain_tuning_cfg.yaml",
        "robot_interface": "robot_interfaces/config/real_g1_cfg.yaml",
    },
    # Add more tasks here as needed
}

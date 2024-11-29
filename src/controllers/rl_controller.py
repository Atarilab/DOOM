# controllers/rl_controller.py
import os
import torch
from controllers.controller_base import ControllerBase
from utils.config_loader import load_config
import numpy as np

WS_DIR = os.getcwd()


class RLController(ControllerBase):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """
    def __init__(self, robot_config, policy_path: str):
        self.stand_up_joint_pos = robot_config["STAND_UP_JOINT_POS"]
        self.stand_down_joint_pos = robot_config["STAND_DOWN_JOINT_POS"]
        self.start_time = 0.0
        
        self.policy = torch.jit.load(os.path.join(WS_DIR, "DOOM",policy_path)).cpu() # make sure we stay on cpu

    def compute_torques(self, state, desired_goal=None):
        elapsed_time = state["elapsed_time"]- self.start_time
        phase = np.tanh(elapsed_time / 1.2)
        cmd = {}
        for i in range(12):
            cmd[f'motor_{i}'] = {
                'q': phase * self.stand_up_joint_pos[i] + (1 - phase) * self.stand_down_joint_pos[i],
                'kp': phase * 50.0 + (1 - phase) * 20.0,
                'dq': 0.0,
                'kd': 3.5,
                'tau': 0.0,
            }
        return cmd
    
    # def __init__(self):
        
    #     pass

    # def compute_torques(self, state, desired_goal):
    #     state_tensor = torch.tensor(state['observation'], dtype=torch.float32)
    #     action = self.policy(state_tensor).detach().numpy()

    #     if self.clip_actions:
    #         action = action.clip(-1.0, 1.0)
    #     return {"command": action}

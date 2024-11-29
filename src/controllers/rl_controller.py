# controllers/rl_controller.py
import os
import torch
from controllers.controller_base import ControllerBase
from utils.config_loader import load_config
import numpy as np
from pprint import pprint # debugging only, removable

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class RLController(ControllerBase):
    """
    The Stand Up Controller is used to stand up from the ground. It is an interpolation from the stand down joint positions
    to the stand up joint positions which are constants.
    """
    def __init__(self, robot_config, policy_path: str):
        self.stand_up_joint_pos = robot_config["STAND_UP_JOINT_POS"]
        self.stand_down_joint_pos = robot_config["STAND_DOWN_JOINT_POS"]
        self.start_time = 0.0
        self.policy = torch.jit.load(os.path.join(FILE_DIR, policy_path)).cpu() # make sure we stay on cpu
        
        # states
        self.base_lin_vel = torch.zeros(3)
        self.base_ang_vel = torch.zeros(3)
        self.projected_gravity = torch.zeros(3)
        self.velocity_commands = torch.zeros(3) # includes x-y vel [0,1], and yaw rate [2]
        self.joint_pos = torch.zeros(12)
        self.joint_vel = torch.zeros(12)
        self.actions = torch.zeros(12)
        
        # helpers
        self.gravity_dir = [0, 0, -1]
    
    

        

    def compute_torques(self, state, desired_goal=None):
        
        if "sport_state_sim" in state: # simulation
            self.base_lin_vel = torch.tensor(state["sport_state_sim"]["velocity"])
        elif "vicon_state" in state: # real robot; using vicon
            raise NotImplementedError("Vicon state not implemented yet")
        
        self.base_ang_vel = torch.tensor(state["low_state"]["imu/gyroscope"])
        self.projected_gravity = torch.zeros(3) # TODO
        self.velocity_commands = torch.zeros(3)
        self.joint_pos = torch.tensor(state["low_state"]["motor/joint_pos"]) # TODO check order; transfor to relative values
        self.joint_vel = torch.tensor(state["low_state"]["motor/joint_vel"]) # TODO check order; transfor to relative values
        self.actions = state["last_motor_cmd"]
        
        pprint(state)
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

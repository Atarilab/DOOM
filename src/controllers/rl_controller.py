# controllers/rl_controller.py
import torch
import os
from typing import Dict, Any, Optional
from itertools import chain

from controllers.controller_base import ControllerBase
from utils.config_loader import load_config

from utils.helpers import reorder_robot_states
from state_manager.obs_manager import ObsTerm
from state_manager.observations import joint_pos_rel, joint_vel, lin_vel_b, ang_vel_b, last_action, projected_gravity_b

class RLLocomotionVelocityController(ControllerBase):
    def __init__(self, configs: Dict[str, Any]):
        
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  configs['controller_config']['policy_path'])
        self.policy = torch.jit.load(model_path)
        self.policy.eval()
        self.action_scale = configs['controller_config']['action_scale']
        self.offset = configs['robot_config']['stand_up_joint_pos']
        self.Kp = configs['controller_config']['stiffness']
        self.Kd = configs['controller_config']['damping']
        
        self.last_action = torch.zeros(configs['controller_config']['action_dim'])
        
        
    def register_observations(self):
        """
        Register observations for this controller. Maintain order to be passed directly to policy.
        
        """
        # Register observations using the mode-specific obs_manager
        self.obs_manager.register('lin_vel_b', ObsTerm(lin_vel_b))
        self.obs_manager.register('ang_vel_b', ObsTerm(ang_vel_b))
        self.obs_manager.register('projected_gravity', ObsTerm(projected_gravity_b))
        # self.obs_manager.register('velocity_commands', ObsTerm(velocity_commands)) #Not Implemented
        self.obs_manager.register('joint_pos', ObsTerm(
            joint_pos_rel, 
            params={
                'default_joint_pos': self.offset
            }
        ))
        self.obs_manager.register('joint_vel', ObsTerm(joint_vel))
        self.obs_manager.register('last_action', ObsTerm(last_action, params={"last_action": lambda: self.last_action}))
        
        
    def compute_torques(self, observations, desired_goal):
        
        obs = torch.tensor(list(chain.from_iterable(observations.values())), dtype=torch.float32)      
        # raw_action = self.policy(obs).detach().numpy()
        # self.last_action = raw_action
          
        # # Switch to leg order used by the Unitree low level: FR, FL, RR, RL
        # processed_action = reorder_robot_states(raw_action, 
        #                                         origin_order=['FL', 'FR', 'RL', 'RR'], 
        #                                         target_order=['FR', 'FL', 'RR', 'RL'])
        # joint_pos = reorder_robot_states(state['joint_pos'], 
        #                                         origin_order=['FL', 'FR', 'RL', 'RR'], 
        #                                         target_order=['FR', 'FL', 'RR', 'RL'])
        # joint_vel = reorder_robot_states(state['joint_vel'], 
        #                                         origin_order=['FL', 'FR', 'RL', 'RR'], 
        #                                         target_order=['FR', 'FL', 'RR', 'RL'])
        
        # joint_pos_targets = processed_action * self.action_scale + self.offset
        
        # cmd = {}
        # for i in range(12):
        #     cmd[f'motor_{i}'] = {
        #         'q': joint_pos_targets[i] - joint_pos[i],
        #         'kp': self.Kp,
        #         'dq': 0.0 - joint_vel[i],
        #         'kd': self.Kd,
        #         'tau': 0.0,
        #     }
        # return cmd

        cmd = {}
        for i in range(12):
            cmd[f'motor_{i}'] = {
                'q': self.offset[i],
                'kp': 50.0,
                'dq': 0.0,
                'kd': 3.5,
                'tau': 0.0,
            }
        return cmd
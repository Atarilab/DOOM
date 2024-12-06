# controllers/rl_controller.py
from abc import abstractmethod
import numpy as np
import torch
import os
from typing import Dict, Any
from itertools import chain

from controllers.controller_base import ControllerBase
from utils.helpers import ObservationHistoryStorage


from state_manager.obs_manager import ObsTerm
from state_manager.observations import joint_pos_rel, joint_vel, lin_vel_b, ang_vel_b, last_action, projected_gravity_b, velocity_commands

class BaseRLLocomotionController(ControllerBase):
    def __init__(self, configs: Dict[str, Any]):
        
        super().__init__(configs)
        
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  configs['controller_config']['policy_path'])
        self.policy = torch.jit.load(model_path).cpu()
        self.policy.eval()
        
        self.action_scale = configs['controller_config']['action_scale']
        self.offset = configs['robot_config']['stand_up_joint_pos']
        self.effort_limit = configs['robot_config']['effort_limit']
        self.Kp = configs['controller_config']['stiffness']
        self.Kd = configs['controller_config']['damping']
        self.joint_obs_unitree_to_isaac_mapping = torch.tensor(
            configs['controller_config']["JOINT_OBSERVATION_UNITREE_TO_ISAAC_LAB_MAPPING"]
        )
        self.default_joint_pos = np.array(
            configs['controller_config']["ISAAC_LAB_DEFAULT_JOINT_POS"]
        )
        self.actions_isaac_to_unitree_mapping = np.array(
            configs['controller_config']["JOINT_ACTION_ISAAC_LAB_TO_UNITREE_MAPPING"]
        )
        self.last_action = torch.zeros(configs['controller_config']['action_dim'])
        self.obs_history_storage = ObservationHistoryStorage(
            num_envs=1,
            num_obs=48,
            max_length=1,
            device="cpu",
        )
        self.cmd = {}    
        self.velocity_commands = np.array([0.5, 0.0, 0.0])
        
        
        
    def compute_torques(self, state, desired_goal):
        # Compute observations
        obs = self.obs_manager.compute_observations(state)
        obs = torch.tensor(list(chain.from_iterable(obs.values())), dtype=torch.float32)      
        
        # Process observation history
        self.obs_history_storage.add(obs.unsqueeze(0))
        obs_history = self.obs_history_storage.get()
        
        # Compute actions from policy
        with torch.no_grad():
            raw_action = self.policy(obs_history)
        
        self.last_action = raw_action[0]
        
        # Compute joint position targets from raw actions
        joint_pos_targets = (raw_action * self.action_scale + self.default_joint_pos)[0]
        joint_pos_targets = joint_pos_targets[self.actions_isaac_to_unitree_mapping]
        joint_pos_targets = self._clip_dof_pos(joint_pos_targets)
        # Prepare motor commands with joint position control    
        for i in range(12):
            self.cmd[f'motor_{i}'] = {
                'q': joint_pos_targets[i].detach().numpy(),
                'kp': self.Kp,
                'dq': 0.0,
                'kd': self.Kd,
                'tau': 0.0,

            }
        
        return self.cmd

        ## Future reference for direct torque commands
        # computed_effort = (
        #     self.Kp * (joint_pos_targets - state['joint_pos']) + 
        #     self.Kd * (0.0 - state['joint_vel'])
        # )
        
        # # Clip and prepare torque commands
        # applied_effort = self._clip_effort(computed_effort).detach().numpy()
        
        # # Prepare motor commands with torque control
        # for i in range(12):
        #     self.cmd[f'motor_{i}'] = {
        #         'q': 0.0,  # No position target
        #         'kp': 0.0,  # No position control
        #         'dq': 0.0,  # No velocity target
        #         'kd': 0.0,  # No velocity control
        #         'tau': applied_effort[i],  # Direct torque command
        #     }
        
        
    
class RLLocomotionVelocityController(BaseRLLocomotionController):
    """Velocity-conditioned (contact-implicit) RL Locomotion Controller"""
    
    def __init__(self, configs: Dict[str, Any]):
        super().__init__(configs)
        
        
    def register_observations(self):
        """
        Register observations for this controller. Maintain order to be passed directly to policy.
        """
        # Register observations using the mode-specific obs_manager
        # with self._lock:
        self.obs_manager.register('lin_vel_b', ObsTerm(lin_vel_b))
        self.obs_manager.register('ang_vel_b', ObsTerm(ang_vel_b))
        self.obs_manager.register('projected_gravity', ObsTerm(projected_gravity_b))
        self.obs_manager.register('velocity_commands', ObsTerm(velocity_commands, params={"velocity_commands": lambda: self.velocity_commands}))
        self.obs_manager.register('joint_pos', ObsTerm(
            joint_pos_rel, 
            params={
                'default_joint_pos': self.default_joint_pos,
                'mapping': self.joint_obs_unitree_to_isaac_mapping
            }
        ))
        self.obs_manager.register('joint_vel', ObsTerm(joint_vel,
            params={
                'mapping': self.joint_obs_unitree_to_isaac_mapping
            }
        ))
        self.obs_manager.register('last_action', ObsTerm(last_action, params={"last_action": lambda: self.last_action}))

    
        
class RLLocomotionContactController(BaseRLLocomotionController):
    """Contact-conditioned (contact-explicit) RL Locomotion Controller"""
    
    def __init__(self, configs: Dict[str, Any]):
        super().__init__(configs)
        
        
    def register_observations(self):
        """
        Register observations for this controller. Maintain order to be passed directly to policy.
        """
        
        # Register observations using the mode-specific obs_manager
        # with self._lock:
        self.obs_manager.register('lin_vel_b', ObsTerm(lin_vel_b))
        self.obs_manager.register('ang_vel_b', ObsTerm(ang_vel_b))
        self.obs_manager.register('projected_gravity', ObsTerm(projected_gravity_b))
        self.obs_manager.register('velocity_commands', ObsTerm(velocity_commands, params={"velocity_commands": lambda: self.velocity_commands}))
        self.obs_manager.register('joint_pos', ObsTerm(
            joint_pos_rel, 
            params={
                'default_joint_pos': self.default_joint_pos,
                'mapping': self.joint_obs_unitree_to_isaac_mapping
            }
        ))
        self.obs_manager.register('joint_vel', ObsTerm(joint_vel,
            params={
                'mapping': self.joint_obs_unitree_to_isaac_mapping
            }
        ))
        self.obs_manager.register('last_action', ObsTerm(last_action, params={"last_action": lambda: self.last_action}))
        
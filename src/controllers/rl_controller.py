# controllers/rl_controller.py
import torch
from controllers.controller_base import ControllerBase
from utils.config_loader import load_config

class RLController(ControllerBase):
    def __init__(self):
        config = load_config('doom.controllers', 'rl_controller_cfg.yaml')
        self.policy = torch.load(config['policy_path'])
        self.policy.eval()
        self.normalize_obs = config.get('observation_normalization', False)
        self.clip_actions = config.get('action_clipping', False)

    def compute_command(self, state, desired_goal):
        state_tensor = torch.tensor(state['observation'], dtype=torch.float32)
        action = self.policy(state_tensor).detach().numpy()

        if self.clip_actions:
            action = action.clip(-1.0, 1.0)
        return {"command": action}

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from state_manager.obs_manager import ObservationManager

class ControllerBase(ABC):
    def __init__(self):
        self.start_time = 0.0
        self.obs_manager: Optional[ObservationManager] = None
    
    def set_obs_manager(self, obs_manager: ObservationManager):
        """
        Set the observation manager for this controller.
        
        :param obs_manager: Observation manager to use
        """
        self.obs_manager = obs_manager

        # Optional: Register observations if the controller knows its requirements
        if hasattr(self, 'register_observations'):
            self.register_observations()
        
    def set_start_time(self, start_time):
        self.start_time = start_time
    
    @abstractmethod
    def compute_torques(self, state, desired_goal):
        """
        Compute control commands based on the current state and desired goal.

        Args:
            state (dict): Current state of the robot/environment.
            desired_goal (dict): Desired state or task goal.
        
        Returns:
            dict: Control command to be sent to the robot/environment.
        """
        pass
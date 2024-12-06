import threading
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from state_manager.obs_manager import ObservationManager


class ControllerBase(ABC):
    def __init__(self, pin_model_wrapper, configs: Dict[str, Any]):
        self.start_time = 0.0
        self.obs_manager: Optional[ObservationManager] = None
        # Build Pinocchio Model for Forward Kinematics
        self.pin_model_wrapper = pin_model_wrapper
        self.unitree_pin_joint_mappings = np.array(configs['robot_config']['unitree_pin_joint_mappings'])
        
        self.dof_pos_limit = np.array([self.pin_model_wrapper.model.lowerPositionLimit[7:][self.unitree_pin_joint_mappings], 
                                            self.pin_model_wrapper.model.upperPositionLimit[7:][self.unitree_pin_joint_mappings]]) # first 7 correspond to floating base position and quat (ignore)
        
        # DOF Pos Conservative Limits 
        soft_limit_factor = 0.95
        self.effort_limit = configs['robot_config']['effort_limit'] # instead get from pin, currently the values from pin don't seem right
        joint_pos_mean = (self.dof_pos_limit[0] + self.dof_pos_limit[1])/2
        joint_pos_range = self.dof_pos_limit[1] - self.dof_pos_limit[0]
        self.soft_dof_pos_limit = [joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,
                                    joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor]
        
        self._lock = threading.Lock()
        
    def set_soft_dof_pos_limits(self, soft_lower_limit, soft_upper_limit):
        self.soft_dof_pos_limit = [soft_lower_limit, soft_upper_limit]
    
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
        """
        Set the start time of the current mode/controller
        :param start_time: The start time.
        """
        self.start_time = start_time
        
    
    def _clip_effort(self, effort: np.ndarray) -> np.ndarray:
        """
        Clip the desired torques based on the motor limits.

        :param effort: The desired torques to clip.
        :return : The clipped torques.
        """
        return effort.clip(min=-self.effort_limit, max=self.effort_limit)
    

    def _clip_dof_pos(self, joint_pos_targets: np.ndarray) -> np.ndarray:
        """
        Clip the joint position based on the soft joint limits.

        :param pos_targets: The desired joint position targets to clip.

        :return : The clipped torques.
        """
        return joint_pos_targets.clip(self.soft_dof_pos_limit[0], self.soft_dof_pos_limit[1])

    
    @abstractmethod
    def compute_torques(self, state, desired_goal) -> Dict[str, np.ndarray]:
        """
        Compute control commands based on the current state and desired goal.

        :param state (dict): Current state of the robot/environment.
        :param desired_goal (dict): Desired state or task goal.
        :return : Control command to be sent to the robot/environment.
        """
        pass
    
    @abstractmethod
    def register_observations(self):
        """
        Register observations for each mode/controller. Maintain order to be passed directly to policy.
        """
        pass
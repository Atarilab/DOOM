from abc import ABC, abstractmethod

class ControllerBase(ABC):
    def __init__(self):
        self.start_time = 0.0
        
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
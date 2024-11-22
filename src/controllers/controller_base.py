from abc import ABC, abstractmethod

class ControllerBase(ABC):
    @abstractmethod
    def compute_command(self, state, desired_goal):
        """
        Compute control commands based on the current state and desired goal.

        Args:
            state (dict): Current state of the robot/environment.
            desired_goal (dict): Desired state or task goal.
        
        Returns:
            dict: Control command to be sent to the robot/environment.
        """
        pass
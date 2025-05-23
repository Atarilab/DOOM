import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, TYPE_CHECKING

import numpy as np
from commands.command_manager import CommandManager
from state_manager.obs_manager import ObservationManager

if TYPE_CHECKING:
    from robots.robot_base import RobotBase

class ControllerBase(ABC):
    """
    Abstract base class for robot controllers, providing a standardized interface
    and common utilities for robot control implementations.

    Manages joint limits, observation tracking, and provides core control infrastructure.
    """

    def __init__(
        self,
        robot: "RobotBase",
        configs: Dict[str, Any] = None,
    ):
        """
        Initialize the base controller with robot and configuration.

        :param robot: Robot model wrapper for additional computations
        :param configs: Configuration dictionary containing robot-specific parameters
        """
        # Timing and synchronization
        self.start_time = 0.0
        self._lock = threading.Lock()

        # Model and manager initialization
        self.robot: "RobotBase" = robot
        self.command_manager: Optional[CommandManager] = None
        self.obs_manager: Optional[ObservationManager] = None
        self.logger = None
        self.mode_manager = None  # Will be set by the mode manager when registering
        self.configs = configs
        self.control_dt = configs["controller_config"]["control_dt"]
        self.latest_state = None
        self.name = None
        self.active = False

        # Joint mapping and limits
        self._setup_joint_limits()

    def get_joystick_mappings(self) -> Dict[str, Callable[[], None]]:
        """
        Define joystick button mappings for this controller.
        Override this method in subclasses to define controller-specific mappings.
        
        Returns:
            Dict mapping button names to callback functions.
            Example: {"A": lambda: self.do_something()}
        """
        return {}

    def _setup_joint_limits(self):
        """
        Set up joint position and effort limits with conservative safety margins.
        """
        # Position limits (excluding first 7 DOFs for floating base)
        lower_limits = self.robot.mj_model.model.jnt_range[1:, 0]
        upper_limits = self.robot.mj_model.model.jnt_range[1:, 1]

        # Conservative limit settings
        soft_limit_factor = 0.97
        self.dof_pos_limit = np.array([lower_limits, upper_limits])

        # Effort limits
        # TODO: Fetch from mj_model
        try:
            self.effort_limit = self.robot.effort_limit
        except NotImplementedError:
            self.effort_limit = np.inf

        # Soft joint position limits
        joint_pos_mean = (lower_limits + upper_limits) / 2
        joint_pos_range = upper_limits - lower_limits

        self.soft_dof_pos_limit = [
            joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,
            joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor,
        ]

    def set_obs_manager(self, obs_manager: ObservationManager):
        """
        Configure observation manager for the controller.

        :param obs_manager: Observation manager instance
        """
        self.obs_manager = obs_manager

        # Automatically register observations if method exists
        if hasattr(self, "register_observations"):
            self.register_observations()

    def set_cmd_manager(self, cmd_manager: CommandManager):
        """
        Configure command manager for the controller.

        :param obs_manager: Command manager instance
        """
        self.command_manager = cmd_manager
        self.logger = cmd_manager.logger

        # Automatically register commands if method exists
        if hasattr(self, "register_commands"):
            self.register_commands()

    def update_state(self, state: Dict[str, Any]) -> None:
        """
        Sets the latest state received from the subscribers. This is done such that the mode-specific
        observations can be computed in real-time.

        :param state: The states directly subscribed from available topics.
        """
        with self._lock:
            self.latest_state = state

    def _clip_effort(self, effort: np.ndarray) -> np.ndarray:
        """
        Enforce motor torque limits.

        :param effort: Desired motor torques
        :return: Torques constrained within motor limits
        """
        return np.clip(effort, -self.effort_limit, self.effort_limit)

    def _clip_dof_pos(self, joint_pos_targets: np.ndarray, joint_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Enforce soft joint position limits.

        :param joint_pos_targets: Desired joint positions
        :return: Positions constrained within soft limits
        """
        if joint_indices is not None:
            return np.clip(joint_pos_targets[joint_indices], self.soft_dof_pos_limit[0][joint_indices], self.soft_dof_pos_limit[1][joint_indices])
        
        return np.clip(joint_pos_targets, self.soft_dof_pos_limit[0], self.soft_dof_pos_limit[1])

    def register_commands(self):
        """
        Register commands for the controller.
        This method can be overridden by subclasses to register specific commands.
        By default, it does nothing.
        """
        pass
    
    @abstractmethod
    def register_observations(self):
        """
        Register required observations for the specific controller mode.
        Implementations should maintain a consistent observation order.
        """
        pass

    @abstractmethod
    def compute_torques(self, state: Dict[str, Any], desired_goal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute control torques based on current state and desired goal.

        :param state: Current robot/environment state
        :param desired_goal: Target state or task objective
        :return: Control torques for robot actuation
        """
        if self.robot.mj_model is not None:
            self.robot.mj_model.update(state)
            
    @abstractmethod
    def set_mode(self):
        """
        When the mode is set, this method is called to initialize the controller and set up initial values.
        """
        pass
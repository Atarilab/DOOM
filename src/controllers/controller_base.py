import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from commands.command_manager import CommandManager
from state_manager.obs_manager import ObservationManager

if TYPE_CHECKING:
    from utils.mj_pin_wrapper.pin_robot import PinQuadRobotWrapper
    from utils.mj_wrapper.mj_robot import MjQuadRobotWrapper


class ControllerBase(ABC):
    """
    Abstract base class for robot controllers, providing a standardized interface
    and common utilities for robot control implementations.

    Manages joint limits, observation tracking, and provides core control infrastructure.
    """

    def __init__(
        self,
        pin_model_wrapper: "PinQuadRobotWrapper",
        mj_model_wrapper: "MjQuadRobotWrapper" = None,
        configs: Dict[str, Any] = None,
    ):
        """
        Initialize the base controller with model wrapper and configuration.

        :param pin_model_wrapper: Pinocchio model wrapper for kinematics
        :param mj_model_wrapper: MuJoCo model wrapper for additional computations
        :param configs: Configuration dictionary containing robot-specific parameters
        """
        # Timing and synchronization
        self.start_time = 0.0
        self._lock = threading.Lock()

        # Model and manager initialization
        self.pin_model_wrapper = pin_model_wrapper
        self.mj_model_wrapper = mj_model_wrapper
        self.command_manager: Optional[CommandManager] = None
        self.obs_manager: Optional[ObservationManager] = None
        self.configs = configs
        self.latest_state = None
        self.name = None

        # Joint mapping and limits
        self._setup_joint_limits(configs)

    def _setup_joint_limits(self, configs: Dict[str, Any]):
        """
        Set up joint position and effort limits with conservative safety margins.

        :param configs: Configuration dictionary
        """
        # Extract joint mappings
        self.unitree_pin_joint_mappings = np.array(configs["robot_config"]["unitree_pin_joint_mappings"])

        # Position limits (excluding first 7 DOFs for floating base)
        base_offset = 7
        lower_limits = self.pin_model_wrapper.model.lowerPositionLimit[base_offset:][self.unitree_pin_joint_mappings]
        upper_limits = self.pin_model_wrapper.model.upperPositionLimit[base_offset:][self.unitree_pin_joint_mappings]

        # Conservative limit settings
        soft_limit_factor = 0.95
        self.dof_pos_limit = np.array([lower_limits, upper_limits])

        # Effort limits
        self.effort_limit = configs["robot_config"]["effort_limit"]

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

        # Automatically register observations if method exists
        if hasattr(self, "register_commands"):
            self.register_commands()

    def set_start_time(self, start_time: float):
        """
        Record the start time for the current controller mode.

        :param start_time: Controller start timestamp
        """
        self.start_time = start_time

    def update_state(self, state: Dict[str, Any]) -> None:
        """
        Sets the latest state received from the subscribers. This is done such that the mode-specific
        observations can be computed in real-time.

        :param state: The states directly subscribed from available topics.
        """
        self.latest_state = state

    def _clip_effort(self, effort: np.ndarray) -> np.ndarray:
        """
        Enforce motor torque limits.

        :param effort: Desired motor torques
        :return: Torques constrained within motor limits
        """
        return np.clip(effort, -self.effort_limit, self.effort_limit)

    def _clip_dof_pos(self, joint_pos_targets: np.ndarray) -> np.ndarray:
        """
        Enforce soft joint position limits.

        :param joint_pos_targets: Desired joint positions
        :return: Positions constrained within soft limits
        """
        return np.clip(joint_pos_targets, self.soft_dof_pos_limit[0], self.soft_dof_pos_limit[1])

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
        if self.mj_model_wrapper is not None:
            self.mj_model_wrapper.update(state)
        pass

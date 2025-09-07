from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Type, Union

from controllers.controller_base import ControllerBase
from utils.mj_wrapper import MjRobotWrapper

if TYPE_CHECKING:
    from state_manager.state_manager import DDSStateSubscriber, ROS2StateSubscriber


class RobotBase(ABC):
    """
    Abstract base class for robot models.
    Subclasses must define all required properties.
    """

    def __init__(self, task, logger, device="cuda:0"):
        self.task = task
        self.logger = logger
        self.device = device
        self.mj_model = MjRobotWrapper(self.xml_path, self.ee_names, self.base_link, device=self.device)
        
        if self.floating_base:
            self.actuated_joint_indices = [self.mj_model.joint_names[joint_name] - 1 for joint_name in self.actuated_joint_names]
            self.non_actuated_joint_indices = [self.mj_model.joint_names[joint_name] - 1 for joint_name in self.non_actuated_joint_names]
        else:
            self.actuated_joint_indices = [self.mj_model.joint_names[joint_name] for joint_name in self.actuated_joint_names]
            self.non_actuated_joint_indices = [self.mj_model.joint_names[joint_name] for joint_name in self.non_actuated_joint_names]


    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def actuated_joint_names(self) -> List[str]:
        pass
    
    @property
    @abstractmethod
    def non_actuated_joint_names(self) -> List[str]:
        pass
    
    @property
    @abstractmethod
    def damping_gain(self) -> float:
        pass

    @property
    @abstractmethod
    def default_joint_positions(self) -> List[float]:
        pass

    @property
    @abstractmethod
    def effort_limit(self) -> Union[float, List[float]]:
        pass

    @property
    @abstractmethod
    def xml_path(self) -> str:
        pass

    @property
    @abstractmethod
    def subscribers(self) -> "Dict[str, Union[ROS2StateSubscriber, DDSStateSubscriber]]":
        pass

    @property
    @abstractmethod
    def available_controllers(self) -> "Dict[str, Dict[str, Type[ControllerBase]]]":
        pass

    @property
    @abstractmethod
    def low_cmd_msg(self):
        """Return the low command message class."""
        pass

    @property
    @abstractmethod
    def low_cmd_msg_type(self):
        """Return the low command message type."""
        pass
    
    @property
    @abstractmethod
    def motor_command_attributes(self) -> List[str]:
        """
        Get the list of motor command attributes supported by this robot.
        Override in subclasses for robot-specific attributes.
        
        Returns:
            List[str]: List of attribute names (e.g., ["q", "kp", "dq", "kd", "tau"])
        """
        return ["q", "kp", "dq", "kd", "tau"]
    
    @property
    @abstractmethod
    def base_link(self) -> str:
        """Return the name of the base link."""
        pass
    
    @property
    @abstractmethod
    def floating_base(self) -> bool:
        """Return if the robot has a floating base."""
        pass
    
    @property
    @abstractmethod
    def ee_names(self) -> List[str]:
        """Return the names of the end effectors of the robot."""
        pass
    
    @property
    def num_joints(self):
        """

        Returns:
            int: The number of joints of the robot.
        """
        return len(self.actuated_joint_names + self.non_actuated_joint_names)

    @property
    def get_joint_names(self) -> List[str]:
        return self.actuated_joint_names + self.non_actuated_joint_names
    
    @property
    def get_actuated_joint_indices(self) -> List[int]:
        return self.actuated_joint_indices
    
    @property
    def get_non_actuated_joint_indices(self) -> List[int]:
        return self.non_actuated_joint_indices


    def init_low_cmd(self, cmd_msg):
        """
        Initialize low-level command message with robot-specific defaults.
        Override in subclasses for robot-specific initialization.
        
        Args:
            cmd_msg: The command message to initialize
        """
        pass

    def update_motor_command(self, cmd_msg, motor_idx: int, motor_data: Dict):
        """
        Update a specific motor command with the provided data.
        Override in subclasses for robot-specific motor command updates.
        
        Args:
            cmd_msg: The command message to update
            motor_idx: Index of the motor to update
            motor_data: Dictionary containing motor command data
        """
        for attr in self.motor_command_attributes:
            if attr in motor_data:
                setattr(cmd_msg.motor_cmd[motor_idx], attr, motor_data[attr])

    def update_command_modes(self, cmd_msg, motor_commands: Dict):
        """
        Update command-level mode settings (e.g., mode_pr, mode_machine).
        Override in subclasses for robot-specific mode updates.
        
        Args:
            cmd_msg: The command message to update
            motor_commands: Dictionary containing mode settings
        """
        pass

    def get_mode_initialization_state(self, combined_state: Dict) -> bool:
        """
        Check if mode initialization is complete based on the current state.
        Override in subclasses for robot-specific mode checking.
        
        Args:
            combined_state: Current robot state
            
        Returns:
            bool: True if mode initialization is complete
        """
        return False

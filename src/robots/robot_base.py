from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Type, Union

from controllers.controller_base import ControllerBase

if TYPE_CHECKING:
    from state_manager.state_manager import DDSStateSubscriber, ROS2StateSubscriber


class RobotBase(ABC):
    """
    Abstract base class for robot models.
    Subclasses must define all required properties.
    """

    def __init__(self, task, logger):
        self.task = task
        self.logger = logger
        self.mj_model = None


    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def joint_names(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def num_joints(self) -> int:
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

    def get_joint_names(self) -> List[str]:
        return self.joint_names

    def get_num_joints(self) -> int:
        return self.num_joints

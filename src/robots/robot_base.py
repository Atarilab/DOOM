from abc import ABC, abstractmethod
from typing import List, Union

class RobotBase(ABC):
    """
    Abstract base class for robot models.
    Subclasses must define all required properties.
    """

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
    
    def get_joint_names(self) -> List[str]:
        return self.joint_names
        
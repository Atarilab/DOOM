import dataclasses
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np

@dataclasses.dataclass
class CommandParameter:
    """
    Defines a configurable command parameter with validation and conversion.
    """
    name: str
    description: str
    min_value: float
    max_value: float
    default_value: float
    type: type = float
    validator: Optional[Callable[[Any], bool]] = None

    def validate(self, value: Any) -> bool:
        """
        Validate the input value against parameter constraints.
        
        :param value: Value to validate
        :return: Whether the value is valid
        """
        try:
            # Convert to specified type
            converted_value = self.type(value)
            
            # Check type conversion worked
            if not isinstance(converted_value, self.type):
                return False
            
            # Check value range
            if converted_value < self.min_value or converted_value > self.max_value:
                return False
            
            # Run custom validator if provided
            if self.validator and not self.validator(converted_value):
                return False
            
            return True
        except (TypeError, ValueError):
            return False

class CommandManager:
    """
    Manages dynamic command configurations for different controllers.
    """

    def __init__(self, logger=None):
        """
        Initialize the command configuration manager.
        
        :param logger: Optional logger for tracking configuration changes
        """
        self._controller_configs: Dict[str, List[CommandParameter]] = {}
        self.logger = logger

    def register_controller_commands(
        self, 
        controller_type: str, 
        command_parameters: List[CommandParameter]
    ):
        """
        Register command parameters for a specific controller type.
        
        :param controller_type: Name of the controller type
        :param command_parameters: List of configurable parameters
        """
        self._controller_configs[controller_type] = command_parameters
        
        if self.logger:
            self.logger.info(f"Registered command configuration for {controller_type}")

    def get_controller_command_specs(self, controller_type: str) -> List[Tuple[str, str, float, float]]:
        """
        Get command specifications for UI configuration.
        
        :param controller_type: Name of the controller type
        :return: List of command specification tuples
        """
        if controller_type not in self._controller_configs:
            return []
        
        return [
            (
                param.name, 
                param.description, 
                param.min_value, 
                param.max_value
            ) for param in self._controller_configs[controller_type]
        ]

    def validate_and_update_commands(
        self, 
        controller_type: str, 
        current_commands: np.ndarray, 
        new_command_values: Dict[str, Any]
    ) -> np.ndarray:
        """
        Validate and update command values for a specific controller.
        
        :param controller_type: Name of the controller type
        :param current_commands: Current command values
        :param new_command_values: Dictionary of new command values
        :return: Updated command values
        """
        if controller_type not in self._controller_configs:
            raise ValueError(f"No command configuration for {controller_type}")
        
        # Create a copy of current commands to modify
        updated_commands = current_commands.copy()
        
        for param in self._controller_configs[controller_type]:
            # If value provided, validate and update
            if param.name in new_command_values:
                new_value = new_command_values[param.name]
                if param.validate(new_value):
                    # Find index of this parameter
                    try:
                        index = [p.name for p in self._controller_configs[controller_type]].index(param.name)
                        updated_commands[index] = float(new_value)
                    except ValueError:
                        if self.logger:
                            self.logger.warning(f"Could not find index for parameter {param.name}")
                else:
                    if self.logger:
                        self.logger.warning(f"Invalid value for {param.name}: {new_value}")
        
        return updated_commands


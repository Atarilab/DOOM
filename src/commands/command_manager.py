import dataclasses
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np
import logging

# TODO: Extend to other types other than float commands
@dataclasses.dataclass
class CommandTerm:
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

    def validate_type(self, value: Any) -> bool:
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

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the command configuration manager.
        :param logger: Optional logger for tracking configuration changes
        """
        self._commands: Dict[str, CommandTerm] = {}
        self.logger = logger or logging.getLogger(__name__)

    def register(self, name: str, command_term: CommandTerm):
        """
        Register a new command.
        
        :param controller_type: Name of the controller type
        :param command_terms: List of configurable parameters
        """
        if name in self._commands:
            self.logger.warning(f"Overwriting existing command: {name}")

        self._commands[name] = command_term

    def get_command_specs(self) -> List[Tuple[str, str, float, float]]:
        """
        Get command specifications for UI configuration.
        
        :param controller_type: Name of the controller type
        :return: List of command specification tuples
        """
        if self._commands == {}:
            return []
        
        return [
            (
                param.name, 
                param.description, 
                param.min_value, 
                param.max_value
            ) for cmd, param in self._commands.items()
        ]

    def validate_and_update_commands(
        self, 
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

        # Create a copy of current commands to modify
        updated_commands = current_commands.copy()
        
        for cmd, param in self._commands.items():
            # If value provided, validate and update
            if param.name in new_command_values:
                new_value = new_command_values[param.name]
                if param.validate_type(new_value):
                    # Find index of this parameter
                    try:
                        index = [p.name for cmd, p in self._commands.items()].index(param.name)
                        updated_commands[index] = float(new_value)
                    except ValueError:
                        if self.logger:
                            self.logger.warning(f"Could not find index for parameter {param.name}")
                else:
                    if self.logger:
                        self.logger.warning(f"Invalid value for {param.name}: {new_value}")
        
        return updated_commands


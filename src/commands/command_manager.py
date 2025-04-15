import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


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
    type: type
    current_value: Any = None
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

    def set_value(self, value: Any):
        self.current_value = value


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
            (param.name, param.description, param.min_value, param.max_value) for cmd, param in self._commands.items()
        ]

    def validate_and_change_commands(
        self, 
        current_commands: np.ndarray, 
        new_command_values: Dict[str, Any]
    ) -> np.ndarray:
        """
        Validate and update command values.

        Args:
            current_commands: Current command values
            new_command_values: New command values to set

        Returns:
            Updated command values
        """
        # Create a copy of current commands
        updated_commands = current_commands.copy()
        
        # Update each command value
        for name, value in new_command_values.items():
            if name in self._commands:
                command_term = self._commands[name]
                if command_term.validate_type(value):
                    command_term.set_value(value)
                    updated_commands[self.command_indices[name]] = value
                else:
                    self.logger.warning(
                        f"Invalid value {value} for command {name}. "
                        f"Must be of type {command_term.type}"
                    )
            else:
                self.logger.warning(f"Unknown command: {name}")
        
        return updated_commands

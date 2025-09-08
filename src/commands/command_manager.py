import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np


class WidgetType(Enum):
    """Types of UI widgets supported for commands."""
    INPUT = "input"  # Numeric input field
    BUTTON = "button"  # Button for discrete choices
    DROPDOWN = "dropdown"  # Dropdown for multiple choices
    SLIDER = "slider"  # Slider for numeric values


@dataclasses.dataclass
class CommandTerm:
    """
    Defines a configurable command parameter with validation and conversion.
    """

    name: str
    description: str
    default_value: Any
    type: type
    widget_type: WidgetType = WidgetType.INPUT
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    current_value: Any = None
    validator: Optional[Callable[[Any], bool]] = None
    # Widget-specific options
    options: Optional[List[Any]] = None  # For dropdown/button widgets
    step: Optional[float] = None  # For slider widgets

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
            if self.min_value is not None and converted_value < self.min_value:
                return False
            if self.max_value is not None and converted_value > self.max_value:
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

    def register_input_command(
        self, 
        name: str, 
        description: str, 
        default_value: float, 
        min_value: Optional[float] = None, 
        max_value: Optional[float] = None,
        step: Optional[float] = None
    ):
        """Register a numeric input command."""
        command_term = CommandTerm(
            name=name,
            description=description,
            default_value=default_value,
            type=float,
            widget_type=WidgetType.INPUT,
            min_value=min_value,
            max_value=max_value,
            step=step
        )
        self.register(name, command_term)

    def register_button_command(
        self, 
        name: str, 
        description: str, 
        options: List[str], 
        default_value: Optional[str] = None
    ):
        """Register a button group command for discrete choices."""
        if default_value is None:
            default_value = options[0] if options else ""
            
        command_term = CommandTerm(
            name=name,
            description=description,
            default_value=default_value,
            type=str,
            widget_type=WidgetType.BUTTON,
            options=options
        )
        self.register(name, command_term)

    def register_dropdown_command(
        self, 
        name: str, 
        description: str, 
        options: List[str], 
        default_value: Optional[str] = None
    ):
        """Register a dropdown command for multiple choices."""
        if default_value is None:
            default_value = options[0] if options else ""
            
        command_term = CommandTerm(
            name=name,
            description=description,
            default_value=default_value,
            type=str,
            widget_type=WidgetType.DROPDOWN,
            options=options
        )
        self.register(name, command_term)

    def register_slider_command(
        self, 
        name: str, 
        description: str, 
        default_value: float, 
        min_value: float, 
        max_value: float,
        step: Optional[float] = None
    ):
        """Register a slider command for numeric values."""
        command_term = CommandTerm(
            name=name,
            description=description,
            default_value=default_value,
            type=float,
            widget_type=WidgetType.SLIDER,
            min_value=min_value,
            max_value=max_value,
            step=step
        )
        self.register(name, command_term)

    def get_command_specs(self) -> List[Tuple[str, str, float, float]]:
        """
        Get command specifications for UI configuration (legacy method for backward compatibility).

        :return: List of command specification tuples
        """
        if self._commands == {}:
            return []

        return [
            (param.name, param.description, param.min_value, param.max_value) for cmd, param in self._commands.items()
        ]

    def get_widget_specs(self) -> List[Dict[str, Any]]:
        """
        Get widget specifications for UI configuration.

        :return: List of widget specification dictionaries
        """
        if self._commands == {}:
            return []

        widget_specs = []
        for cmd_name, param in self._commands.items():
            spec = {
                "name": param.name,
                "description": param.description,
                "widget_type": param.widget_type.value,
                "default_value": param.default_value,
                "current_value": param.current_value,
                "type": param.type.__name__,
            }
            
            # Add widget-specific parameters
            if param.widget_type in [WidgetType.INPUT, WidgetType.SLIDER]:
                spec["min_value"] = param.min_value
                spec["max_value"] = param.max_value
                if param.step is not None:
                    spec["step"] = param.step
                    
            elif param.widget_type in [WidgetType.BUTTON, WidgetType.DROPDOWN]:
                if param.options is not None:
                    spec["options"] = param.options
                    
            widget_specs.append(spec)
            
        return widget_specs

    def validate_and_change_commands(
        self, current_commands: np.ndarray, new_command_values: Dict[str, Any]
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
                else:
                    self.logger.warning(
                        f"Invalid value {value} for command {name}. " f"Must be of type {command_term.type}"
                    )
            else:
                self.logger.warning(f"Unknown command: {name}")

        return updated_commands

from typing import Any, Dict, Optional

from commands.command_manager import CommandManager
from controllers.stand_controller import ControllerBase
from state_manager.obs_manager import ObservationManager


class ModeManager:
    """
    A flexible mode management system that allows dynamic registration of modes (controllers).
    Modes are defined as a dictionary of controllers, where the key is the mode name and the value is a dictionary of controllers.
    """

    def __init__(self, logger=None, device="cuda:0"):
        self._modes: Dict[str, Dict[str, ControllerBase]] = {}
        self._current_mode: Optional[str] = None
        self._current_submode: Optional[str] = None
        self._mode_obs_managers: Dict[str, ObservationManager] = {}
        self._submode_cmd_managers: Dict[str, CommandManager] = {}
        self.logger = logger
        self.device = device

    def register_mode(self, mode_name: str, controllers: Dict[str, ControllerBase]):
        """
        Register a new mode with individual observation manager and command manager for each submode.

        :param mode_name: Name of the mode
        :param controllers: Dictionary of controllers for this mode
        """
        # Create an observation manager for each submode
        obs_managers = {submode_name: ObservationManager(logger=self.logger, device=self.device) for submode_name in controllers.keys()}

        self._modes[mode_name] = controllers
        self._mode_obs_managers[mode_name] = obs_managers

        # Pass corresponding obs manager to each controller
        for submode_name, controller in controllers.items():
            if hasattr(controller, "set_obs_manager"):
                controller.set_obs_manager(obs_managers[submode_name])

            if hasattr(controller, "register_commands"):
                self._submode_cmd_managers[submode_name] = CommandManager(logger=self.logger)
                controller.set_cmd_manager(self._submode_cmd_managers[submode_name])

    def set_mode(self, mode_name: str, submode: Optional[str] = None):
        """
        Set the current mode and optional submode.

        :param mode_name: Name of the mode to set
        :param submode: Optional submode within the mode
        :raises ValueError: If mode or submode is not registered
        """
        if mode_name not in self._modes:
            raise ValueError(f"Mode {mode_name} not registered")

        if submode is not None and submode not in self._modes[mode_name]:
            raise ValueError(f"Submode {submode} not registered for mode {mode_name}")

        # Deactivate existing controller
        if self._current_submode and self._current_mode:
            controller = self._modes[self._current_mode][self._current_submode]
            if hasattr(controller, "active"):
                controller.active = False

        self._current_mode = mode_name
        self._current_submode = submode

        if submode is not None:
            self.logger.debug(f"Mode set to: {self._current_mode} - {self._current_submode}")

        # Get the new controller
        if self._current_submode:
            controller = self._modes[self._current_mode][self._current_submode]
        else:
            controller = self._modes[self._current_mode].get("default", None)

        # Run set_mode function if it exists in the controller
        if controller and hasattr(controller, "set_mode"):
            controller.set_mode()
        else:
            self.logger.debug(f"Mode set to: {self._current_mode}")

    def get_active_controller(self) -> ControllerBase:
        """
        Get the active controller based on current mode and submode.

        :return: Active controller
        :raises ValueError: If no mode is set
        """
        if self._current_mode is None:
            raise ValueError("No mode is currently set")

        if self._current_submode:
            return self._modes[self._current_mode][self._current_submode]

        return self._modes[self._current_mode].get("default", None)

    def get_active_obs_manager(self) -> ObservationManager:
        """
        Get the observation manager for the current submode.

        :return: Active observation manager
        :raises ValueError: If no mode is set
        """
        if self._current_mode is None:
            raise ValueError("No mode is currently set")

        # If no submode, use 'default' for the mode
        if self._current_submode is None:
            return self._mode_obs_managers[self._current_mode]["default"]

        return self._mode_obs_managers[self._current_mode][self._current_submode]

    def get_current_mode_info(self) -> Dict[str, Optional[str]]:
        """
        Get current mode and submode information.

        :return: Dictionary with current mode and submode
        """
        return {"mode": self._current_mode, "submode": self._current_submode}

    def get_mode_info(self, mode_name: str) -> Dict[str, Any]:
        """
        Get mode and submode information for a given mode.

        :param mode_name: Name of the mode
        :return: Dictionary with mode and submode
        """
        return {"mode": mode_name, "submode": list(self._modes[mode_name].keys())}

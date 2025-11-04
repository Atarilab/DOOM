import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import pygame

if TYPE_CHECKING:
    from utils.mode_manager import ModeManager


class JoystickManager:
    """Manages joystick input and mapping to robot commands."""

    def __init__(
        self, mode_manager: "ModeManager", robot: str, logger: Optional[logging.Logger] = None, debug: bool = False
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.joystick = None
        self.axis_id = {}
        self.button_id = {}
        self.key_map = {}
        self.mode_manager = mode_manager
        self.robot = robot
        self.active_controller = None
        self.debug = debug
        self._setup_joystick()

        # Thread management
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._last_key_value = 0
        self._update_rate = 0.02  # 50Hz update rate

        # Command cooldown
        self._last_command_time = 0.0
        self._command_cooldown = 0.2  # 0.2 second cooldown between commands

        # Submodes indices
        self._submode_indices = {}  # Track current submode index for each mode

        # Start the joystick thread
        self.start_thread()

    def _setup_joystick(self, device_id=0, js_type="xbox"):
        """Initialize joystick and set up button/axis mappings."""
        pygame.init()
        pygame.joystick.init()
        self.joystick_count = pygame.joystick.get_count()

        if self.joystick_count <= 0:
            self.logger.warning("No joystick detected")
            return
        self.joystick = pygame.joystick.Joystick(device_id)
        self.joystick.init()
        self.logger.info("Joystick initialized successfully")

        if js_type == "xbox":
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 3,  # Right stick axis x
                "RY": 4,  # Right stick axis y
                "LT": 2,  # Left trigger
                "RT": 5,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 2,
                "Y": 3,
                "B": 1,
                "A": 0,
                "LB": 4,
                "RB": 5,
                "SELECT": 6,
                "START": 7,
            }

            self.key_map = {
                "R1": 0,
                "L1": 1,
                "start": 2,
                "select": 3,
                "R2": 4,
                "L2": 5,
                "F1": 6,
                "F2": 7,
                "A": 8,
                "B": 9,
                "X": 10,
                "Y": 11,
                "up": 12,
                "right": 13,
                "down": 14,
                "left": 15,
            }
        else:
            self.logger.error("Unsupported gamepad type")

    def set_active_controller(self, controller):
        """Set the currently active controller for joystick mapping."""
        with self._lock:
            self.active_controller = controller

    def update(self) -> int:
        """
        Get the current joystick state without processing it.

        Returns:
            int: Bitmap of pressed buttons/axes
        """
        with self._lock:
            return self._last_key_value

    def _joystick_thread_func(self):
        """Thread function that continuously updates joystick state."""
        self.logger.info(f"Joystick thread started (daemon={not self.debug})")
        while self._running:
            try:
                if not self.joystick:
                    time.sleep(0.1)
                    continue

                pygame.event.get()
                key_state = [0] * 16

                # Update button states
                key_state[self.key_map["R1"]] = self.joystick.get_button(self.button_id["RB"])
                key_state[self.key_map["L1"]] = self.joystick.get_button(self.button_id["LB"])
                key_state[self.key_map["start"]] = self.joystick.get_button(self.button_id["START"])
                key_state[self.key_map["select"]] = self.joystick.get_button(self.button_id["SELECT"])
                key_state[self.key_map["R2"]] = self.joystick.get_axis(self.axis_id["RT"]) > 0
                key_state[self.key_map["L2"]] = self.joystick.get_axis(self.axis_id["LT"]) > 0
                key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
                key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
                key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
                key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
                key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
                key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
                key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
                key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

                # Convert to integer bitmap
                key_value = 0
                for i in range(16):
                    key_value += key_state[i] << i

                # Check if enough time has passed since last command
                current_time = time.time()
                time_since_last_command = current_time - self._last_command_time

                # Only process commands if cooldown period has passed
                if time_since_last_command >= self._command_cooldown:
                    # Handle mode switching based on button combinations
                    if self.robot == "UnitreeGo2":
                        if key_state[self.key_map["select"]]:
                            self.mode_manager.set_mode("ZERO")
                            self._last_command_time = current_time
                        elif key_state[self.key_map["start"]] and self.active_controller.__class__.__name__ in [
                            "DampingController",
                            "ZeroTorqueController",
                        ]:
                            self.mode_manager.set_mode("STAND", "STAY_DOWN")
                            self._last_command_time = current_time
                        elif (
                            key_state[self.key_map["start"]]
                            and self.active_controller.__class__.__name__ != "DampingController"
                        ):
                            self.mode_manager.set_mode("DAMPING")
                            self._last_command_time = current_time
                        elif key_state[self.key_map["up"]] and self.active_controller.__class__.__name__ in {
                            "Go2StayDownController",
                            "Go2StandDownController",
                        }:
                            self.mode_manager.set_mode("STAND", "STAND_UP")
                            self._last_command_time = current_time
                        elif (
                            key_state[self.key_map["down"]]
                            and self.active_controller.__class__.__name__ == "Go2StandUpController"
                        ):
                            self.mode_manager.set_mode("STAND", "STAND_DOWN")
                            self._last_command_time = current_time
                        elif (
                            key_state[self.key_map["down"]]
                            and key_state[self.key_map["L1"]]
                            and self.active_controller.__class__.__name__ == "Go2StandDownController"
                        ):
                            self.mode_manager.set_mode("STAND", "STAY_DOWN")
                            self._last_command_time = current_time
                        elif (
                            key_state[self.key_map["L1"]]
                            and key_state[self.key_map["R1"]]
                            and self.active_controller.__class__.__name__ == "Go2StandUpController"
                        ):
                            # Cycle through available submodes of the for the main mode
                            main_mode = list(self.mode_manager._modes.keys())[3]
                            available_submodes = self.mode_manager.get_mode_info(main_mode)["submode"]

                            if main_mode not in self._submode_indices:
                                self._submode_indices[main_mode] = 0

                            current_index = self._submode_indices[main_mode]
                            self.mode_manager.set_mode(main_mode, available_submodes[current_index])

                            self.logger.info(f"Mode set to {main_mode} - {available_submodes[current_index]}")

                            self._submode_indices[main_mode] = (current_index + 1) % len(available_submodes)
                            self._last_command_time = current_time

                    elif self.robot == "UnitreeG1":
                        if key_state[self.key_map["select"]]:
                            self.mode_manager.set_mode("ZERO")
                            self._last_command_time = current_time
                        elif key_state[self.key_map["start"]] and self.active_controller.__class__.__name__ in [
                            "DampingController",
                            "ZeroTorqueController",
                        ]:
                            self.mode_manager.set_mode("STAND", "STAND_UP")
                            self._last_command_time = current_time
                        elif (
                            key_state[self.key_map["start"]]
                            and self.active_controller.__class__.__name__ != "DampingController"
                        ):
                            self.mode_manager.set_mode("DAMPING")
                            self._last_command_time = current_time
                        # elif key_state[self.key_map["up"]] and self.active_controller.__class__.__name__ in {
                        #     "Go2StayDownController",
                        #     "Go2StandDownController",
                        # }:
                        #     self.mode_manager.set_mode("STAND", "STAND_UP")
                        #     self._last_command_time = current_time
                        # elif (
                        #     key_state[self.key_map["down"]]
                        #     and self.active_controller.__class__.__name__ == "Go2StandUpController"
                        # ):
                        #     self.mode_manager.set_mode("STAND", "STAND_DOWN")
                        #     self._last_command_time = current_time
                        # elif (
                        #     key_state[self.key_map["down"]]
                        #     and key_state[self.key_map["L1"]]
                        #     and self.active_controller.__class__.__name__ == "Go2StandDownController"
                        # ):
                        #     self.mode_manager.set_mode("STAND", "STAY_DOWN")
                        #     self._last_command_time = current_time
                        elif (
                            key_state[self.key_map["L1"]]
                            and key_state[self.key_map["R1"]]
                            and self.active_controller.__class__.__name__
                            in ["G1StandUpController", "G1ZeroLegController"]
                        ):
                            # Cycle through available submodes of the for the main mode
                            main_mode = list(self.mode_manager._modes.keys())[3]
                            available_submodes = self.mode_manager.get_mode_info(main_mode)["submode"]

                            if main_mode not in self._submode_indices:
                                self._submode_indices[main_mode] = 0

                            current_index = self._submode_indices[main_mode]
                            self.mode_manager.set_mode(main_mode, available_submodes[current_index])

                            self.logger.info(f"Mode set to {main_mode} - {available_submodes[current_index]}")

                            self._submode_indices[main_mode] = (current_index + 1) % len(available_submodes)
                            self._last_command_time = current_time

                    # Execute controller-specific mappings if available
                    if self.active_controller and hasattr(self.active_controller, "get_joystick_mappings"):
                        mappings = self.active_controller.get_joystick_mappings()
                        for button, callback in mappings.items():
                            # Handle key combinations (e.g. "L1-right")
                            if "-" in button:
                                keys = button.split("-")
                                # Check if all keys in combination are pressed
                                if all(key in self.key_map and key_state[self.key_map[key]] for key in keys):
                                    try:
                                        callback()
                                        self._last_command_time = current_time
                                        break
                                    except Exception as e:
                                        self.logger.error(f"Error executing joystick mapping for {button}: {e}")
                            # Handle single keys
                            elif (
                                button in self.key_map
                                and key_state[self.key_map[button]]
                                and not any(key_state[self.key_map[key]] for key in self.key_map if key != button)
                            ):
                                try:
                                    callback()
                                    self._last_command_time = current_time
                                    break
                                except Exception as e:
                                    self.logger.error(f"Error executing joystick mapping for {button}: {e}")

                if key_value != 0:
                    self.logger.debug(f"Joystick state: {key_value}")

                with self._lock:
                    self._last_key_value = key_value
                    self.active_controller = self.mode_manager.get_active_controller()

                # Sleep to maintain update rate
                time.sleep(self._update_rate)

            except Exception as e:
                self.logger.error(f"Error in joystick thread: {e}")
                time.sleep(0.1)  # Prevent rapid error loops

    def start_thread(self):
        """Start the joystick processing thread."""
        if not self._running:
            self._running = True
            if self.joystick_count <= 0:
                return

            self._thread = threading.Thread(target=self._joystick_thread_func, daemon=not self.debug)
            self._thread.start()

    def stop_thread(self):
        """Stop the joystick processing thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self.logger.info("Joystick thread stopped")

    def cleanup(self):
        """Clean up pygame resources."""
        self.stop_thread()
        pygame.quit()

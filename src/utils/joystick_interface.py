import pygame
import logging
from typing import Optional
from utils.ui_interface import ModeManager

class JoystickManager:
    """Manages joystick input and mapping to robot commands."""

    def __init__(self, mode_manager: ModeManager, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.joystick = None
        self.axis_id = {}
        self.button_id = {}
        self.key_map = {}
        self.mode_manager = mode_manager
        self.active_controller = None
        self._setup_joystick()

    def _setup_joystick(self, device_id=0, js_type="xbox"):
        """Initialize joystick and set up button/axis mappings."""
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
            self.logger.info("Joystick initialized successfully")
        else:
            self.logger.warning("No gamepad detected")
            return

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
        self.active_controller = controller
       
    def update(self) -> int:
        """
        Update joystick state and execute controller-specific mappings.
        
        Returns:
            int: Bitmap of pressed buttons/axes
        """
        if not self.joystick:
            return 0

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

        # Handle mode switching based on button combinations
        if key_state[self.key_map["start"]] and self.active_controller.__class__.__name__ == "IdleController":
            self.mode_manager.set_mode("STANDING", "STAY_DOWN")
        elif key_state[self.key_map["start"]] and not self.active_controller.__class__.__name__ == "IdleController":
            self.mode_manager.set_mode("IDLE")
            
        elif key_state[self.key_map["up"]] and self.active_controller.__class__.__name__ == "StayDownController":
            self.mode_manager.set_mode("STANDING", "STAND_UP")
        elif key_state[self.key_map["down"]] and self.active_controller.__class__.__name__ == "StandUpController":
            self.mode_manager.set_mode("STANDING", "STAND_DOWN")
        elif key_state[self.key_map["down"]] and key_state[self.key_map["L1"]] and self.active_controller.__class__.__name__ == "StandDownController":
            self.mode_manager.set_mode("STANDING", "STAY_DOWN")
        elif key_state[self.key_map["L1"]] and key_state[self.key_map["R1"]] and self.active_controller.__class__.__name__ == "StandUpController":
            self.mode_manager.set_mode("RL-CONTACT", "RL-CONTACT")
            

        # Execute controller-specific mappings if available
        if self.active_controller and hasattr(self.active_controller, 'get_joystick_mappings'):
            mappings = self.active_controller.get_joystick_mappings()
            for button, callback in mappings.items():
                if button in self.key_map and key_state[self.key_map[button]]:
                    try:
                        callback()
                    except Exception as e:
                        self.logger.error(f"Error executing joystick mapping for {button}: {e}")

        if key_value != 0:
            self.logger.debug(f"Joystick state: {key_value}")
            
        self.active_controller = self.mode_manager.get_active_controller()
            
        return key_value

    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()
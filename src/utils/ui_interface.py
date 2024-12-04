import time
from typing import Dict, Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Header, Button, Static
from textual.reactive import reactive

from controllers.stand_controller import ControllerBase
from utils.logger import logging

from state_manager.obs_manager import ObservationManager


class ModeManager:
    """
    A flexible mode management system that allows dynamic registration of modes and controllers.
    """
    def __init__(self, logger=None):
        self._modes: Dict[str, Dict[str, ControllerBase]] = {}
        self._current_mode: Optional[str] = None
        self._current_submode: Optional[str] = None
        self._mode_obs_managers: Dict[str, ObservationManager] = {}
        self.logger = logger
        
    def register_mode(self, mode_name: str, controllers: Dict[str, ControllerBase]):
        """
        Register a new mode with individual obs managers for each submode.
        
        :param mode_name: Name of the mode
        :param controllers: Dictionary of controllers for this mode
        """
        # Create an observation manager for each submode
        obs_managers = {
            submode_name: ObservationManager(logger=self.logger) 
            for submode_name in controllers.keys()
        }
        
        self._modes[mode_name] = controllers
        self._mode_obs_managers[mode_name] = obs_managers
        
        # Pass corresponding obs manager to each controller
        for submode_name, controller in controllers.items():
            if hasattr(controller, 'set_obs_manager'):
                controller.set_obs_manager(obs_managers[submode_name])
                
        
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
        
        self._current_mode = mode_name
        self._current_submode = submode
        
        if submode is not None:
            self.logger.debug(f"Mode set to: {self._current_mode} - {self._current_submode}")
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
        
        return self._modes[self._current_mode].get('default', None)
    
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
            return self._mode_obs_managers[self._current_mode]['default']
        
        return self._mode_obs_managers[self._current_mode][self._current_submode]
      
    
    def get_current_mode_info(self) -> Dict[str, Optional[str]]:
        """
        Get current mode and submode information.
        
        :return: Dictionary with current mode and submode
        """
        return {
            'mode': self._current_mode,
            'submode': self._current_submode
        }

class RobotControlUI(App):
    """
    A Generic Textual UI to control robots.
    """
    CSS = """
        Screen {
            background: rgb(0, 0, 0);  /* Deep dark background */
            color: rgb(220, 220, 240);    /* Soft light text */
            align: center middle;
        }

        #control-container {
            width: 90%;
            height: 80%;
            border: round $primary;
            background: rgb(30, 30, 45);    /* Slightly lighter than screen background */
            padding: 2;
        }

        #title {
            text-align: center;
            text-style: bold;
            color: rgb(100, 150, 255);  /* Bright accent color */
            margin-bottom: 2;
            padding: 1;
        }

        #status {
            text-align: center;
            background: rgb(45, 45, 70);
            padding: 1;
            border: none;
            margin-bottom: 2;
        }

        .menu {
            height: auto;
            margin: 1;
            padding: 1;
            display: none;
            background: rgb(40, 40, 60);
            border: tall $background;
        }

        #main-menu {
            display: block;
            layout: vertical;
            align: center middle;
        }

        .section-header {
            text-align: center;
            text-style: bold;
            color: rgb(150, 180, 255);
            margin-bottom: 1;
            padding: 1;
            border-bottom: tall $background;
        }

        Button {
            min-width: 20;
            margin: 1 0;
            background: rgb(50, 50, 80);  /* Darker button background */
            color: rgb(200, 200, 230);    /* Soft light text */
            border: round $background;
        }

        Button:hover {
            background: rgb(70, 70, 110);  /* Lighter on hover */
            color: $text;
        }

        Button.selected {
            background: $accent;
            color: $text;
            border: tall $background;
        }

        .back-button {
            margin-top: 2;
            background: rgb(80, 40, 40);  /* Darker red tone for back button */
            color: rgb(255, 150, 150);
        }

        .back-button:hover {
            background: rgb(100, 50, 50);
            color: rgb(255, 180, 180);
        }
    """
            
    current_mode = reactive(None)
    current_submode = reactive(None)
    
    def __init__(self, mode_manager: ModeManager, logger: Optional[logging.Logger] = None):
        super().__init__()
        self.mode_manager = mode_manager
        self.logger = logger or logging.getLogger(__name__)
        self.mode_structure = self._build_mode_structure()
    
    def _build_mode_structure(self):
        """
        Dynamically build UI structure based on registered modes.
        
        :return: Dictionary representing UI hierarchy
        """
        structure = {}
        for mode, submodes in self.mode_manager._modes.items():
            structure[mode] = list(submodes.keys())
        return structure
    
    def compose(self) -> ComposeResult:
        """Create dynamic UI based on mode structure."""
        yield Header(show_clock=True)
        
        with Container(id="control-container"):
            yield Static("ATARI DOOM Robot Control Interface", id="title")
            yield Static("Current Status: IDLE", id="status")
            
            # Main menu with mode buttons
            with Vertical(id="main-menu", classes="menu"):
                yield Static("Robot Modes", classes="section-header")
                
                # Always include IDLE button in main menu
                yield Button("IDLE", classes="mode-button", id="mode-idle")
                
                # Generate other mode-level buttons dynamically
                for mode in self.mode_structure.keys():
                    if mode.upper() != 'IDLE':
                        yield Button(mode, classes="mode-button", id=f"mode-{mode.lower()}")
            
            # Dynamically generate submode menus
            for mode, submodes in self.mode_structure.items():
                with Vertical(id=f"{mode.lower()}-menu", classes="menu"):
                    yield Static(f"{mode} Submodes", classes="section-header")
                    
                    # Always add IDLE button to each submode menu
                    yield Button(
                        "IDLE", 
                        classes="mode-button", 
                        id="mode-idle"
                    )
                    
                    # Add submodes if applicable
                    if submodes:
                        for submode in submodes:
                            yield Button(
                                submode, 
                                classes="mode-button", 
                                id=f"{mode.lower()}-{submode.lower()}"
                            )
                    
                    # Add a back button to return to main menu
                    yield Button(
                        "← Back to Main Menu", 
                        classes="back-button", 
                        id=f"back-{mode.lower()}"
                    )
    
    def on_mount(self):
        """
        Ensure the initial mode is set and displayed correctly.
        """
        # Set initial mode if not already set
        if not self.mode_manager.get_current_mode_info()['mode']:
            self.mode_manager.set_mode('IDLE')
    
    def on_button_pressed(self, event: Button.Pressed):
        """
        Handle button presses with more comprehensive logic.
        
        :param event: Button press event
        """
        button_id = event.button.id
        
        # Back to main menu buttons
        if button_id.startswith("back-"):
            self.switch_to_menu("main-menu")
            return
        
        # Special handling for IDLE mode
        if button_id == "mode-idle":
            self.mode_manager.set_mode('IDLE')
            
            # Signal the robot controller about mode change
            if hasattr(self.app, 'robot_controller'):
                self.app.robot_controller.mode_change_event.set()
            
            # Update UI to reflect current state
            self.update_status()
            self.logger.info("Switched to IDLE mode")
            return
        
        # Mode selection
        if button_id.startswith("mode-"):
            mode = button_id.replace("mode-", "").upper()
            
            # If mode has submodes, switch to its submenu
            if self.mode_structure.get(mode, []):
                self.switch_to_menu(f"{mode.lower()}-menu")
            else:
                # Direct mode selection without submodes
                self.mode_manager.set_mode(mode)
            
            return
        
        # Submode selection
        for mode in self.mode_structure.keys():
            if button_id.startswith(f"{mode.lower()}-"):
                submode = button_id.replace(f"{mode.lower()}-", "").upper()
                self.mode_manager.set_mode(mode, submode)
                
                # Set start time for the active controller
                # Note: Can be used to set other properties within the controller
                active_controller = self.mode_manager.get_active_controller()
                if hasattr(active_controller, 'set_start_time'):
                    active_controller.set_start_time(time.time())
                
                break
            
        # Signal the robot controller about mode change
        if hasattr(self.app, 'robot_controller'):
            self.app.robot_controller.mode_change_event.set()
        
        # Update UI to reflect current state
        self.update_status()
        
        current_mode_info = self.mode_manager.get_current_mode_info()
        self.logger.info(f"Mode changed via UI: {current_mode_info}")
    
    def switch_to_menu(self, menu_id: str):
        """
        Switch the visible menu.
        
        :param menu_id: ID of the menu to display
        """
        # Hide all menus
        for menu in self.query(".menu"):
            menu.styles.display = "none"
        
        # Show selected menu
        selected_menu = self.query_one(f"#{menu_id}")
        selected_menu.styles.display = "block"
    
    def update_status(self):
        """
        Update the status display with current mode information.
        """
        current_mode_info = self.mode_manager.get_current_mode_info()
        status_text = f"Current Mode: {current_mode_info['mode']}"
        
        if current_mode_info['submode']:
            status_text += f" - {current_mode_info['submode']}"
        
        self.query_one("#status").update(status_text)
        
        
    def on_unmount(self) -> None:
        """Set robot to IDLE mode when UI is closed."""
        self.mode_manager.set_mode('IDLE')
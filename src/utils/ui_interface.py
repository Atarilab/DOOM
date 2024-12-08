import time
from typing import Dict, Optional, Any, List, Tuple

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Button, Static, Input, Label
from textual.reactive import reactive
from textual.validation import Number


from controllers.stand_controller import ControllerBase
from commands.command_manager import CommandManager
from utils.logger import logging

from textual.widget import Widget


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
        self._submode_cmd_managers: Dict[str, CommandManager] = {}
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
        command_managers = {}

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

        self._current_mode = mode_name
        self._current_submode = submode

        if submode is not None:
            self.logger.debug(
                f"Mode set to: {self._current_mode} - {self._current_submode}"
            )
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

class CommandWidget(Vertical):
    """
    A widget for configuring robot controller commands with dynamic input fields.
    
    Args:
        controller: The active robot controller
        command_specs: List of command specifications 
            Each spec is a tuple of (command_name, label, min_value, max_value)
    """
    
    def __init__(
        self, 
        controller: Any, 
        command_specs: List[Tuple[str, str, float, float]], 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.controller = controller
        self.command_specs = command_specs
        self.logger = logging.getLogger(__name__)

    def compose(self):
        """Dynamically create input fields based on command specifications."""
        # Title for command configuration
        yield Label("Configure Robot Commands", classes="section-header")

        # Create a container for input fields to group them
        with Vertical(classes="command-inputs-container"):
            
            yield Button(
                "Update Commands", 
                variant="success",
                classes="update-button", 
                id="update-commands-btn"
            )
            # Create input fields for each command specification
            for command_name, label, min_val, max_val in self.command_specs:
                with Horizontal(classes="command-input-row"):
                    # Label for the input
                    yield Label(label, classes="command-label")
                    
                    # Numeric input with validation
                    input_field = Input(
                        placeholder=f"Enter {label}",
                        validators=[
                            Number(minimum=min_val, maximum=max_val)
                        ],
                        classes="command-input",
                        id=f"input-{command_name}"
                    )
                    yield input_field

            
            
class RobotControlUI(App):
    """
    A Generic Textual UI to control robots.
    """

    CSS = """
    
        Screen {
            background: rgb(0, 0, 0);
            color: rgb(220, 220, 240);
            align: center middle;
        }

        #main-container {
            width: 100%;
            height: 100%;
            layout: horizontal;
            padding: 1;
        }

        #menu-container {
            width: 2fr;
            height: 100%;
            margin-right: 1;
        }

        #command-container {
            width: 1fr;
            height: 100%;
            background: rgb(40, 40, 60);
            border: round $background;
            padding: 1;
            display: none;  /* Hidden by default */
        }

        #command-container.visible {
            display: block;
        }

        #control-container {
            width: 100%;
            height: 80%;
            border: round $primary;
            background: rgb(30, 30, 45);
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

        .section-header {
            text-align: center;
            text-style: bold;
            color: rgb(150, 180, 255);
            margin-bottom: 1;
            padding: 1;
            border-bottom: tall $background;
        }

        .command-input-row {
            margin: 1;  /* Reduced margin */
            layout: horizontal;
            align: left middle;
            height: 3;  /* Consistent height */
        }

        .command-label {
            width: 40%;
            text-align: right;
            padding-right: 1;
            margin-right: 1;
        }

       .command-input {
            width: 40%;
            margin-right: 1;
        }

        .update-button {
            width: 50%;
            margin: 1;  /* Reduced margin */
            align: center middle;
            background: rgb(50, 100, 50);  /* Green background */
            color: white;
        }

        .update-button:hover {
            background: rgb(60, 120, 60);  /* Lighter green on hover */
        }

        #command-placeholder {
            width: 100%;
            display: block;
            align: center middle;
    }
    """

    current_mode = reactive(None)
    current_submode = reactive(None)

    def __init__(
        self, mode_manager: ModeManager, command_manager: Optional[CommandManager] = None, logger: Optional[logging.Logger] = None
    ):
        super().__init__()
        self.mode_manager = mode_manager
        self.command_manager = command_manager
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

        with Container(id="main-container"):
            # Left side: Menu and Controls
            with Container(id="menu-container"):
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
                            if mode.upper() != "IDLE":
                                yield Button(
                                    mode, classes="mode-button", id=f"mode-{mode.lower()}"
                                )

                    # Dynamically generate submode menus
                    for mode, submodes in self.mode_structure.items():
                        with Vertical(id=f"{mode.lower()}-menu", classes="menu"):
                            yield Static(f"{mode} Submodes", classes="section-header")

                            # Always add IDLE button to each submode menu
                            yield Button("IDLE", classes="mode-button", id="mode-idle")

                            # Add submodes if applicable
                            if submodes:
                                for submode in submodes:
                                    yield Button(
                                        submode,
                                        classes="mode-button",
                                        id=f"{mode.lower()}-{submode.lower()}",
                                    )

                            # Add a back button to return to main menu
                            yield Button(
                                "← Back to Main Menu",
                                classes="back-button",
                                id=f"back-{mode.lower()}",
                            )

            # Right side: Command Configuration Container
            with Container(id="command-container"):
                yield Static("Command Configuration", classes="section-header")
                yield Static(id="command-placeholder")
                    

    def on_mount(self):
        """
        Ensure the initial mode is set and displayed correctly.
        """
        # Set initial mode if not already set
        if not self.mode_manager.get_current_mode_info()["mode"]:
            self.mode_manager.set_mode("IDLE")
            
        # self.extend_ui_with_command()
        self.update_command_widget()

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
            self.mode_manager.set_mode("IDLE")

            # Signal the robot controller about mode change
            if hasattr(self.app, "robot_controller"):
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
                if hasattr(active_controller, "set_start_time"):
                    active_controller.set_start_time(time.time())

                break
            

        # Signal the robot controller about mode change
        if hasattr(self.app, "robot_controller"):
            self.app.robot_controller.mode_change_event.set()

        # Update UI to reflect current state
        self.update_status()

        current_mode_info = self.mode_manager.get_current_mode_info()
        self.logger.info(f"Mode changed via UI: {current_mode_info}")
        
        self.logger.info(f"Button Pressed: {event.button.id}")
        active_controller = self.mode_manager.get_active_controller()
        if event.button.id == "update-commands-btn" and hasattr(active_controller, "update_commands"):
            self.update_commands(active_controller)
            
        self.show_widgets()
        
        
    def update_commands(self, active_controller: ControllerBase):
        """
        Validate and apply command configurations to the controller.
        """
        updates = {}
        is_valid = True

        # Validate and collect input values
        command_specs = active_controller.command_manager.get_command_specs()
        for command_name, label, min_val, max_val in command_specs:
            input_field = self.query_one(f"#input-{command_name}")
            
            # Skip empty inputs
            if not input_field.value:
                continue

            try:
                # Convert and validate input
                value = float(input_field.value)
                
                # Validate against min and max
                if min_val <= value <= max_val:
                    updates[command_name] = value
                else:
                    input_field.styles.background = "red"
                    is_valid = False
            except ValueError:
                input_field.styles.background = "red"
                is_valid = False

        # Apply updates if all inputs are valid
        if is_valid:
            try:

                if hasattr(active_controller, "update_commands"):
                    active_controller.update_commands(updates)

                # Provide visual feedback
                update_button = self.query_one("#update-commands-btn")
                update_button.styles.background = "green"
                self.logger.info("Commands updated successfully!")
            
            except Exception as e:
                # Log and handle any update errors
                self.logger.info(f"Error updating commands: {e}")
                update_button = self.query_one("#update-commands-btn")
                update_button.styles.background = "red"
        else:
            # Indicate validation failure
            update_button = self.query_one("#update-commands-btn")
            update_button.styles.background = "red"
        
    def show_widgets(self):
        """
        Update the visibility and content of command configuration based on current mode.
        """

        # Get the command configuration container
        command_config_container = self.query_one("#command-container")
        config_placeholder = self.query_one("#command-placeholder")

        # Remove any existing configuration widgets
        for existing_widget in config_placeholder.query(".command"):
            existing_widget.remove()

        # Determine if command configuration should be shown
        should_show_config = False

        try:
            # Get the active controller
            active_controller = self.mode_manager.get_active_controller()
            
            # Check if the controller has command specifications
            if (active_controller.command_manager):
                # Creaactive_controller = self.mode_manager.get_active_controller()te command configuration widget
                command_config_widget = CommandWidget(
                    controller=active_controller,
                    command_specs=active_controller.command_manager.get_command_specs(),
                    classes="command"
                )

                # Mount the widget
                config_placeholder.mount(command_config_widget)
                should_show_config = True

        except Exception as e:
            self.logger.error(f"Error setting up command configuration: {e}")

        # Toggle visibility of command configuration container
        if should_show_config:
            command_config_container.add_class("visible")
        else:
            command_config_container.remove_class("visible")

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

        if current_mode_info["submode"]:
            status_text += f" - {current_mode_info['submode']}"

        self.query_one("#status").update(status_text)


    def on_unmount(self) -> None:
        """Set robot to IDLE mode when UI is closed."""
        self.mode_manager.set_mode("IDLE")


from typing import Any, List, Optional, Tuple

from textual.app import App, ComposeResult
from textual.color import Color
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.validation import Number
from textual.widgets import Button, Header, Input, Label, Select, Static

from commands.command_manager import CommandManager
from controllers.stand_controller import ControllerBase
from utils.logger import logging
from utils.mode_manager import ModeManager

# Define color constants
BUTTON_DEFAULT_COLOR = Color(50, 50, 80)
BUTTON_ACTIVE_COLOR = Color(70, 70, 110)


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
            margin-bottom: 1;
            padding: 1;
        }
        
        #subtitle {
            text-align: center;
            color: #6cb4ee;
            margin-top: 1;
            margin-bottom: 1;
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
        




        .gait-buttons-container {
            layout: grid;
            grid-size: 2;
            grid-columns: 1fr 1fr;
            grid-rows: auto;
            align: center middle;
            margin: 1 1;
            height: auto;
        }

        .gait-button {
            width: 50%;
            margin: 0 1;
            background: rgb(50, 50, 80);
            color: rgb(200, 200, 230);
            border: round $background;
            padding: 0 1;
        }

        .gait-button:hover {
            background: rgb(70, 70, 110);
            color: $text;
        }

        .gait-button.selected {
            background: $accent;
            color: $text;
            border: tall $background;
        }
    """

    current_mode = reactive(None)
    current_submode = reactive(None)

    def __init__(
        self,
        mode_manager: ModeManager,
        command_manager: Optional[CommandManager] = None,
        logger: Optional[logging.Logger] = None,
        task_name: str = None,
    ):
        super().__init__()
        self.mode_manager = mode_manager
        self.command_manager = command_manager
        self.logger = logger or logging.getLogger(__name__)
        self.mode_structure = self._build_mode_structure()
        self.task_name = task_name

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
                    try:
                        controller_type, task, interface, robot = self.task_name.split("-")
                    except ValueError:
                        controller_type = task = interface = robot = "Unknown"
                    subtitle = f"[b]Controller:[/b] {controller_type.upper()}    [b]Task:[/b] {task.capitalize()}    [b]Interface:[/b] {interface.capitalize()}    [b]Robot:[/b] {robot.capitalize()}"
                    yield Static(subtitle, id="subtitle")
                    yield Static("Current Status: ZERO", id="status")

                    # Main menu with mode buttons
                    with Vertical(id="main-menu", classes="menu"):
                        yield Static("Robot Modes", classes="section-header")

                        # Always include ZERO and DAMPING buttons in main menu
                        yield Button("ZERO", classes="mode-button", id="mode-zero")
                        yield Button("DAMPING", classes="mode-button", id="mode-damping")

                        # Generate other mode-level buttons dynamically (excluding ZERO and DAMPING)
                        for mode in self.mode_structure.keys():
                            if mode.upper() not in ["ZERO", "DAMPING"]:
                                yield Button(
                                    mode,
                                    classes="mode-button",
                                    id=f"mode-{mode.lower()}",
                                )

                    # Dynamically generate submode menus
                    for mode, submodes in self.mode_structure.items():
                        with Vertical(id=f"{mode.lower()}-menu", classes="menu"):
                            yield Static(f"{mode} Submodes", classes="section-header")

                            # Always add ZERO and DAMPING buttons to each submode menu
                            yield Button("ZERO", classes="mode-button", id="mode-zero")
                            yield Button("DAMPING", classes="mode-button", id="mode-damping")

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
            self.mode_manager.set_mode("ZERO")

        # self.extend_ui_with_command()
        self.show_widgets()

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

        # Special handling for ZERO mode
        if button_id == "mode-zero":
            self.mode_manager.set_mode("ZERO")

            # Update UI to reflect current state
            self.update_status()
            self.logger.info("Switched to ZERO mode")
            return

        if button_id == "mode-damping":
            self.mode_manager.set_mode("DAMPING")

            # Update UI to reflect current state
            self.update_status()
            self.logger.info("Switched to DAMPING mode")
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

                break

        # Update UI to reflect current state
        self.update_status()

        current_mode_info = self.mode_manager.get_current_mode_info()
        self.logger.info(f"Mode changed via UI: {current_mode_info}")

        self.logger.info(f"Button Pressed: {event.button.id}")
        active_controller = self.mode_manager.get_active_controller()
        if event.button.id == "update-commands-btn" and hasattr(active_controller, "change_commands"):
            self.change_commands(active_controller)

        self.show_widgets()

    def change_commands(self, active_controller: ControllerBase):
        """
        Validate and apply command configurations to the controller.
        """
        updates = {}
        is_valid = True

        # Check if this is a contact controller with gait patterns
        if hasattr(active_controller, "gait_patterns"):
            # For contact controllers, we don't need to validate inputs
            # The gait selection is handled by button clicks
            return

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
                if hasattr(active_controller, "change_commands"):
                    active_controller.change_commands(updates)

                # Provide visual feedback
                update_button = self.query_one("#update-commands-btn")
                update_button.styles.background = "green"

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
            if active_controller.command_manager:
                # Creaactive_controller = self.mode_manager.get_active_controller()te command configuration widget
                command_config_widget = CommandWidget(
                    controller=active_controller,
                    command_specs=active_controller.command_manager.get_command_specs(),
                    classes="command",
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
        """Set robot to DAMPING mode when UI is closed."""
        self.mode_manager.set_mode("DAMPING")


class CommandWidget(Vertical):
    """
    A widget for configuring robot controller commands with dynamic input fields.

    Args:
        controller: The active robot controller
        command_specs: List of command specifications
            Each spec is a tuple of (command_name, label, min_value, max_value)
        logger: The logger
    """

    def __init__(
        self,
        controller: Any,
        command_specs: List[Tuple[str, str, float, float]],
        *args,
        **kwargs,
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
                "Change Commands",
                variant="success",
                classes="update-button",
                id="update-commands-btn",
            )

            # Use the new widget specifications from command manager
            widget_specs = self.controller.command_manager.get_widget_specs()

            for spec in widget_specs:
                yield Label(spec["description"], classes="command-label")

                if spec["widget_type"] == "input":
                    # Create numeric input field
                    with Horizontal(classes="command-input-row"):
                        input_field = Input(
                            placeholder=f"Enter {spec['description']}",
                            validators=[
                                Number(
                                    minimum=spec.get("min_value", -float("inf")),
                                    maximum=spec.get("max_value", float("inf")),
                                )
                            ],
                            classes="command-input",
                            id=f"input-{spec['name']}",
                        )
                        yield input_field

                elif spec["widget_type"] == "button":
                    # Create button group for discrete choices
                    with Horizontal(classes="gait-buttons-container"):
                        for option in spec["options"]:
                            yield Button(
                                option.capitalize(),
                                classes="gait-button",
                                id=f"button-{spec['name']}-{option}",
                            )

                elif spec["widget_type"] == "dropdown":
                    # Create dropdown for multiple choices
                    with Horizontal(classes="command-input-row"):
                        dropdown = Select(
                            options=[(opt, opt) for opt in spec["options"]],
                            classes="command-dropdown",
                            id=f"dropdown-{spec['name']}",
                        )
                        yield dropdown

                elif spec["widget_type"] == "slider":
                    # Create input field for slider (fallback since Slider widget not available)
                    with Horizontal(classes="command-input-row"):
                        input_field = Input(
                            placeholder=f"Enter {spec['description']}",
                            validators=[
                                Number(
                                    minimum=spec.get("min_value", -float("inf")),
                                    maximum=spec.get("max_value", float("inf")),
                                )
                            ],
                            classes="command-input",
                            id=f"input-{spec['name']}",
                        )
                        yield input_field

    def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses for command selection."""
        button_id = event.button.id

        if button_id.startswith("button-"):
            # Parse button ID: button-{command_name}-{option}
            parts = button_id.split("-", 2)
            if len(parts) == 3:
                command_name = parts[1]
                option = parts[2]
                updates = {command_name: option}

                # Update the controller
                if hasattr(self.controller, "change_commands"):
                    self.controller.change_commands(updates)

                    # Visual feedback using predefined colors
                    # Reset all buttons for this command group
                    for btn in self.query(".gait-button"):
                        btn.styles.background = BUTTON_DEFAULT_COLOR
                    event.button.styles.background = BUTTON_ACTIVE_COLOR

    def on_select_changed(self, event: Select.Changed):
        """Handle dropdown selection changes."""
        select_id = event.select.id

        if select_id.startswith("dropdown-"):
            # Parse dropdown ID: dropdown-{command_name}
            command_name = select_id.replace("dropdown-", "")
            updates = {command_name: event.value}

            # Update the controller
            if hasattr(self.controller, "change_commands"):
                self.controller.change_commands(updates)

    def on_input_changed(self, event: Input.Changed):
        """Handle input field changes."""
        input_id = event.input.id

        if input_id.startswith("input-"):
            # Parse input ID: input-{command_name}
            command_name = input_id.replace("input-", "")

            try:
                # Convert to float for numeric inputs
                value = float(event.value) if event.value else 0.0
                updates = {command_name: value}

                # Update the controller
                if hasattr(self.controller, "change_commands"):
                    self.controller.change_commands(updates)
            except ValueError:
                # Invalid input, ignore
                pass

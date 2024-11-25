from enum import Enum, auto
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Header, Footer, Button, Static, Select
from textual.reactive import reactive
from textual.css.query import NoMatches
from textual.message import Message

from typing import Dict, Optional


from dataclasses import dataclass


class RobotMode(Enum):
    IDLE = auto()
    STANDING = auto()
    STANCE = auto()
    RL = auto()     
    # MPC = auto()    

class StandingState(Enum):
    IDLE = auto()
    STAND_UP = auto()
    STAND_DOWN = auto()
    STAY_DOWN = auto()

class RLGait(Enum):
    TROT = auto()
    WALK = auto()
    GALLOP = auto()
    BOUND = auto()

# class MPCGait(Enum):
#     CRAWL = auto()
#     TROT = auto()
#     PACE = auto()
#     DYNAMIC_TROT = auto()
    
@dataclass
class MotorCommand:
    q: float = 0.0
    kp: float = 0.0
    dq: float = 0.0
    kd: float = 0.0
    tau: float = 0.0

@dataclass
class GaitConfig:
    name: str
    params: Dict[str, float]

class RobotControlUI(App):
    """A Textual UI for robot control with RL and MPC support."""
    
    CSS = """
    Screen {
        align: center middle;
    }

    #control-container {
        width: 90%;
        height: 80%;
        border: round $primary;
        padding: 1;
    }
    
    #title {
        content-align: center middle;
    }

    .mode-button {
        width: 30;
        margin: 1;
    }

    #status {
        height: 3;
        content-align: center middle;
        text-style: bold;
    }

    .menu {
        height: auto;
        margin: 1;
        padding: 1;
        display: none;
    }

    #main-menu {
        display: block;
    }

    .section-header {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
        padding: 1;
    }

    Button {
        min-width: 20;
        margin: 1 0;
    }

    Button:hover {
        background: $accent;
    }

    Button.selected {
        background: $accent;
        border: tall $background;
    }

    .back-button {
        margin-top: 2;
        background: $primary-darken-2;
    }

    Select {
        width: 100%;
        margin: 1 0;
    }

    #gait-container {
        margin: 1 0;
        border: tall $primary-darken-1;
        padding: 1;
    }
    """

    current_mode = reactive(RobotMode.IDLE)
    standing_state = reactive(StandingState.IDLE)
    current_rl_gait = reactive(RLGait.TROT)
    # current_mpc_gait = reactive(MPCGait.TROT)

    def __init__(self, robot_controller: 'RobotController'):
        super().__init__()
        self.robot_controller = robot_controller

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(show_clock=True)
        
        with Container(id="control-container"):
            yield Static("ATARI DOOM Robot Control Interface", id="title")
            yield Static("Current Status: IDLE", id="status")
            
            # Main menu
            with Vertical(id="main-menu", classes="menu"):
                yield Static("Robot Modes", classes="section-header")
                yield Button("IDLE", classes="mode-button", id="mode-idle")
                yield Button("STANDING", classes="mode-button", id="mode-standing")
                yield Button("STANCE", classes="mode-button", id="mode-stance")
                yield Button("RL", classes="mode-button", id="mode-rl")
                # yield Button("MPC", classes="mode-button", id="mode-mpc")
            
            # Standing menu
            with Vertical(id="standing-menu", classes="menu"):
                yield Static("Standing Controllers", classes="section-header")
                yield Button("Standing Idle", classes="mode-button", id="standing-idle")
                yield Button("Stand Up", classes="mode-button", id="standing-stand_up")
                yield Button("Stand Down", classes="mode-button", id="standing-stand_down")
                yield Button("Stay Down", classes="mode-button", id="standing-stay_down")
                yield Button("← Back to Main Menu", classes="back-button", id="back-button")

            # RL menu
            with Vertical(id="rl-menu", classes="menu"):
                yield Static("RL Controllers", classes="section-header")
                with Container(id="gait-container"):
                    yield Static("Select Gait:")
                    yield Select(
                        [(gait.name, gait.name) for gait in RLGait],
                        id="rl-gait-select",
                        value=RLGait.TROT.name
                    )
                yield Button("← Back to Main Menu", classes="back-button", id="back-button-rl")

            # # MPC menu
            # with Vertical(id="mpc-menu", classes="menu"):
            #     yield Static("MPC Controllers", classes="section-header")
            #     with Container(id="gait-container"):
            #         yield Static("Select Gait:")
            #         yield Select(
            #             [(gait.name, gait.name) for gait in MPCGait],
            #             id="mpc-gait-select",
            #             value=MPCGait.TROT.name
            #         )
            #     yield Button("← Back to Main Menu", classes="back-button", id="back-button-mpc")

    def switch_to_menu(self, menu_id: str) -> None:
        """Switch visibility between menus."""
        for menu in self.query(f".menu"):
            menu.styles.display = "block" if menu.id == menu_id else "none"

    def update_button_states(self) -> None:
        """Update the visual state of buttons based on current modes."""
        # Update main menu buttons
        for mode in RobotMode:
            try:
                button = self.query_one(f"#mode-{mode.name.lower()}")
                button.set_class(self.current_mode == mode, "selected")
            except NoMatches:
                pass

        # Update standing menu buttons if in standing mode
        if self.current_mode == RobotMode.STANDING:
            for state in StandingState:
                try:
                    button = self.query_one(f"#standing-{state.name.lower()}")
                    button.set_class(self.standing_state == state, "selected")
                except NoMatches:
                    pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id.startswith("back-button"):
            self.switch_to_menu("main-menu")
            self.update_button_states()
            
        elif button_id.startswith("mode-"):
            mode = RobotMode[button_id.replace("mode-", "").upper()]
            self.robot_controller.set_mode(mode)
            self.current_mode = mode
            
            # Switch to appropriate menu
            if mode == RobotMode.STANDING:
                self.switch_to_menu("standing-menu")
            elif mode == RobotMode.RL:
                self.switch_to_menu("rl-menu")
            # elif mode == RobotMode.MPC:
            #     self.switch_to_menu("mpc-menu")
            
            self.update_button_states()
                
        elif button_id.startswith("standing-"):
            state = StandingState[button_id.replace("standing-", "").upper()]
            self.robot_controller.set_standing_state(state)
            self.standing_state = state
            self.update_button_states()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle gait selection changes."""
        if event.select.id == "rl-gait-select":
            gait = RLGait[event.value]
            self.current_rl_gait = gait
            self.robot_controller.set_rl_gait(gait)
        # elif event.select.id == "mpc-gait-select":
        #     gait = MPCGait[event.value]
        #     self.current_mpc_gait = gait
        #     self.robot_controller.set_mpc_gait(gait)

    def watch_current_mode(self, mode: RobotMode) -> None:
        """React to changes in robot mode."""
        status_text = f"Current Status: {mode.name}"
        if mode == RobotMode.STANDING:
            status_text += f" - {self.standing_state.name}"
        elif mode == RobotMode.RL:
            status_text += f" - {self.current_rl_gait.name}"
        # elif mode == RobotMode.MPC:
        #     status_text += f" - {self.current_mpc_gait.name}"
        self.query_one("#status").update(status_text)
        self.update_button_states()
        
    def on_unmount(self) -> None:
        """Set robot to IDLE mode when UI is closed."""
        self.robot_controller.set_mode(RobotMode.IDLE)
        self.current_mode = RobotMode.IDLE
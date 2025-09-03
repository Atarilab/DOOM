import time
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from controllers.controller_base import ControllerBase
from state_manager.obs_manager import ObsTerm

if TYPE_CHECKING:
    from robots.robot_base import RobotBase


class G1GainTuningController(ControllerBase):
    """
    A continuous phase-based P-D controller for the G1 robot that moves from joint position A to B
    and back continuously, with tunable gains for each joint.
    
    This controller is designed for tuning the P-D gains of the G1 robot by performing
    continuous oscillatory movements between two joint configurations.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any]):
        super().__init__(robot=robot, configs=configs)

        self.name = "G1GainTuningController"
        
        # Timing parameters
        self.cycle_time = 4.0  # Time for one complete cycle (A->B->A)
        self.phase_time = self.cycle_time / 2.0  # Time for A->B or B->A
        
        # Joint configuration for position A (default/rest position)
        self.position_a = np.array(configs["controller_config"]["position_a"])
        
        # Joint configuration for position B (target position)
        self.position_b = np.array(configs["controller_config"]["position_b"])
        
        # Default P-D gains for each joint
        self.default_kps = np.array(configs["controller_config"]["default_kps"])
        self.default_kds = np.array(configs["controller_config"]["default_kds"])
        
        # Current tunable gains (initialized to defaults)
        self.current_kps = self.default_kps.copy()
        self.current_kds = self.default_kds.copy()
        
        # Control state
        self.start_time = 0.0
        self.current_phase = 0.0  # 0.0 = at position A, 1.0 = at position B
        
        # Ensure we have the right number of joints
        expected_joints = 29  # G1 has 29 motors
        if len(self.position_a) != expected_joints:
            raise ValueError(f"Expected {expected_joints} joints, got {len(self.position_a)}")
        if len(self.position_b) != expected_joints:
            raise ValueError(f"Expected {expected_joints} joints, got {len(self.position_b)}")
        if len(self.default_kps) != expected_joints:
            raise ValueError(f"Expected {expected_joints} joints, got {len(self.default_kps)}")
        if len(self.default_kds) != expected_joints:
            raise ValueError(f"Expected {expected_joints} joints, got {len(self.default_kds)}")

    def set_mode(self):
        """Initialize the controller when mode is set."""
        self.start_time = time.time()
        self.current_phase = 0.0

    def register_observations(self):
        """Register observations for this controller."""
        from state_manager.observations import current_time

        self.obs_manager.register(
            "time",
            ObsTerm(
                current_time,
                obs_dim=1,
            ),
        )

    def register_commands(self):
        """Register tunable command parameters for gain tuning."""
        if self.command_manager is None:
            return
            
        from commands.command_manager import CommandTerm
        
        cmd_manager = self.command_manager
        if cmd_manager is None:
            return
            
        # Register P gains for each joint
        for i in range(len(self.current_kps)):
            joint_name = f"kp_joint_{i:02d}"
            cmd_manager.register(
                joint_name,
                CommandTerm(
                    name=joint_name,
                    description=f"P gain for joint {i}",
                    default_value=float(self.default_kps[i]),
                    type=float,
                    min_value=0.0,
                    max_value=1000.0,
                    current_value=float(self.current_kps[i])
                )
            )
        
        # Register D gains for each joint
        for i in range(len(self.current_kds)):
            joint_name = f"kd_joint_{i:02d}"
            cmd_manager.register(
                joint_name,
                CommandTerm(
                    name=joint_name,
                    description=f"D gain for joint {i}",
                    default_value=float(self.default_kds[i]),
                    type=float,
                    min_value=0.0,
                    max_value=100.0,
                    current_value=float(self.current_kds[i])
                )
            )
        
        # Register cycle time parameter
        cmd_manager.register(
            "cycle_time",
            CommandTerm(
                name="cycle_time",
                description="Time for one complete cycle (A->B->A) in seconds",
                default_value=4.0,
                type=float,
                min_value=1.0,
                max_value=20.0,
                current_value=self.cycle_time
            )
        )

    def update_gains_from_commands(self):
        """Update the current gains based on command manager values."""
        if self.command_manager is None:
            return
            
        # Update P gains
        for i in range(len(self.current_kps)):
            joint_name = f"kp_joint_{i:02d}"
            if joint_name in self.command_manager._commands:
                cmd_term = self.command_manager._commands[joint_name]
                if cmd_term.current_value is not None:
                    self.current_kps[i] = float(cmd_term.current_value)
        
        # Update D gains
        for i in range(len(self.current_kds)):
            joint_name = f"kd_joint_{i:02d}"
            if joint_name in self.command_manager._commands:
                cmd_term = self.command_manager._commands[joint_name]
                if cmd_term.current_value is not None:
                    self.current_kds[i] = float(cmd_term.current_value)
        
        # Update cycle time
        if "cycle_time" in self.command_manager._commands:
            cmd_term = self.command_manager._commands["cycle_time"]
            if cmd_term.current_value is not None:
                self.cycle_time = float(cmd_term.current_value)
                self.phase_time = self.cycle_time / 2.0

    def compute_phase(self, elapsed_time: float) -> float:
        """
        Compute the current phase (0.0 to 1.0) based on elapsed time.
        
        Args:
            elapsed_time: Time elapsed since controller start
            
        Returns:
            Phase value from 0.0 (position A) to 1.0 (position B)
        """
        # Normalize time to cycle
        cycle_progress = (elapsed_time % self.cycle_time) / self.cycle_time
        
        if cycle_progress < 0.5:
            # First half: A -> B (0.0 to 1.0)
            phase = 2.0 * cycle_progress
        else:
            # Second half: B -> A (1.0 to 0.0)
            phase = 2.0 * (1.0 - cycle_progress)
        
        return np.clip(phase, 0.0, 1.0)

    def compute_lowlevelcmd(self, state):
        """Compute low-level motor commands for the G1 robot."""
        super().compute_lowlevelcmd(state)
        
        # Update gains from command manager
        self.update_gains_from_commands()
        
        # Get current time
        if self.obs_manager is None:
            return {}
        obs = self.obs_manager.compute(state)
        # current_time = obs["time"]
        elapsed_time = time.time() - self.start_time
        
        # Compute current phase
        self.current_phase = self.compute_phase(elapsed_time)
        
        # Interpolate between position A and B based on phase
        target_positions = (1.0 - self.current_phase) * self.position_a + self.current_phase * self.position_b
        
        # Generate motor commands
        cmd = {}
        # Handle actuated joints
        if self.robot.actuated_joint_indices is not None:
            for i, motor_idx in enumerate(self.robot.actuated_joint_indices):
                if motor_idx < len(target_positions):
                    cmd[f"motor_{motor_idx}"] = {
                        "q": target_positions[motor_idx],
                        "kp": self.current_kps[motor_idx],
                        "dq": 0.0,
                        "kd": self.current_kds[motor_idx],
                        "tau": 0.0,
                    }
                else:
                    # Fallback for joints not in our target array
                    cmd[f"motor_{motor_idx}"] = {
                        "q": 0.0,
                        "kp": 0.0,
                        "dq": 0.0,
                        "kd": 0.0,
                        "tau": 0.0,
                    }
            
            # Handle non-actuated joints
            for i, motor_idx in enumerate(self.robot.non_actuated_joint_indices):
                cmd[f"motor_{motor_idx}"] = {
                    "q": 0.0,
                    "kp": 0.0,
                    "dq": 0.0,
                    "kd": 0.0,
                    "tau": 0.0,
                }
        else:
            # Fallback: assume all joints are actuated
            for i in range(self.robot.num_joints):
                if i < len(target_positions):
                    cmd[f"motor_{i}"] = {
                        "q": target_positions[i],
                        "kp": self.current_kps[i],
                        "dq": 0.0,
                        "kd": self.current_kds[i],
                        "tau": 0.0,
                    }
                else:
                    cmd[f"motor_{i}"] = {
                        "q": 0.0,
                        "kp": 0.0,
                        "dq": 0.0,
                        "kd": 0.0,
                        "tau": 0.0,
                    }
        
        return cmd

    def get_joystick_mappings(self) -> Dict[str, Any]:
        """Define joystick button mappings for this controller."""
        return {
            "L2-R2": lambda: self.reset_gains_to_default(),
            "up": lambda: self.adjust_cycle_time(0.5),
            "down": lambda: self.adjust_cycle_time(-0.5),
        }

    def reset_gains_to_default(self):
        """Reset all gains to their default values."""
        self.current_kps = self.default_kps.copy()
        self.current_kds = self.default_kds.copy()
        self.cycle_time = 4.0
        self.phase_time = self.cycle_time / 2.0
        
        # Update command manager if available
        if self.command_manager is not None:
            for i in range(len(self.current_kps)):
                kp_name = f"kp_joint_{i:02d}"
                kd_name = f"kd_joint_{i:02d}"
                
                if kp_name in self.command_manager._commands:
                    self.command_manager._commands[kp_name].current_value = float(self.current_kps[i])
                if kd_name in self.command_manager._commands:
                    self.command_manager._commands[kd_name].current_value = float(self.current_kds[i])
            
            if "cycle_time" in self.command_manager._commands:
                self.command_manager._commands["cycle_time"].current_value = self.cycle_time

    def adjust_cycle_time(self, delta: float):
        """Adjust the cycle time by the given delta."""
        new_cycle_time = np.clip(self.cycle_time + delta, 0.5, 10.0)
        self.cycle_time = new_cycle_time
        self.phase_time = self.cycle_time / 2.0
        
        # Update command manager if available
        if self.command_manager is not None and "cycle_time" in self.command_manager._commands:
            self.command_manager._commands["cycle_time"].current_value = self.cycle_time

    def get_status_info(self) -> Dict[str, Any]:
        """Get current status information for debugging/monitoring."""
        return {
            "controller_name": self.name,
            "current_phase": self.current_phase,
            "cycle_time": self.cycle_time,
            "phase_time": self.phase_time,
            "elapsed_time": time.time() - self.start_time if self.start_time > 0 else 0.0,
            "current_kps": self.current_kps.tolist(),
            "current_kds": self.current_kds.tolist(),
            "position_a": self.position_a.tolist(),
            "position_b": self.position_b.tolist(),
        }

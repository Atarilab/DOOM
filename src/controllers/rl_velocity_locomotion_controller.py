
import time
import torch
from typing import Any, Dict
from commands.command_manager import CommandTerm
from state_manager.obs_manager import ObsTerm

from controllers.rl_controller_base import RLControllerBase

class RLQuadrupedLocomotionVelocityController(RLControllerBase):
    """
    Velocity-conditioned quadruped RL Locomotion Controller
    Uses contact-implicit reinforcement learning policy
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any]):
        super().__init__(robot=robot, configs=configs)

        # Default velocity commands
        self.velocity_commands = torch.tensor([0.0, 0.0, 0.0])

    def set_mode(self):
        """Runs when the mode is changed in the UI."""
        super().set_mode()

    def change_commands(self, new_commands: Dict[str, Any]):
        """
        Change velocity commands with validation.

        :param new_commands: Dictionary of new command values
        """

        try:
            # Create a new tensor with the updated values
            new_velocity_commands = self.velocity_commands.clone()

            if "x_velocity" in new_commands:
                new_velocity_commands[0] = new_commands["x_velocity"]
            if "y_velocity" in new_commands:
                new_velocity_commands[1] = new_commands["y_velocity"]
            if "yaw" in new_commands:
                new_velocity_commands[2] = new_commands["yaw"]

            self.velocity_commands = new_velocity_commands

            if self.logger is not None:
                self.logger.debug(f"Command Updated: {new_commands}")
        except ValueError as e:
            # Log error or handle validation failure
            if self.logger is not None:
                self.logger.error(f"Command update failed: {e}")

    def register_commands(self):
        self.command_manager.register(
            "x_velocity",
            CommandTerm(
                type=float,
                name="x_velocity",
                description="X Velocity (m/s)",
                min_value=-1.0,
                max_value=1.0,
                default_value=0.0,
        ),)

        self.command_manager.register(
            "y_velocity",
            CommandTerm(
                type=float,
                name="y_velocity",
                description="Y Velocity (m/s)",
                min_value=-1.0,
                max_value=1.0,
                default_value=0.0,
        ),)

        self.command_manager.register(
            "yaw",
            CommandTerm(
                type=float,
                name="yaw_rate",
                description="Yaw Rate (rad/s)",
                min_value=-3.14,
                max_value=3.14,
                default_value=0.0,
        ),)

    def register_observations(self):
        """
        Register observations for velocity-conditioned locomotion. Maintains a specific order for direct policy input.
        Lambda is used to get the latest value from the class variables.
        """
        from state_manager.observations import (
            ang_vel_b,
            joint_pos_rel,
            joint_vel,
            last_action,
            lin_vel_b,
            projected_gravity_b,
            velocity_commands,
        )
        self.obs_manager.register("lin_vel_b", ObsTerm(lin_vel_b))
        self.obs_manager.register("ang_vel_b", ObsTerm(ang_vel_b))
        self.obs_manager.register("projected_gravity", ObsTerm(projected_gravity_b))
        self.obs_manager.register(
            "velocity_commands",
            ObsTerm(
                velocity_commands,
                params={"velocity_commands": lambda: self.velocity_commands},
        ),)
        self.obs_manager.register(
            "joint_pos",
            ObsTerm(
                joint_pos_rel,
                params={
                    "default_joint_pos": self.default_joint_pos.numpy(),
                    "mapping": self.joint_obs_unitree_to_isaac_mapping,
        },),)
        self.obs_manager.register(
            "joint_vel",
            ObsTerm(joint_vel, params={"mapping": self.joint_obs_unitree_to_isaac_mapping}),
        )
        self.obs_manager.register(
            "last_action",
            ObsTerm(last_action, params={"last_action": lambda: self.raw_action}),
        )

    def get_joystick_mappings(self):
        """
        Define joystick button mappings for gait changes and heading control.

        Returns:
            Dict mapping button names to callback functions.
        """
        return {

            # X-Velocity
            "up": lambda: self.change_commands({
                "x_velocity": self.velocity_commands[0] + 0.1,
                "y_velocity": self.velocity_commands[1],
            }),
            "down": lambda: self.change_commands({
                "x_velocity": self.velocity_commands[0] - 0.1,
                "y_velocity": self.velocity_commands[1],
            }),
            # Y-Velocity
            "left": lambda: self.change_commands({
                "x_velocity": self.velocity_commands[0],
                "y_velocity": self.velocity_commands[1] + 0.1,
            }),
            "right": lambda: self.change_commands({
                "x_velocity": self.velocity_commands[0],
                "y_velocity": self.velocity_commands[1] - 0.1,
        }),}

    def compute_torques(self, state, desired_goal):
        """
        Compute motor commands using the learned policy.

        :param state: Current robot state
        :param desired_goal: Desired goal state (not used in this implementation)
        :return: Motor commands dictionary
        """
        super().compute_torques(state, desired_goal)

        start_time = time.perf_counter()

        try:
            joint_pos_targets = self.compute_joint_pos_targets()

            # Prepare motor commands
            self.cmd = {
                f"motor_{i}": {
                    "q": joint_pos_targets[i],
                    "kp": self.Kp,
                    "dq": 0.0,
                    "kd": self.Kd,
                    "tau": 0.0,
                }
                for i in range(self.robot.num_joints)
            }

            # Track command preparation time
            self.cmd_preparation_time = time.perf_counter() - start_time

        except Exception as e:
            self.logger.error(f"Error computing torques: {e}")
            self.cmd = {
                f"motor_{i}": {
                    "q": self.default_joint_pos[i],
                    "kp": self.Kp,
                    "dq": 0.0,
                    "kd": self.Kd,
                    "tau": 0.0,
                }
                for i in range(self.robot.num_joints)
            }

        return self.cmd

class RLHumanoidLocomotionVelocityController(RLControllerBase):
    """
    Velocity-conditioned humanoid RL Locomotion Controller
    Uses contact-implicit reinforcement learning policy
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any]):
        super().__init__(robot=robot, configs=configs)

        self.velocity_commands = torch.tensor([0.5, 0.0, 0.0])
        self.cmd_scale = torch.tensor([2.0, 2.0, 0.25])
        self.max_cmd = torch.tensor([0.8, 0.5, 1.57])

        self.leg_kps = self.Kp
        self.leg_kds = self.Kd
        self.arm_waist_kps = configs["controller_config"]["arm_waist_kps"]
        self.arm_waist_kds = configs["controller_config"]["arm_waist_kds"]
        self.leg_joint2motor_idx = configs["controller_config"]["leg_joint2motor_idx"]
        self.arm_waist_joint2motor_idx = configs["controller_config"]["arm_waist_joint2motor_idx"]
        self.actions_mapping = self.leg_joint2motor_idx
        self.policy_joint_indices = self.leg_joint2motor_idx

        self.counter = 0
        self.period = 0.8

    def set_mode(self):
        """Runs when the mode is changed in the UI."""
        super().set_mode()

    def change_commands(self, new_commands: Dict[str, Any]):
        """
        Change velocity commands with validation.

        :param new_commands: Dictionary of new command values
        """

        try:
            # Create a new tensor with the updated values
            new_velocity_commands = self.velocity_commands.clone()

            if "x_velocity" in new_commands:
                new_velocity_commands[0] = new_commands["x_velocity"]
            if "y_velocity" in new_commands:
                new_velocity_commands[1] = new_commands["y_velocity"]
            if "yaw" in new_commands:
                new_velocity_commands[2] = new_commands["yaw"]

            self.velocity_commands = new_velocity_commands

            if self.logger is not None:
                self.logger.debug(f"Command Updated: {new_commands}")
        except ValueError as e:
            # Log error or handle validation failure
            if self.logger is not None:
                self.logger.error(f"Command update failed: {e}")

    def register_commands(self):
        self.command_manager.register(
            "x_velocity",
            CommandTerm(
                type=float,
                name="x_velocity",
                description="X Velocity (m/s)",
                min_value=-self.max_cmd[0],
                max_value=self.max_cmd[0],
                default_value=0.0,
        ),)

        self.command_manager.register(
            "y_velocity",
            CommandTerm(
                type=float,
                name="y_velocity",
                description="Y Velocity (m/s)",
                min_value=-self.max_cmd[1],
                max_value=self.max_cmd[1],
                default_value=0.0,
        ),)

        self.command_manager.register(
            "yaw",
            CommandTerm(
                type=float,
                name="yaw_rate",
                description="Yaw Rate (rad/s)",
                min_value=-self.max_cmd[2],
                max_value=self.max_cmd[2],
                default_value=0.0,
        ),)

    def register_observations(self):
        """
        Register observations for velocity-conditioned locomotion. Maintains a specific order for direct policy input.
        Lambda is used to get the latest value from the class variables.
        """
        
        from state_manager.observations import (
            ang_vel_b,
            joint_pos_rel,
            joint_vel,
            last_action,
            lin_vel_b,
            phase,
            projected_gravity_b,
            velocity_commands,
        )

        self.obs_manager.register("lin_vel_b", ObsTerm(lin_vel_b))
        self.obs_manager.register("ang_vel_b", ObsTerm(ang_vel_b, params={"scale": 0.25}))
        self.obs_manager.register("projected_gravity", ObsTerm(projected_gravity_b))
        self.obs_manager.register(
            "velocity_commands",
            ObsTerm(
                velocity_commands,
                params={
                    "velocity_commands": lambda: self.velocity_commands,
                    "scale": self.cmd_scale,
        },),)
        self.obs_manager.register(
            "joint_pos",
            ObsTerm(
                joint_pos_rel,
                params={
                    "default_joint_pos": self.default_joint_pos.numpy(),
                    "mapping": self.leg_joint2motor_idx,
        },),)
        self.obs_manager.register(
            "joint_vel",
            ObsTerm(joint_vel, params={"scale": 0.05, "mapping": self.leg_joint2motor_idx}),
        )
        self.obs_manager.register(
            "last_action",
            ObsTerm(last_action, params={"last_action": lambda: self.raw_action}),
        )
        self.obs_manager.register(
            "phase",
            ObsTerm(
                phase,
                params={
                    "counter": lambda: self.counter,
                    "period": self.period,
                    "control_dt": self.control_dt,
        },),)

    def compute_torques(self, state, desired_goal):
        """
        Compute motor commands using the learned policy.

        :param state: Current robot state
        :param desired_goal: Desired goal state (not used in this implementation)
        :return: Motor commands dictionary
        """
        start_time = time.perf_counter()

        if hasattr(self, "counter"):
            self.counter += 1

        joint_pos_targets = self.compute_joint_pos_targets()
        if self.logger is not None:
            self.logger.debug(f"Joint pos targets: {joint_pos_targets}")
        # Prepare motor commands
        self.cmd = {}
        # Handle leg joints
        for idx, motor_idx in enumerate(self.leg_joint2motor_idx):
            self.cmd[f"motor_{motor_idx}"] = {
                "q": joint_pos_targets[idx],
                "kp": self.leg_kps[idx],
                "dq": 0.0,
                "kd": self.leg_kds[idx],
                "tau": 0.0,
                "mode": 1,
            }
        # Handle arm/waist joints
        for idx, motor_idx in enumerate(self.arm_waist_joint2motor_idx):
            self.cmd[f"motor_{motor_idx}"] = {
                "q": 0.0,
                "kp": self.arm_waist_kps[idx],
                "dq": 0.0,
                "kd": self.arm_waist_kds[idx],
                "tau": 0.0,
                "mode": 1,
            }

        # Track command preparation time
        self.cmd_preparation_time = time.perf_counter() - start_time
        return self.cmd

    def get_joystick_mappings(self):
        """
        Define joystick button mappings for gait changes and heading control.

        Returns:
            Dict mapping button names to callback functions.
        """
        return {

            # X-Velocity
            "up": lambda: self.change_commands({
                "x_velocity": self.velocity_commands[0] + 0.1,
                "y_velocity": self.velocity_commands[1],
            }),
            "down": lambda: self.change_commands({
                "x_velocity": self.velocity_commands[0] - 0.1,
                "y_velocity": self.velocity_commands[1],
            }),
            # Y-Velocity
            "left": lambda: self.change_commands({
                "x_velocity": self.velocity_commands[0],
                "y_velocity": self.velocity_commands[1] + 0.1,
            }),
            "right": lambda: self.change_commands({
                "x_velocity": self.velocity_commands[0],
                "y_velocity": self.velocity_commands[1] - 0.1,
        }),}
import time
from typing import Any, Dict, TYPE_CHECKING

import torch
import numpy as np

from commands.command_manager import CommandTerm
from controllers.rl_controller_base import RLControllerBase
from state_manager.obs_manager import ObsTerm

if TYPE_CHECKING:
    from robots.robot_base import RobotBase


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
            ),
        )

        self.command_manager.register(
            "y_velocity",
            CommandTerm(
                type=float,
                name="y_velocity",
                description="Y Velocity (m/s)",
                min_value=-1.0,
                max_value=1.0,
                default_value=0.0,
            ),
        )

        self.command_manager.register(
            "yaw",
            CommandTerm(
                type=float,
                name="yaw_rate",
                description="Yaw Rate (rad/s)",
                min_value=-3.14,
                max_value=3.14,
                default_value=0.0,
            ),
        )

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
            ),
        )
        self.obs_manager.register(
            "joint_pos",
            ObsTerm(
                joint_pos_rel,
                params={
                    "default_joint_pos": self.default_joint_pos.numpy(),
                    "mapping": self.joint_obs_unitree_to_isaac_mapping,
                },
            ),
        )
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
            "up": lambda: self.change_commands(
                {
                    "x_velocity": self.velocity_commands[0] + 0.1,
                    "y_velocity": self.velocity_commands[1],
                }
            ),
            "down": lambda: self.change_commands(
                {
                    "x_velocity": self.velocity_commands[0] - 0.1,
                    "y_velocity": self.velocity_commands[1],
                }
            ),
            # Y-Velocity
            "left": lambda: self.change_commands(
                {
                    "x_velocity": self.velocity_commands[0],
                    "y_velocity": self.velocity_commands[1] + 0.1,
                }
            ),
            "right": lambda: self.change_commands(
                {
                    "x_velocity": self.velocity_commands[0],
                    "y_velocity": self.velocity_commands[1] - 0.1,
                }
            ),
        }

    def compute_lowlevelcmd(self, state):
        """
        Compute motor commands using the learned policy.

        :param state: Current robot state
        :return: Motor commands dictionary
        """
        super().compute_lowlevelcmd(state)

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

        self.velocity_commands = torch.tensor([0.0, 0.0, 0.0])
        self.max_cmd = torch.tensor([0.8, 0.5, 1.57])
        

        # self.leg_kps = self.Kp
        # self.leg_kds = self.Kd
        # self.arm_waist_kps = configs["controller_config"]["arm_waist_kps"]
        # self.arm_waist_kds = configs["controller_config"]["arm_waist_kds"]
        # self.leg_joint2motor_idx = configs["controller_config"]["leg_joint2motor_idx"]
        # self.arm_waist_joint2motor_idx = configs["controller_config"]["arm_waist_joint2motor_idx"]
        # self.actions_mapping = self.leg_joint2motor_idx
        # self.policy_joint_indices = self.leg_joint2motor_idx

        # self.Kp = [
        #     200, 150, 150, 200, 20, 20,
        #     200, 150, 150, 200, 20, 20,
        #     200, 200, 200,
        #     35, 35, 35, 35, 35, 35, 35,
        #     35, 35, 35, 35, 35, 35, 35,
        # ]
        # self.Kd = [
        #     5, 5, 5, 5, 2, 2,
        #     5, 5, 5, 5, 2, 2,
        #     5, 5, 5,
        #     12, 12, 12, 12, 12, 12, 12,
        #     12, 12, 12, 12, 12, 12, 12,
        # ]

        self.leg_joint2motor_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.arm_waist_joint2motor_idx = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        self.actions_mapping = self.leg_joint2motor_idx 
        # self.Kp = [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40] + [300, 300, 300,
        #                                                                         100, 100, 50, 50, 20, 20, 20,
        #                                                                         100, 100, 50, 50, 20, 20, 20]
        # self.Kd = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2] + [3, 3, 3,
        #                                                     2, 2, 2, 2, 1, 1, 1,
        #                                                     2, 2, 2, 2, 1, 1, 1]

        self.Kp = [
            60,
            60,
            60,
            100,
            40,
            40,  # legs
            60,
            60,
            60,
            100,
            40,
            40,  # legs
            60,
            40,
            40,  # waist
            40,
            40,
            40,
            40,
            40,
            40,
            40,  # arms
            40,
            40,
            40,
            40,
            40,
            40,
            40,  # arms
        ]
        self.Kd = [
            1,
            1,
            1,
            2,
            1,
            1,  # legs
            1,
            1,
            1,
            2,
            1,
            1,  # legs
            1,
            1,
            1,  # waist
            1,
            1,
            1,
            1,
            1,
            1,
            1,  # arms
            1,
            1,
            1,
            1,
            1,
            1,
            1,  # arms
        ]

        self.counter = 0

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
            ),
        )

        self.command_manager.register(
            "y_velocity",
            CommandTerm(
                type=float,
                name="y_velocity",
                description="Y Velocity (m/s)",
                min_value=-self.max_cmd[1],
                max_value=self.max_cmd[1],
                default_value=0.0,
            ),
        )

        self.command_manager.register(
            "yaw",
            CommandTerm(
                type=float,
                name="yaw_rate",
                description="Yaw Rate (rad/s)",
                min_value=-self.max_cmd[2],
                max_value=self.max_cmd[2],
                default_value=0.0,
            ),
        )

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
                params={
                    "velocity_commands": lambda: self.velocity_commands,
                },
            ),
        )

        self.obs_manager.register(
            "joint_pos",
            ObsTerm(
                joint_pos_rel,
                params={
                    "default_joint_pos": self.default_joint_pos.numpy(),
                    "mapping": self.joint_obs_unitree_to_isaac_mapping,
                },
            ),
        )
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
            "up": lambda: self.change_commands(
                {
                    "x_velocity": self.velocity_commands[0] + 0.1,
                    "y_velocity": self.velocity_commands[1],
                }
            ),
            "down": lambda: self.change_commands(
                {
                    "x_velocity": self.velocity_commands[0] - 0.1,
                    "y_velocity": self.velocity_commands[1],
                }
            ),
            # Y-Velocity
            "left": lambda: self.change_commands(
                {
                    "x_velocity": self.velocity_commands[0],
                    "y_velocity": self.velocity_commands[1] + 0.1,
                }
            ),
            "right": lambda: self.change_commands(
                {
                    "x_velocity": self.velocity_commands[0],
                    "y_velocity": self.velocity_commands[1] - 0.1,
                }
            ),
        }


class RLHumanoidUnitreeLocomotionVelocityController(RLControllerBase):
    """
    Velocity-conditioned humanoid RL Locomotion Controller for Unitree G1
    Uses contact-implicit reinforcement learning policy
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any]):
        super().__init__(robot=robot, configs=configs)
        self.device = "cuda:0"
        self.velocity_commands = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.max_cmd = torch.tensor([0.8, 0.5, 1.57])

        self.arm_waist_kps = configs["controller_config"]["arm_waist_kps"]
        self.arm_waist_kds = configs["controller_config"]["arm_waist_kds"]

        self.leg_joint2motor_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], device=self.device)
        self.arm_waist_joint2motor_idx = torch.tensor([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], device=self.device)
        self.Kp = configs["controller_config"]["stiffness"]
        self.Kd = configs["controller_config"]["damping"]

        self.leg_kps = self.Kp
        self.leg_kds = self.Kd

        self.counter = 0
        
        # Performance optimization: Pre-allocate tensors on GPU
        
        self.obs_tensor = torch.zeros(1, configs["controller_config"]["obs_dim"], dtype=torch.float32, device=self.device)  # Adjust size based on actual obs dim
        self.joint_pos_buffer = torch.zeros(12, dtype=torch.float32, device=self.device)
        self.joint_vel_buffer = torch.zeros(12, dtype=torch.float32, device=self.device)
        self.ang_vel_buffer = torch.zeros(3, dtype=torch.float32, device=self.device)
        self.gravity_buffer = torch.zeros(3, dtype=torch.float32, device=self.device)
        self.cmd_buffer = torch.zeros(3, dtype=torch.float32, device=self.device)
        self.action_buffer = torch.zeros(12, dtype=torch.float32, device=self.device)  # Adjust size based on action dim
        self.phase_buffer = torch.zeros(2, dtype=torch.float32, device=self.device)
        
        # Pre-compute scaling factors
        self.ang_vel_scale = 0.25
        self.joint_vel_scale = 0.05
        
        self.cmd_scale = torch.tensor([2.0, 2.0, 0.25], device=self.device)
        
        # Cache for phase computation
        self.period = 0.8
        self.two_pi = 2 * np.pi
        
        # Pre-allocate motor command dictionary
        self.G1_NUM_MOTOR = 29
        self.cmd = {f"motor_{i}": {"q": 0.0, "kp": 0.0, "dq": 0.0, "kd": 0.0, "tau": 0.0} for i in range(self.G1_NUM_MOTOR)}
        self.cmd["mode_pr"] = 0  # Mode.PR
        
        # Move all tensors to GPU for consistency
        self.default_joint_pos = self.default_joint_pos.to(self.device)
        self.action_scale = self.action_scale.to(self.device)
        self.velocity_commands = self.velocity_commands.to(self.device)
        
        # Initialize raw_action on GPU if not already done
        if not hasattr(self, 'raw_action') or self.raw_action is None:
            self.raw_action = torch.zeros(12, dtype=torch.float32, device=self.device)
        else:
            self.raw_action = self.raw_action.to(self.device)
        
        # Pre-allocate joint position targets on GPU
        self.joint_pos_targets = self.default_joint_pos.clone()

    def set_mode(self):
        """Runs when the mode is changed in the UI."""
        super().set_mode()
        self.velocity_commands = torch.tensor([0.0, 0.0, 0.0], device=self.device)

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
                min_value=float(-self.max_cmd[0]),
                max_value=float(self.max_cmd[0]),
                default_value=0.0,
            ),
        )

        self.command_manager.register(
            "y_velocity",
            CommandTerm(
                type=float,
                name="y_velocity",
                description="Y Velocity (m/s)",
                min_value=float(-self.max_cmd[1]),
                max_value=float(self.max_cmd[1]),
                default_value=0.0,
            ),
        )

        self.command_manager.register(
            "yaw",
            CommandTerm(
                type=float,
                name="yaw_rate",
                description="Yaw Rate (rad/s)",
                min_value=float(-self.max_cmd[2]),
                max_value=float(self.max_cmd[2]),
                default_value=0.0,
            ),
        )

    def register_observations(self):
        """
        Register observations for velocity-conditioned locomotion. Maintains a specific order for direct policy input.
        Lambda is used to get the latest value from the class variables.
        """

        from state_manager.observations import (
            ang_vel_b,
            unitree_gravity_orientation,
            joint_pos_rel,
            joint_vel,
            last_action,
            phase_with_timing,
            velocity_commands,
        )

        self.obs_manager.register("ang_vel_b", ObsTerm(ang_vel_b, params={"scale": 0.25}))
        self.obs_manager.register("projected_gravity", ObsTerm(unitree_gravity_orientation))
        self.obs_manager.register(
            "velocity_commands",
            ObsTerm(
                velocity_commands,
                params={"velocity_commands": lambda: self.velocity_commands * self.cmd_scale},
            ),
        )

        self.obs_manager.register(
            "joint_pos",
            ObsTerm(
                joint_pos_rel,
                params={
                    "default_joint_pos": self.default_joint_pos,
                    "mapping": self.leg_joint2motor_idx,
                },
            ),
        )
        self.obs_manager.register(
            "joint_vel",
            ObsTerm(joint_vel, params={"mapping": self.leg_joint2motor_idx, "scale": 0.05}),
        )
        self.obs_manager.register(
            "last_action",
            ObsTerm(last_action, params={"last_action": lambda: self.raw_action}),
        )
        # self.obs_manager.register(
        #     "phase",
        #     ObsTerm(phase_with_timing, params={"logger": lambda: self.logger, "period": 0.8, "control_dt": self.control_dt, "decimation": self.decimation}),
        # )

    def _compute_gravity_from_quat(self, quat):
        """Optimized gravity computation from quaternion."""
        qw, qx, qy, qz = quat
        self.gravity_buffer[0] = 2 * (-qz * qx + qw * qy)
        self.gravity_buffer[1] = -2 * (qz * qy + qw * qx)
        self.gravity_buffer[2] = 1 - 2 * (qw * qw + qz * qz)
        return self.gravity_buffer

    def _compute_phase(self, count):
        """Optimized phase computation."""
        phase = count % self.period / self.period
        self.phase_buffer[0] = np.sin(self.two_pi * phase)
        self.phase_buffer[1] = np.cos(self.two_pi * phase)
        return self.phase_buffer

    def _build_observation_tensor(self, state):
        """Build observation tensor efficiently using pre-allocated buffers."""
        # Update buffers with new data (minimal device transfers)
        joint_pos = torch.tensor(state["robot/joint_pos"][:12], dtype=torch.float32, device=self.device)
        self.joint_pos_buffer.copy_(joint_pos - self.default_joint_pos)
        
        self.joint_vel_buffer.copy_(torch.tensor(state["robot/joint_vel"][:12], dtype=torch.float32, device=self.device) * self.joint_vel_scale)
        self.ang_vel_buffer.copy_(torch.tensor(state["robot/gyroscope"], dtype=torch.float32, device=self.device) * self.ang_vel_scale)
        
        # Compute gravity efficiently
        self._compute_gravity_from_quat(state["robot/base_quat"])
        
        # Update command and action buffers - ensure they're on the right device
        if not isinstance(self.velocity_commands, torch.Tensor):
            self.velocity_commands = torch.tensor(self.velocity_commands, dtype=torch.float32, device=self.device)
        self.cmd_buffer.copy_(self.velocity_commands)
        
        # Ensure raw_action is a tensor on the correct device
        if not isinstance(self.raw_action, torch.Tensor):
            self.raw_action = torch.tensor(self.raw_action, dtype=torch.float32, device=self.device)
        elif self.raw_action.device != self.device:
            self.raw_action = self.raw_action.to(self.device)
        self.action_buffer.copy_(self.raw_action)
        
        # Compute phase
        count = self.counter * self.control_dt
        self._compute_phase(count)
        
        # Concatenate efficiently using pre-allocated tensor
        # Assuming observation order: ang_vel(3) + gravity(3) + cmd(3) + qj(12) + dqj(12) + action(12) + phase(2) = 47
        start_idx = 0
        self.obs_tensor[0, start_idx:start_idx+3] = self.ang_vel_buffer
        start_idx += 3
        self.obs_tensor[0, start_idx:start_idx+3] = self.gravity_buffer
        start_idx += 3
        self.obs_tensor[0, start_idx:start_idx+3] = self.cmd_buffer
        start_idx += 3
        self.obs_tensor[0, start_idx:start_idx+12] = self.joint_pos_buffer
        start_idx += 12
        self.obs_tensor[0, start_idx:start_idx+12] = self.joint_vel_buffer
        start_idx += 12
        self.obs_tensor[0, start_idx:start_idx+12] = self.action_buffer
        start_idx += 12
        self.obs_tensor[0, start_idx:start_idx+2] = self.phase_buffer
        
        return self.obs_tensor

    def compute_lowlevelcmd(self, state):
        """
        Compute motor commands using the learned policy.

        :param state: Current robot state
        :return: Motor commands dictionary
        """
        start_time = time.perf_counter()
        self.counter += 1
        
        if self.counter % self.decimation == 0:
            # Build observation tensor efficiently
            obs_tensor = self._build_observation_tensor(state)
            
            obs_time = time.perf_counter() - start_time
            if self.logger:
                self.logger.debug(f"Obs preparation time (in seconds): {obs_time}")
            
            # Policy inference
            policy_start_time = time.perf_counter()
            with torch.no_grad():
                action = self.policy(obs_tensor)
                self.raw_action = action.detach().squeeze()
            policy_time = time.perf_counter() - policy_start_time
            if self.logger:
                self.logger.debug(f"Policy execution time (in seconds): {policy_time}")
            
            # Update joint position targets
            self.joint_pos_targets = self.default_joint_pos + self.raw_action * self.action_scale
        
        if not hasattr(self, "joint_pos_targets"):
            self.joint_pos_targets = self.default_joint_pos.clone()

        # Build motor commands efficiently using pre-allocated dictionary
        # Leg joints
        for i in range(len(self.leg_joint2motor_idx)):
            motor_idx = self.leg_joint2motor_idx[i]
            self.cmd[f"motor_{motor_idx}"].update({
                "q": float(self.joint_pos_targets[i].cpu().item()),
                "kp": float(self.leg_kps[i]),
                "dq": 0.0,
                "kd": float(self.leg_kds[i]),
                "tau": 0.0,
            })

        # Arm/waist joints
        for i in range(len(self.arm_waist_joint2motor_idx)):
            motor_idx = self.arm_waist_joint2motor_idx[i]
            self.cmd[f"motor_{motor_idx}"].update({
                "q": 0.0,
                "kp": float(self.arm_waist_kps[i]),
                "dq": 0.0,
                "kd": float(self.arm_waist_kds[i]),
                "tau": 0.0,
            })
        
        self.cmd_preparation_time = time.perf_counter() - start_time
        if self.logger:
            self.logger.debug(f"Command preparation time (in seconds): {self.cmd_preparation_time}")
        return self.cmd

    def get_joystick_mappings(self):
        """
        Define joystick button mappings for gait changes and heading control.

        Returns:
            Dict mapping button names to callback functions.
        """
        return {
            # X-Velocity
            "up": lambda: self.change_commands(
                {
                    "x_velocity": self.velocity_commands[0] + 0.1,
                    "y_velocity": self.velocity_commands[1],
                }
            ),
            "down": lambda: self.change_commands(
                {
                    "x_velocity": self.velocity_commands[0] - 0.1,
                    "y_velocity": self.velocity_commands[1],
                }
            ),
            # Y-Velocity
            "left": lambda: self.change_commands(
                {
                    "x_velocity": self.velocity_commands[0],
                    "y_velocity": self.velocity_commands[1] + 0.1,
                }
            ),
            "right": lambda: self.change_commands(
                {
                    "x_velocity": self.velocity_commands[0],
                    "y_velocity": self.velocity_commands[1] - 0.1,
                }
            ),
        }

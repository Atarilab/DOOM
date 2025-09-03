import time
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import torch

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
        self.velocity_commands = torch.tensor([0.0, 0.0, 0.0], device=self.device)

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

        self.obs_manager.register("lin_vel_b", ObsTerm(lin_vel_b, obs_dim=3, device=self.device))
        self.obs_manager.register("ang_vel_b", ObsTerm(ang_vel_b, obs_dim=3, device=self.device))
        self.obs_manager.register(
            "velocity_commands",
            ObsTerm(
                velocity_commands,
                params={"velocity_commands": lambda: self.velocity_commands},
                obs_dim=3,
                device=self.device,
            ),
        )
        self.obs_manager.register("projected_gravity", ObsTerm(projected_gravity_b, obs_dim=3, device=self.device))
        self.obs_manager.register(
            "joint_pos",
            ObsTerm(
                joint_pos_rel,
                params={
                    "default_joint_pos": self.default_joint_pos_np,
                    "mapping": self.joint_obs_unitree_to_isaac_mapping,
                },
                obs_dim=12,
                device=self.device,
            ),
        )
        self.obs_manager.register(
            "joint_vel",
            ObsTerm(joint_vel, params={"mapping": self.joint_obs_unitree_to_isaac_mapping}, obs_dim=12, device=self.device),
        )
        self.obs_manager.register(
            "last_action",
            ObsTerm(last_action, params={"last_action": lambda: self.raw_action}, obs_dim=12, device=self.device),
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
            if not self.use_threading:
                obs_tensor = self.obs_manager.compute_full_tensor(state, batch_idx=0)
                obs = self.obs_manager.get_from_buffer().squeeze()
                
                                    
                joint_pos_targets = self.compute_joint_pos_targets_from_policy(obs)
            else:
                joint_pos_targets = self.compute_joint_pos_targets()
                
            # Clip the joint pos targets for safety
            if hasattr(self, "soft_dof_pos_limit"):
                joint_pos_targets = self._clip_dof_pos(joint_pos_targets)

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
    

class RLQuadrupedLocomotionVelocityControllerTorque(RLQuadrupedLocomotionVelocityController):
    """
    Velocity-conditioned quadruped RL Locomotion Controller
    Uses contact-implicit reinforcement learning policy
    """

    def compute_lowlevelcmd(self, state):
        """
        Compute motor commands using the learned policy.

        :param state: Current robot state
        :return: Motor commands dictionary
        """
        if self.robot.mj_model is not None:
            self.robot.mj_model.update(state)

        start_time = time.perf_counter()

        try:
            if not self.use_threading:
                obs_tensor = self.obs_manager.compute_full_tensor(state, batch_idx=0)
                obs = self.obs_manager.get_from_buffer().squeeze()
                
                                    
                torques = self.compute_joint_pos_targets_from_policy(obs)
            else:
                raise ValueError("Should be run w/o threading for faster inference.")
                
            torques = np.clip(torques, -23.5, 23.5)

            # Prepare motor commands
            self.cmd = {
                f"motor_{i}": {
                    "q": 0,
                    "kp": 0,
                    "dq": 0.0,
                    "kd": 0,
                    "tau": torques[i],
                }
                for i in range(12)
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

        # Ensure tensors are on the correct device
        self.velocity_commands = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        # Pre-allocate motor command dictionary
        self.G1_NUM_MOTOR = 29
        self.cmd = {f"motor_{i}": {"q": 0.0, "kp": 0.0, "dq": 0.0, "kd": 0.0, "tau": 0.0} for i in range(self.G1_NUM_MOTOR)}
        self.cmd["mode_pr"] = 0  # Mode.PR
        self.default_joint_pos_np = self.default_joint_pos.cpu().numpy()
        self.Kp = configs["controller_config"]["stiffness"]
        self.Kd = configs["controller_config"]["damping"]
        self.decimation = configs["controller_config"]["decimation"]
        self.control_dt = configs["controller_config"]["control_dt"]
        self.counter = 0
        self.joint_pos_targets = self.default_joint_pos.clone()


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
                min_value=-0.5,
                max_value=0.5,
                default_value=0.0,
            ),
        )

        self.command_manager.register(
            "y_velocity",
            CommandTerm(
                type=float,
                name="y_velocity",
                description="Y Velocity (m/s)",
                min_value=-0.5,
                max_value=0.5,
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

        self.obs_manager.register("lin_vel_b", ObsTerm(lin_vel_b, obs_dim=3, device=self.device))
        self.obs_manager.register("ang_vel_b", ObsTerm(ang_vel_b, obs_dim=3, device=self.device))
        self.obs_manager.register("projected_gravity", ObsTerm(projected_gravity_b, obs_dim=3, device=self.device))
        self.obs_manager.register(
            "velocity_commands",
            ObsTerm(
                velocity_commands,
                params={
                    "velocity_commands": lambda: self.velocity_commands,
                },
                obs_dim=3,
                device=self.device,
            ),
        )

        self.obs_manager.register(
            "joint_pos",
            ObsTerm(
                joint_pos_rel,
                params={
                    "default_joint_pos": self.default_joint_pos_np,
                    "mapping": self.robot.joints_isaac2unitree,
                },
                obs_dim=29,
                device=self.device,
            ),
        )
        self.obs_manager.register(
            "joint_vel",
            ObsTerm(joint_vel, params={"mapping": self.robot.joints_isaac2unitree}, obs_dim=29, device=self.device),
        )
        self.obs_manager.register(
            "last_action",
            ObsTerm(last_action, params={"last_action": lambda: self.raw_action}, obs_dim=29, device=self.device),
        )
        
    def compute_lowlevelcmd(self, state):
        """
        Compute motor commands using the learned policy.

        :param state: Current robot state
        :return: Motor commands dictionary
        """
        super().compute_lowlevelcmd(state)

        start_time = time.perf_counter()
        self.counter += 1
        
        if self.counter % self.decimation == 0:
            try:
                if not self.use_threading:
                    obs_tensor = self.obs_manager.compute_full_tensor(state, batch_idx=0)
                    self.joint_pos_targets = self.compute_joint_pos_targets_from_policy(obs_tensor)
                else:
                    self.joint_pos_targets = self.compute_joint_pos_targets()
                    
                # # Clip the joint pos targets for safety
                # if hasattr(self, "soft_dof_pos_limit"):
                #     self.joint_pos_targets = self._clip_dof_pos(self.joint_pos_targets)

                # Prepare motor commands
                self.cmd = {
                    f"motor_{i}": {
                        "q": self.joint_pos_targets[i],
                        "kp": self.Kp[i],
                        "dq": 0.0,
                        "kd": self.Kd[i],
                        "tau": 0.0,
                    }
                    for i in range(self.robot.num_joints)
                }

                self.logger.debug(f"Joint pos targets: {self.joint_pos_targets}")
                # Track command preparation time
                self.cmd_preparation_time = time.perf_counter() - start_time

            except Exception as e:
                self.logger.error(f"Error computing torques: {e}")
                self.cmd = {
                    f"motor_{i}": {
                        "q": self.default_joint_pos[i],
                        "kp": self.Kp[i],
                        "dq": 0.0,
                        "kd": self.Kd[i],
                        "tau": 0.0,
                    }
                    for i in range(self.robot.num_joints)
                }

        return self.cmd
        
    # def compute_lowlevelcmd(self, state):
    #     """
    #     Compute motor commands using the learned policy.

    #     :param state: Current robot state
    #     :return: Motor commands dictionary
    #     """
    #     start_time = time.perf_counter()
    #     self.counter += 1
        
    #     if self.counter % self.decimation == 0:
    #         # Build observation tensor efficiently
    #         obs_tensor = self.obs_manager.compute_full_tensor(state, batch_idx=0)
    #         self.joint_pos_targets = self.compute_joint_pos_targets_from_policy(obs_tensor)[self.robot.joints_isaac2unitree]
    #         # Clip the joint pos targets for safety
    #         if hasattr(self, "soft_dof_pos_limit"):
    #             self.joint_pos_targets = self._clip_dof_pos(self.joint_pos_targets)
                
    #     for i in range(self.robot.num_joints):
    #         self.cmd[f"motor_{i}"].update({
    #             "q": float(self.joint_pos_targets[i]),
    #             "kp": float(self.Kp[i]),
    #             "dq": 0.0,
    #             "kd": float(self.Kd[i]),
    #             "tau": 0.0,
    #         })

    #     self.cmd_preparation_time = time.perf_counter() - start_time
    #     if self.logger:
    #         self.logger.debug(f"Command preparation time (in seconds): {self.cmd_preparation_time}")
    #     return self.cmd

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
    This is a velocity-conditioned locomotion policy controller using a cyclic phase, provided by Unitree.
    Only the leg joints (12) are actuated, the arm and waist joints (17) are PD controlled to zero positions.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any]):
        super().__init__(robot=robot, configs=configs)
        self.velocity_commands = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.max_cmd = torch.tensor([0.8, 0.5, 1.57])

        self.arm_waist_kps = configs["controller_config"]["arm_waist_kps"]
        self.arm_waist_kds = configs["controller_config"]["arm_waist_kds"]

        self.leg_joint2motor_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], device=self.device)
        self.arm_waist_joint2motor_idx = torch.tensor([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], device=self.device)
        self.actions_mapping = self.leg_joint2motor_idx
        self.Kp = configs["controller_config"]["stiffness"]
        self.Kd = configs["controller_config"]["damping"]

        self.leg_kps = self.Kp
        self.leg_kds = self.Kd
        
        
        # Pre-compute scaling factors
        self.ang_vel_scale = 0.25
        self.joint_vel_scale = 0.05
        self.cmd_scale = torch.tensor([2.0, 2.0, 0.25], device=self.device)
        self.period = 0.8

        
        # Pre-allocate motor command dictionary
        self.G1_NUM_MOTOR = 29
        self.cmd = {f"motor_{i}": {"q": 0.0, "kp": 0.0, "dq": 0.0, "kd": 0.0, "tau": 0.0} for i in range(self.G1_NUM_MOTOR)}
        self.cmd["mode_pr"] = 0  # Mode.PR
        self.default_joint_pos_np = self.default_joint_pos.cpu().numpy()
        # Move all tensors to GPU for consistency
        self.default_joint_pos = self.default_joint_pos.to(self.device)
        self.velocity_commands = self.velocity_commands.to(self.device)
        
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
            joint_pos_rel,
            joint_vel,
            last_action,
            phase_with_timing,
            projected_gravity_b,
            velocity_commands,
        )

        self.obs_manager.register("ang_vel_b", ObsTerm(ang_vel_b, params={"scale": 0.25}, obs_dim=3, device=self.device))
        self.obs_manager.register("projected_gravity", ObsTerm(projected_gravity_b, obs_dim=3, device=self.device))
        self.obs_manager.register(
            "velocity_commands",
            ObsTerm(
                velocity_commands,
                params={"velocity_commands": lambda: self.velocity_commands * self.cmd_scale},
                obs_dim=3,
                device=self.device,
            ),
        )

        self.obs_manager.register(
            "joint_pos",
            ObsTerm(
                joint_pos_rel,
                params={
                    "default_joint_pos": self.default_joint_pos_np,
                    "mapping": self.leg_joint2motor_idx.cpu().numpy(),
                },
                obs_dim=12,
                device=self.device,
            ),
        )
        self.obs_manager.register(
            "joint_vel",
            ObsTerm(joint_vel, obs_dim=12, params={"mapping": self.leg_joint2motor_idx.cpu().numpy(), "scale": 0.05}, device=self.device),
        )
        self.obs_manager.register(
            "last_action",
            ObsTerm(last_action, obs_dim=12, params={"last_action": lambda: self.raw_action}, device=self.device),
        )
        self.obs_manager.register(
            "phase",
            ObsTerm(phase_with_timing, obs_dim=2, params={"counter": lambda: self.counter, "period": 0.8, "control_dt": self.control_dt, "decimation": self.decimation}, device=self.device),
        )

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
            obs_tensor = self.obs_manager.compute_full_tensor(state, batch_idx=0)
            self.joint_pos_targets = self.compute_joint_pos_targets_from_policy(obs_tensor)
            # Clip the joint pos targets for safety
            if hasattr(self, "soft_dof_pos_limit"):
                self.joint_pos_targets = self._clip_dof_pos(self.joint_pos_targets, joint_indices=self.leg_joint2motor_idx.cpu().numpy())


        # Build motor commands efficiently using pre-allocated dictionary
        # Leg joints
        for i in range(len(self.leg_joint2motor_idx)):
            motor_idx = self.leg_joint2motor_idx[i]
            self.cmd[f"motor_{motor_idx}"].update({
                "q": float(self.joint_pos_targets[i]),
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

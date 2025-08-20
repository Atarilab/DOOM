import threading
import time
from typing import TYPE_CHECKING, Any, Dict

import torch

from commands.command_manager import CommandTerm
from controllers.rl_controller_base import RLControllerBase
from state_manager.obs_manager import ObsTerm
from utils.math import combine_frame_transforms, subtract_frame_transforms

if TYPE_CHECKING:
    from robots.robot_base import RobotBase


class RLHumanoidBimanualContactController(RLControllerBase):
    """
    Contact-conditioned RL Bimanual Controller
    Uses contact-explicit reinforcement learning policy
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any]):

        super().__init__(robot=robot, configs=configs)

        
        # Contact command parameters
        self.command_duration = 1.0  # Duration of each contact plan in seconds
        self.object_size = torch.tensor([0.25, 0.25, 0.25], dtype=torch.float32, device=self.device)

        self.repose_contact_plan_ = torch.tensor(
            [
                [
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                [
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            ], 
            dtype=torch.bool, device=self.device
        )

        self.command_start_time = time.time()  # When the current plan started
        self._command_lock = threading.RLock()  # Reentrant lock for gait changes

        self.time_left = self.command_duration
        self.current_goal_idx = 0
        self.current_contact_plan = self.repose_contact_plan_[:, self.current_goal_idx : 2]
        self.goal_completion_counter = 0

        self.leg_joint2motor_idx = configs["controller_config"]["leg_joint2motor_idx"]
        self.arm_waist_joint2motor_idx = configs["controller_config"]["arm_waist_joint2motor_idx"]
        
        self.default_leg_pos = [-0.1,  0.0,  0.0,  0.3, -0.2, 0.0, 
                  -0.1,  0.0,  0.0,  0.3, -0.2, 0.0]
        
        self.leg_kps = [
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
            
        ]
        
        self.leg_kds = [
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
            
        ]

        self.actuated_joint_names = [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        self.motor_joint_indices = [i-1 for i in self.robot.mj_model.joint_names.values()]


        self.actuated_joint_indices = [
            self.robot.mj_model.joint_names[joint_name] - 1 for joint_name in self.actuated_joint_names
        ]
        self.non_actuated_joint_indices = [i for i in self.motor_joint_indices if i not in self.actuated_joint_indices]

        
        self.repose_contact_pos_o = torch.tensor([
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            ], 
            dtype=torch.float32, device=self.device)
        
        self.repose_contact_pos_o[:, :, 0] *= self.object_size[0] / 2
        self.repose_contact_pos_o[:, :, 1] *= self.object_size[1] / 2
        self.contact_pose_o = torch.zeros(2, 7, dtype=torch.float32, device=self.device)
        
        self.contact_pose_o[:, :3] = self.repose_contact_pos_o[self.current_goal_idx, :, :3]
        self.contact_pose_o[:, 3] = 1.0
        
        self.contact_pose_b = torch.zeros_like(self.contact_pose_o)
        self.contact_pose_w = torch.zeros_like(self.contact_pose_o)
        
        self.actions_mapping = torch.arange(len(self.actuated_joint_indices), dtype=torch.int32, device=self.device)

    def register_observations(self):
        """
        Register observations for contact-conditioned locomotion.
        Includes contact pattern and timing information.
        """
        from state_manager.observations import (
            ang_vel_w,
            contact_locations_b,
            contact_plan,
            contact_time_left,
            contact_pose_b,
            dummy_contact_status,
            ee_pos_rel_b,
            joint_pos_limit_normalized,
            joint_vel,
            last_action,
            lin_vel_w,
            object_size,
            object_pose_command_b,
            goal_pose_diff,
            root_pos_w,
            root_quat_w,
        )

        self.obs_manager.register(
            "joint_pos",
            ObsTerm(
                joint_pos_limit_normalized,
                params={
                    "soft_dof_limits": self.soft_dof_pos_limit,
                    "mapping": self.actuated_joint_indices,
                },
                obs_dim=len(self.actuated_joint_indices),
                device=self.device,
            ),
        )
        self.obs_manager.register(
            "joint_vel",
            ObsTerm(
                joint_vel,
                params={
                    "mapping": self.actuated_joint_indices,
                },
                obs_dim=len(self.actuated_joint_indices),
                device=self.device,
            ),
        )

        self.obs_manager.register(
            "object_pos_w",
            ObsTerm(
                root_pos_w,
                params={"asset_name": "object"},
                obs_dim=3,
                device=self.device,
            ),
        )

        self.obs_manager.register(
            "object_quat",
            ObsTerm(
                root_quat_w,
                params={"asset_name": "object"},
                obs_dim=4,
                device=self.device,
            ),
        )

        self.obs_manager.register("object_lin_vel_w", ObsTerm(lin_vel_w, params={"asset_name": "object"}, obs_dim=3, device=self.device))
        self.obs_manager.register("object_ang_vel_w", ObsTerm(ang_vel_w, params={"asset_name": "object"}, obs_dim=3, device=self.device))

        # Register contact pose observation
        self.obs_manager.register(
            "contact_pose_b",
            ObsTerm(
                contact_pose_b,
                params={"contact_pose_b": lambda: self.contact_pose_b},
                obs_dim=14,  # 2 end-effectors * 7 (3 pos + 4 quat)
                device=self.device,
            ),
        )


        self.obs_manager.register(
            "contact_time_left",
            ObsTerm(contact_time_left, params={"contact_time_left": lambda: self.time_left}, obs_dim=1, device=self.device),
        )
        self.obs_manager.register(
            "contact_plan",
            ObsTerm(
                contact_plan,
                params={"contact_plan": lambda: self.current_contact_plan},
                obs_dim=4,
                device=self.device,
            ),
        )
        
        self.obs_manager.register("object_pose_command_b", ObsTerm(object_pose_command_b, obs_dim=7, device=self.device))
        self.obs_manager.register("goal_pose_diff", ObsTerm(goal_pose_diff, params={"asset_name": "object"}, obs_dim=7, device=self.device))

        #########################################
        self.obs_manager.register(
            "last_action",
            ObsTerm(last_action, params={"last_action": lambda: self.raw_action}, obs_dim=len(self.actuated_joint_indices), device=self.device),
        )

    def register_commands(self):
        """Register contact command parameters."""
        self.command_manager.register(
            "task",
            CommandTerm(
                type=str,
                name="task",
                description="Task to perform (repose, reorient)",
                default_value="repose",
            ),
        )

    def set_mode(self):
        """Runs when the mode is changed in the UI.
        Generates the future feet positions in the init frame for horizon planning.
        """
        # Call the base class set_mode to activate the controller
        super().set_mode()

        # Set default gait to stance when switching back to RL controller
        with self._command_lock:
            self.current_contact_plan = self.repose_contact_plan_[:, 0:2]

            self.command_duration = 1.0

    def compute_lowlevelcmd(self, state):
        """
        Compute motor commands using the learned policy.

        :param state: Current robot state
        :return: Motor commands dictionary
        """
        if self.robot.mj_model is not None:
            self.robot.mj_model.update(state)

        start_time = time.perf_counter()

        # Update the latest state for the observation processing thread
        with self._lock:
            self.latest_state = state

        # Update contact plan timing with thread safety
        with self._command_lock:
            current_time = time.time()
            elapsed = current_time - self.command_start_time
            self.time_left = max(0, self.command_duration - elapsed)
            if elapsed >= self.command_duration:
                self.command_start_time = current_time
                self._resample_commands()

        # Compute contact poses
        self._update_contact_poses()

        try:
            # If we do not use threading, we need to compute the obs first, pass it to the policy, and then compute the joint pos targets
            # else we can compute the joint pos targets directly from the obs tensor
            if self.counter % self.decimation == 0:
                if not self.use_threading:
                    obs_tensor = self.obs_manager.compute_full_tensor(self.latest_state, batch_idx=0)
                    self.joint_pos_targets[self.actuated_joint_indices] = self.compute_joint_pos_targets_from_policy(obs_tensor)
                else:
                    self.joint_pos_targets[self.actuated_joint_indices] = self.compute_joint_pos_targets()
                    
            self.joint_pos_targets[self.non_actuated_joint_indices] = 0.0
                
            # # Clip the joint pos targets for safety
            # if hasattr(self, "soft_dof_pos_limit"):
            #     self.joint_pos_targets = self._clip_dof_pos(self.joint_pos_targets)
                
            # First, set commands for actuated joints as before
            
            for idx, joint_idx in enumerate(self.actuated_joint_indices):
                self.cmd[f"motor_{joint_idx}"] = {
                    "q": self.joint_pos_targets[joint_idx],
                    # "q": self.default_joint_pos_np[idx],
                    "kp": self.Kp[idx],
                    "dq": 0.0,
                    "kd": self.Kd[idx],
                    "tau": 0.0,
                }
            for idx, joint_idx in enumerate(self.non_actuated_joint_indices):
                self.cmd[f"motor_{joint_idx}"] = {
                    "q": self.default_leg_pos[idx],
                    # "kp": self.leg_kps[idx],
                    "kp": 0.0,
                    "dq": 0.0,
                    # "kd": self.leg_kds[idx],
                    "kd": 0.0,
                    "tau": 0.0,
                }

            # Track command preparation time
            self.cmd_preparation_time = time.perf_counter() - start_time

        except Exception as e:
            self.logger.error(f"Error computing torques: {e}")
            for i, joint_idx in enumerate(self.actuated_joint_indices):
                self.cmd[f"motor_{joint_idx}"] = {
                    "q": self.default_joint_pos_np[i],
                    "kp": 0.0,
                    "dq": 0.0,
                    "kd": 0.0,
                    "tau": 0.0,
                }
            for idx, joint_idx in enumerate(self.non_actuated_joint_indices):
                self.cmd[f"motor_{joint_idx}"] = {
                    "q": self.default_leg_pos[idx],
                    "kp": self.leg_kps[idx],
                    "dq": 0.0,
                    "kd": self.leg_kds[idx],
                    "tau": 0.0,
                }

        return self.cmd

    def _update_contact_poses(self):
        """Update contact poses in world and base frames."""
        try:
            # Check if required state keys exist
            if "object/base_pos_w" not in self.latest_state or "object/base_quat" not in self.latest_state:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.warning("Object pose not found in state, skipping contact pose update")
                return
                
            if "robot/base_pos_w" not in self.latest_state or "robot/base_quat" not in self.latest_state:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.warning("Robot pose not found in state, skipping contact pose update")
                return
            
            # Get object pose from state
            object_pos_w = torch.tensor(self.latest_state["object/base_pos_w"], dtype=torch.float32, device=self.device)
            object_quat_w = torch.tensor(self.latest_state["object/base_quat"], dtype=torch.float32, device=self.device)
            
            # Get robot base pose
            robot_pos_w = torch.tensor(self.latest_state["robot/base_pos_w"], dtype=torch.float32, device=self.device)
            robot_quat_w = torch.tensor(self.latest_state["robot/base_quat"], dtype=torch.float32, device=self.device)
            
            # Contact poses in the world frame
            self.contact_pose_w[:, :3], self.contact_pose_w[:, 3:7] = combine_frame_transforms(
                t01=object_pos_w.unsqueeze(0).expand(2, -1),
                q01=object_quat_w.unsqueeze(0).expand(2, -1),
                t12=self.contact_pose_o[:, :3],
                q12=self.contact_pose_o[:, 3:7],
            )

            # Contact poses in the base frame (for policy observations)
            self.contact_pose_b[:, :3], self.contact_pose_b[:, 3:7] = subtract_frame_transforms(
                t01=robot_pos_w.unsqueeze(0).expand(2, -1),
                q01=robot_quat_w.unsqueeze(0).expand(2, -1),
                t02=self.contact_pose_w[:, :3],
                q02=self.contact_pose_w[:, 3:7],
            )
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"Error updating contact poses: {e}")
            else:
                print(f"Error updating contact poses: {e}")

    def _update_contact_pose_o(self):
        """Update contact pose_o based on current goal index."""
        try:
            # Update contact pose_o based on current goal index
            if self.current_goal_idx < self.repose_contact_pos_o.shape[0]:
                self.contact_pose_o[:, :3] = self.repose_contact_pos_o[self.current_goal_idx, :, :3]
                self.contact_pose_o[:, 3] = 1.0  # Set quaternion w component to 1 (identity)
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"Error updating contact pose_o: {e}")
            else:
                print(f"Error updating contact pose_o: {e}")

    def _resample_commands(self):
        """Resample the commands for the controller with thread safety."""
        try:
            with self._command_lock:
                
                self.current_goal_idx += 1
                
                # Update contact pose_o based on current goal index
                self._update_contact_pose_o()

                if self.current_goal_idx >= self.repose_contact_pos_o.shape[0]-1:
                    self.current_goal_idx = 0
                    
                self.logger.info(f"Current goal index: {self.current_goal_idx}")

        except Exception as e:
            if hasattr(self, "command_manager") and self.command_manager and hasattr(self.command_manager, "logger"):
                self.logger.error(f"Error resampling commands: {e}")
            else:
                print(f"Error resampling commands: {e}")

    
    """
    Joystick mappings and callbacks        
    """

    def get_joystick_mappings(self):
        """
        Define joystick button mappings for gait changes and heading control.

        Returns:
            Dict mapping button names to callback functions.
        """
        return {
            # Reset to stance
            "L2-R2": lambda: self.change_commands({"gait": "stance"}),
            # Normal Gaits
            "A": lambda: self.change_commands({"gait": "trot"}),
            # Step Size
            "up": lambda: self._handle_step_size_change("up"),
            "down": lambda: self._handle_step_size_change("down"),
            # Command Duration
            "L1-up": lambda: self._handle_command_duration_change("increase"),
            "L1-down": lambda: self._handle_command_duration_change("decrease"),
            "L1-R1": lambda: self._handle_reset(),
        }

    """
    Function handlers for changing the contact commands for the rl-contact-locomotion controller
    - gait, step size, heading, lateral position, command duration, stance width
    """

    def change_commands(self, new_commands: Dict[str, Any]):
        """Change the robot's contact commands with thread safety.

        This method handles changes to the robot's gait pattern. When a new gait is requested,
        it is stored as a pending change rather than applied immediately. The actual gait
        transition happens during the next resampling phase to ensure smooth transitions.

        Args:
            new_commands: Dictionary containing command updates. Currently supports:
                - 'gait': String specifying the new gait pattern (e.g. 'trot', 'pace', etc.)

        Raises:
            ValueError: If an invalid gait pattern is specified
        """
        try:
            if "task" in new_commands:
                new_gait = new_commands["gait"].lower()
                if new_gait in self.gait_patterns and new_gait != self.current_gait:
                    with self._command_lock:
                        # Only set pending change if it's different from current gait
                        # and we're not already in a transition to this gait
                        if self.pending_gait_change != new_gait:
                            self.pending_gait_change = new_gait
                            self._gait_change_event.clear()  # Reset event for new change

                            # If we're already in a transition, reset it to start fresh
                            if self.in_transition:
                                self.transition_counter = 0
                                self.transition_progress = 0.0
                                self.transition_start_gait = self.current_gait
                                self.transition_end_gait = new_gait

        except Exception as e:
            self.logger.error(f"Contact command update failed: {e}")


    def _handle_command_duration_change(self, direction: str):
        """
        Handle command duration changes.

        Args:
            direction: String indicating whether to increase or decrease the duration
        """
        # Define duration change amount in seconds
        duration_change = 0.1  # 50ms change

        with self._command_lock:
            if direction == "increase":
                # Increase command duration
                self.pending_command_duration = min(1.0, self.command_duration + duration_change)
            elif direction == "decrease":
                # Decrease command duration (with a minimum value)
                self.pending_command_duration = max(0.1, self.command_duration - duration_change)
            elif direction == "default":
                self.pending_command_duration = 1.0

            # Set the pending flag
            self.command_duration_change_pending = True

            if self.logger is not None:
                self.logger.debug(f"Pending command duration change: {self.pending_command_duration:.2f} seconds")

    def _handle_reset(self):
        """Reset command duration and offset values to default."""
        with self._command_lock:
            # Reset command duration
            self.pending_command_duration = 1.0
            self.command_duration_change_pending = True

            if self.logger is not None:
                self.logger.debug("Reset command duration to default")

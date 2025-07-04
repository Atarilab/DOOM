import threading
import time
from typing import Any, Dict, TYPE_CHECKING

import numpy as np
import torch
from commands.command_manager import CommandTerm

from state_manager.obs_manager import ObsTerm

from controllers.rl_controller_base import RLControllerBase

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
        self.object_size = (0.25, 0.25, 0.25)
        
        self.repose_contact_plan_ = torch.tensor(
            [
                [False,True,True,True,True,True,True,True,True,True,True,True,],
                [False,True,True,True,True,True,True,True,True,True,True,True,],
            ]
        )
        
        self.pending_command_duration = self.command_duration  # Initialize pending command duration
        self.command_duration_change_pending = False  # Flag to indicate pending command duration change
        self.current_contact_plan = torch.ones((2, 2), dtype=torch.bool)  # Current contact pattern
        self.command_start_time = time.time()  # When the current plan started
        # Thread synchronization for gait changes
        self._gait_lock = threading.RLock()  # Reentrant lock for gait changes
        self._gait_change_event = threading.Event()  # Event to signal gait changes

        self.current_gait = "stance"  # Default gait
        self.pending_gait_change = None  # Store pending gait change
        self.in_transition = False  # Flag to indicate if we're in a transition phase
        self.transition_counter = 0  # Counter to track resample steps during transition
        self.transition_duration = configs["controller_config"].get("transition_duration", 0)  # Number of resample steps for transition (increased from 2)
        self.transition_progress = 0.0
        self.transition_start_gait = None  # Starting gait for transition
        self.transition_end_gait = None  # Ending gait for transition
        self.time_left = self.command_duration
        self.current_goal_idx = 0
        self.current_contact_plan = self.repose_contact_plan_[:, self.current_goal_idx:2]
        self.goal_completion_counter = 0
        
        self.leg_joint2motor_idx = configs["controller_config"]["leg_joint2motor_idx"]
        self.arm_waist_joint2motor_idx = configs["controller_config"]["arm_waist_joint2motor_idx"]
        
        self.actuated_joint_names = [
            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 
            'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
            'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
            'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
            'left_elbow_joint', 'right_elbow_joint',
            'left_wrist_roll_joint', 'right_wrist_roll_joint',
            'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
            'left_wrist_yaw_joint', 'right_wrist_yaw_joint'
        ]
        
        self.policy_joint_observation_names = [
            'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 
            'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 
            'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 
            'left_knee_joint', 'right_knee_joint',
            'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
            'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
            'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
            'left_ankle_roll_joint', 'right_ankle_roll_joint',
            'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
            'left_elbow_joint', 'right_elbow_joint',
            'left_wrist_roll_joint', 'right_wrist_roll_joint',
            'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
            'left_wrist_yaw_joint', 'right_wrist_yaw_joint'
        ]
        self.motor_joint_indices = self.robot.mj_model.joint_names.values()
        self.waist_Kp = 40.0
        self.waist_Kd = 8.0
        self.arm_Kp = 35.0
        self.arm_Kd = 12.0
        
        self.actuated_joint_action_indices = [self.robot.mj_model.joint_names[joint_name] for joint_name in self.actuated_joint_names]
        self.policy_joint_observation_indices = [self.robot.mj_model.joint_names[joint_name] for joint_name in self.policy_joint_observation_names]
        
        self.Kp = [self.waist_Kp, self.waist_Kp, self.waist_Kp,
                   self.arm_Kp, self.arm_Kp,
                   self.arm_Kp, self.arm_Kp,
                   self.arm_Kp, self.arm_Kp,
                   self.arm_Kp, self.arm_Kp,
                   self.arm_Kp, self.arm_Kp,
                   self.arm_Kp, self.arm_Kp,
                   self.arm_Kp, self.arm_Kp]
        self.Kd = [self.waist_Kd, self.waist_Kd, self.waist_Kd,
                   self.arm_Kd, self.arm_Kd,
                   self.arm_Kd, self.arm_Kd,
                   self.arm_Kd, self.arm_Kd,
                   self.arm_Kd, self.arm_Kd,
                   self.arm_Kd, self.arm_Kd,
                   self.arm_Kd, self.arm_Kd,
                   self.arm_Kd, self.arm_Kd]

    def register_observations(self):
        """
        Register observations for contact-conditioned locomotion.
        Includes contact pattern and timing information.
        """
        from state_manager.observations import (  
            joint_pos_limit_normalized,
            joint_vel,
            lin_vel_w,
            ang_vel_w,
            last_action,
            dummy_contact_status,
            object_size,
            contact_locations_b,
            contact_plan,
            contact_time_left,
            ee_pos_rel_b,
            root_pos_w,
            root_quat_w,
        )
        
        self.obs_manager.register(
            "joint_pos",
            ObsTerm(
                joint_pos_limit_normalized,
                params={
                    "soft_dof_limits": self.soft_dof_pos_limit,
                    "mapping": self.policy_joint_observation_indices,
                },
            ),
        )
        self.obs_manager.register(
            "joint_vel",
            ObsTerm(joint_vel,
            params={
                "mapping": self.policy_joint_observation_indices,
            }),
        )
        
        self.obs_manager.register(
            "object_pos_w",
            ObsTerm(
                root_pos_w,
                params={
                    "asset_name": "object"
                },
            ),
        )
        
        self.obs_manager.register(
            "object_quat",
            ObsTerm(
                root_quat_w,
                params={
                    "asset_name": "object"
                },
            ),
        )
        
        self.obs_manager.register("object_lin_vel_w", ObsTerm(lin_vel_w, params={"asset_name": "object"}))
        self.obs_manager.register("object_ang_vel_w", ObsTerm(ang_vel_w, params={"asset_name": "object"}))
        
        self.obs_manager.register("object_size", ObsTerm(object_size, params={"size": self.object_size}))
        self.obs_manager.register("dummy_contact_status", ObsTerm(dummy_contact_status))
        
        #########################################
        # TODO: Fill in Contact Goals here
        self.obs_manager.register(
            "contact_locations",
            ObsTerm(
                contact_locations_b,
                params={
                    "mj_model": self.robot.mj_model,
                    "future_feet_positions_w": lambda: self.future_feet_positions_w,
                    "obs_horizon": 2,
                    "current_goal_idx": lambda: self.current_goal_idx,
                },
            ),
        )
        self.obs_manager.register(
            "contact_time_left",
            ObsTerm(contact_time_left, params={"contact_time_left": lambda: self.time_left}),
        )
        self.obs_manager.register(
            "contact_plan",
            ObsTerm(
                contact_plan,
                params={"contact_plan": lambda: self.current_contact_plan},
            ),
        )
        
        self.obs_manager.register(
            "ee_pos_rel_b",
            ObsTerm(
                ee_pos_rel_b,
                params={
                    "mj_model": self.robot.mj_model,
                    "future_feet_positions_w": lambda: self.future_feet_positions_w,
                    "current_goal_idx": lambda: self.current_goal_idx,
                },
            ),
        )
        
        #########################################
        self.obs_manager.register(
            "last_action",
            ObsTerm(last_action, params={"last_action": lambda: self.raw_action}),
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
        with self._gait_lock:
            self.current_gait = "stance"
            self.current_contact_plan = self.gait_patterns["stance"]
            
            self.pending_gait_change = None
            self.in_transition = False
            self.transition_counter = 0
            self.transition_progress = 0.0
            self.transition_start_gait = None
            self.transition_end_gait = None
            self.command_duration = 1.0

            self.pending_step_size = 0.0


        base_pos_w = torch.tensor(self.robot.mj_model.get_body_position_world("base_link"), dtype=torch.float32)
        self.lateral_pos = base_pos_w[1]
        self.generate_future_feet_positions(pos=base_pos_w)

    def compute_torques(self, state, desired_goal):
        """
        Compute motor commands using the learned policy.

        :param state: Current robot state
        :param desired_goal: Desired goal state (not used in this implementation)
        :return: Motor commands dictionary
        """
        if self.robot.mj_model is not None:
            self.robot.mj_model.update(state)
            
        start_time = time.perf_counter()

        # Update the latest state for the observation processing thread
        with self._lock:
            self.latest_state = state

        # Update contact plan timing with thread safety
        with self._gait_lock:
            current_time = time.time()
            elapsed = current_time - self.command_start_time
            self.time_left = max(0, self.command_duration - elapsed)
            if elapsed >= self.command_duration:
                self.command_start_time = current_time
                self._resample_commands()

        try:
            joint_pos_targets = self.compute_joint_pos_targets()
            actuated_set = set(self.actuated_joint_action_indices)
            # First, set commands for actuated joints as before
            for joint_idx in self.actuated_joint_action_indices:
                self.cmd[f"motor_{joint_idx}"] = {
                    "q": joint_pos_targets[joint_idx],
                    "kp": self.Kp[joint_idx],
                    "dq": 0.0,
                    "kd": self.Kd[joint_idx],
                    "tau": 0.0,
                }
            # Then, for all other motor joints, set q to 0 if not already set
            for joint_idx in self.motor_joint_indices:
                if joint_idx not in actuated_set:
                    self.cmd[f"motor_{joint_idx}"] = {
                        "q": 0.0,
                        "kp": 0.0,
                        "dq": 0.0,
                        "kd": 0.0,
                        "tau": 0.0,
                    }

            # Track command preparation time
            self.cmd_preparation_time = time.perf_counter() - start_time
            
        except Exception as e:
            self.logger.error(f"Error computing torques: {e}")
            self.cmd = {
                f"motor_{joint_idx}": {
                    "q": self.default_joint_pos[joint_idx],
                    "kp": self.Kp[joint_idx],
                    "dq": 0.0,
                    "kd": self.Kd[joint_idx],
                    "tau": 0.0,
                }
                for joint_idx in self.actuated_joint_action_indices
            }
        
        return self.cmd

    def _resample_commands(self):
        """Resample the commands for the controller with thread safety."""
        try:
            with self._gait_lock:
                # Apply pending step size change if any
                
                if hasattr(self, 'step_size_change_pending') and self.step_size_change_pending:
                    self.feet_step_size = self.pending_step_size
                    self.step_size_change_pending = False
                    if self.logger is not None:
                        self.logger.debug(f"Applied step size change: {self.feet_step_size:.2f} meters")
                    # Regenerate future feet positions with new step size
                    avg_base_xy = self.future_feet_positions_w[:, self.current_goal_idx].mean(dim=0).clone()
                    self.generate_future_feet_positions(pos=avg_base_xy)

                # Apply pending command duration change if any
                if self.command_duration_change_pending:
                    self.command_duration = self.pending_command_duration
                    self.command_duration_change_pending = False
                    if self.logger is not None:
                        self.logger.debug(f"Applied command duration change: {self.command_duration:.2f} seconds")
                
                # Apply pending offset change if any
                if self.stance_width_change_pending:
                    self.current_offset[0, 1] = self.pending_stance_width_front
                    self.current_offset[1, 1] = -self.pending_stance_width_front
                    self.current_offset[2, 1] = self.pending_stance_width_rear
                    self.current_offset[3, 1] = -self.pending_stance_width_rear

                    self.stance_width_change_pending = False
                    if self.logger is not None:
                        self.logger.debug(f"Applied stance width change: {self.current_offset}")
                    # Regenerate future feet positions with new stance width
                    avg_base_xy = self.future_feet_positions_w[:, self.current_goal_idx].mean(dim=0).clone()

                    self.generate_future_feet_positions(pos=avg_base_xy)
                    
                # Handle gait transitions
                if self.pending_gait_change is not None:
                    if not self.in_transition:
                        # Start transition phase
                        self.in_transition = True
                        self.transition_counter = 0
                        self.transition_progress = 0.0
                        self.transition_start_gait = self.current_gait
                        self.transition_end_gait = self.pending_gait_change

                        # For safety, start with a stable stance if transitioning to a dynamic gait
                        if self.pending_gait_change in [gait for gait in self.gait_patterns.keys() if gait not in ["transition", "stance"]]:
                            self.current_gait = "transition"
                            self.current_contact_plan = self.gait_patterns["transition"]
                            
                        if self.transition_duration == 0:
                            self.in_transition = False
                            self.current_gait = self.pending_gait_change
                            self.current_contact_plan = self.gait_patterns[self.pending_gait_change]

                            # Call set_mode() if the pending gait is "stance"
                            if self.pending_gait_change == "stance":
                                avg_base_xy = self.future_feet_positions_w[:, self.current_goal_idx].mean(dim=0).clone()
                                self.generate_future_feet_positions(pos=avg_base_xy)

                            self.pending_gait_change = None
                            self.transition_start_gait = None
                            self.transition_end_gait = None
                            
                            # Signal that the gait change is complete
                            self._gait_change_event.set()

                            self.logger.debug(f"Transition complete, applied gait: {self.current_gait}")

                            # Signal that the gait change is complete
                    else:
                        # Increment transition counter and update progress
                        self.transition_counter += 1
                        self.transition_progress = min(1.0, self.transition_counter / self.transition_duration)

                        # Only apply the pending gait after transition duration
                        if self.transition_counter >= self.transition_duration:
                            # Transition phase complete - apply the pending gait
                            self.in_transition = False
                            self.current_gait = self.pending_gait_change
                            self.current_contact_plan = self.gait_patterns[self.pending_gait_change]

                            # Call set_mode() if the pending gait is "stance"
                            if self.pending_gait_change == "stance":
                                avg_base_xy = self.future_feet_positions_w[:, self.current_goal_idx].mean(dim=0).clone()
                                self.generate_future_feet_positions(pos=avg_base_xy)

                            self.pending_gait_change = None
                            self.transition_start_gait = None
                            self.transition_end_gait = None

                            # Signal that the gait change is complete
                            self._gait_change_event.set()

                            self.logger.debug(f"Transition complete, applied gait: {self.current_gait}")


                # Normal gait progression (only if not in transition phase)
                if not self.in_transition and self.current_gait not in ["stance", "transition"]:
                    self.goal_completion_counter += 1
                
                    self.current_goal_idx += (
                        1 if self.goal_completion_counter % 2 != 0 and self.goal_completion_counter > 0 else 0
                    )
                    self.current_contact_plan = torch.stack(
                        [
                            self.current_contact_plan[1],
                            self.current_contact_plan[0],
                        ]
                    )
                
                # Safety mechanism, if the goal index exceeds the horizon length, reset the goal index and replan feet sequences
                if self.current_goal_idx >= self.horizon_length:
                    base_pos_w = torch.tensor(self.robot.mj_model.get_body_position_world("base_link"), dtype=torch.float32)
                    self.generate_future_feet_positions(pos=base_pos_w)
                        
            
            
                        
        except Exception as e:
            if hasattr(self, "command_manager") and self.command_manager and hasattr(self.command_manager, "logger"):
                self.logger.error(f"Error resampling commands: {e}")
            else:
                print(f"Error resampling commands: {e}")

    def generate_future_feet_positions(self, pos=torch.zeros(3)):
        """
        Generates future feet positions for the controller.
        Uses the heading_command relative to the robot's base yaw to determine the direction of movement.
        """
        self.goal_completion_counter = 0
        self.current_goal_idx = 0

        # Use current offset values
        offset = self.current_offset.clone()
        
        # Use the heading command relative to robot's yaw
        with self.heading_command_lock:
            # Convert heading command to tensor for calculations
            heading_command_tensor = torch.tensor(self.heading_command, dtype=torch.float32)
            # Calculate target yaw by adding the heading command to the robot's yaw
            target_yaw = heading_command_tensor
            
        # Create rotation matrix for the target yaw
        cos_yaw = torch.cos(target_yaw)
        sin_yaw = torch.sin(target_yaw)
        rotation_matrix = torch.tensor([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]], dtype=torch.float32)

        # Rotate the offset positions
        rotated_offset = torch.matmul(rotation_matrix, offset.T).T

        # Check if pos is a tensor of zeros (default case)
        if torch.all(pos == 0):
            # Get the robot's current position
            robot_pos = torch.tensor(self.robot.mj_model.get_body_position_world("base_link"), dtype=torch.float32)
            # Use only the x-dimension of the robot's position
            pos = torch.tensor([robot_pos[0],self.lateral_pos, 0.0], dtype=torch.float32)

        self.future_feet_positions_w[:] = (pos + rotated_offset).unsqueeze(1)

        stride_offsets = torch.arange(self.horizon_length, dtype=torch.float32).unsqueeze(0) * torch.tensor(
            self.feet_step_size, dtype=torch.float32
        ).unsqueeze(-1)
        
        # r = torch.zeros(4, self.horizon_length)
        # x_rand = r.uniform_(-0.06, 0.06).clone()
        # y_rand = r.uniform_(-0.06, 0.06).clone()
        
        

        # Apply stride in the direction of the target heading
        direction_x = cos_yaw
        direction_y = sin_yaw
        self.future_feet_positions_w[:, :, 0] += stride_offsets.squeeze(-1) * direction_x
        self.future_feet_positions_w[:, :, 1] += stride_offsets.squeeze(-1) * direction_y
        # self.future_feet_positions_w[:, :, 2] = -0.02 if self.interface == "real" else 0.08
        self.future_feet_positions_w[:, :, 2] = 0.03
        
        # if self.current_gait in ["trot", "pace"]:
        #     self.future_feet_positions_w[[1, 2], :, 0] -= (self.feet_step_size / 2) * direction_x
        #     self.future_feet_positions_w[[1, 2], :, 1] -= (self.feet_step_size / 2) * direction_y
            
        # if self.current_gait in ["pace"]:
        #     self.future_feet_positions_w[[0, 3], :, 0] += 0.12 * direction_x
        #     self.future_feet_positions_w[[0, 3], :, 1] += 0.12 * direction_y
        self.future_feet_positions_init_frame = self.robot.mj_model.transform_world_to_init_frame(
            self.future_feet_positions_w.numpy()
        )
       
        
        if self.visualize["future_feet_positions"]:
            self.pub_future_feet_positions()
    
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
                    with self._gait_lock:
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


    def _handle_step_size_change(self, direction: str):
        """Handle forwardstep size changes."""
        # Increase step size by 0.01 meters when up button is pressed
        if direction == "up":
            new_step_size = self.feet_step_size + 0.1
        elif direction == "down":
            new_step_size = self.feet_step_size - 0.1
        # Limit maximum step size to 0.2 meters for safety
        new_step_size = max(-0.3, min(new_step_size, 0.3))
        
        # Set the pending step size change instead of applying it immediately
        with self._gait_lock:
            self.pending_step_size = new_step_size
            self.step_size_change_pending = True

    def _handle_command_duration_change(self, direction: str):
        """
        Handle command duration changes.
        
        Args:
            direction: String indicating whether to increase or decrease the duration
        """
        # Define duration change amount in seconds
        duration_change = 0.1  # 50ms change
        
        with self._gait_lock:
            if direction == "increase":
                # Increase command duration
                self.pending_command_duration = min(1.0,self.command_duration + duration_change)
            elif direction == "decrease":
                # Decrease command duration (with a minimum value)
                self.pending_command_duration = max(0.1, self.command_duration - duration_change)
            elif direction == "default":
                self.pending_command_duration = 0.35
                
            # Set the pending flag
            self.command_duration_change_pending = True
            
            if self.logger is not None:
                self.logger.debug(f"Pending command duration change: {self.pending_command_duration:.2f} seconds")

    def _handle_reset(self):
        """Reset command duration and offset values to default."""
        with self._gait_lock:
            # Reset command duration
            self.pending_command_duration = 0.35
            self.command_duration_change_pending = True

            if self.logger is not None:
                self.logger.debug("Reset command duration to default")

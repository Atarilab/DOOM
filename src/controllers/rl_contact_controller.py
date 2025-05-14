import threading
import time
from typing import Any, Dict, TYPE_CHECKING

import numpy as np
import torch
from commands.command_manager import CommandTerm

from state_manager.obs_manager import ObsTerm
from std_msgs.msg import ColorRGBA, Float32MultiArray, MultiArrayDimension
from visualization_msgs.msg import Marker, MarkerArray

from controllers.rl_controller import BaseRLLocomotionController

if TYPE_CHECKING:
    from robots.robot_base import RobotBase

class RLQuadrupedLocomotionContactController(BaseRLLocomotionController):
    """
    Contact-conditioned RL Locomotion Controller
    Uses contact-explicit reinforcement learning policy
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any]):
        self.init_completed = False
        super().__init__(robot=robot, configs=configs)

        # Initialize publshers for visualization
        self._init_publishers(configs)

        # Contact command parameters
        self.command_duration = 0.35  # Duration of each contact plan in seconds
        self.pending_command_duration = self.command_duration  # Initialize pending command duration
        self.command_duration_change_pending = False  # Flag to indicate pending command duration change
        self.current_contact_plan = torch.ones((2, 4), dtype=torch.bool)  # Current contact pattern
        self.command_start_time = time.time()  # When the current plan started
        # Thread synchronization for gait changes
        self._gait_lock = threading.RLock()  # Reentrant lock for gait changes
        self._gait_change_event = threading.Event()  # Event to signal gait changes

        # Default offset values ( stride lengths and stance widths would be add to these values )
        self.default_offset = torch.tensor(
            [(0.1934, 0.1465, 0.0),
            (0.1934, -0.1465, 0.0),
            (-0.1934, 0.1465, 0.0),
            (-0.1934, -0.1465, 0.0),],
            dtype=torch.float32,
        )
        self.current_offset = self.default_offset.clone()
        self.pending_stance_width_front = self.current_offset[0, 1]
        self.pending_stance_width_rear = self.current_offset[0, 1]
        self.stance_width_change_pending = False

        # Horizon planning
        self.future_feet_positions_init_frame = None
        self.horizon_length = configs["controller_config"]["horizon_length"]
        self.feet_step_size = configs["controller_config"]["feet_step_size"]
        self.lateral_pos = 0.0
        self.future_feet_positions_init_frame = None
        self.future_feet_positions_w = torch.zeros(4, self.horizon_length, 3)
        self.future_feet_positions_b = torch.zeros(4, self.horizon_length, 3)
        self.desired_ee_position_w = np.zeros((4, 3))
        
        # Heading command for feet positions
        self.heading_command = 0.0  # Default heading (in radians)
        self.heading_command_lock = threading.RLock()  # Lock for heading command updates
        self.heading_change_pending = False  # Flag to indicate pending heading change
        self.lateral_pos_change_pending = False  # Flag to indicate pending lateral pos change

        # Define gait patterns (FL, FR, RL, RR)
        self.gait_patterns = {
            "transition": torch.tensor(
                [
                    [True, True, True, True],  # Phase 1: All legs in contact
                    [True, True, True, True],  # Phase 2: All legs in contact
                ]
            ),
            "stance": torch.tensor(
                [
                    [True, True, True, True],  # Phase 1: All legs in contact
                    [True, True, True, True],  # Phase 2: All legs in contact
                ]
            ),
            "trot": torch.tensor(
                [
                    [False, True, True, False],  # Phase 2: FR and RL in contact
                    [True, False, False, True],  # Phase 1: FL and RR in contact
                ]
            ),
            "trot-jump1": torch.tensor(
                [
                    [False, True, True, False],  # Phase 2: FR and RL in contact
                    [False, False, False, False],  # Phase 1: FL and RR in contact
                ]
            ),
            "trot-jump2": torch.tensor(
                [
                    [True, False, False, True],  # Phase 2: FR and RL in contact
                    [False, False, False, False],  # Phase 1: FL and RR in contact
                ]
            ),
            
            "pace": torch.tensor(
                [
                    [True, False, True, False],  # Phase 1: FL and RL in contact
                    [False, True, False, True],  # Phase 2: FR and RR in contact
                ]
            ),
            "pace-jump1": torch.tensor(
                [
                    [False, False, False, False],  # Phase 2: All legs in flight
                    [True, False, True, False],  # Phase 1: FL and RL in contact
                ]
            ),
            "pace-jump2": torch.tensor(
                [
                    [False, False, False, False],  # Phase 2: All legs in flight
                    [False, True, False, True],  # Phase 1: FR and RR in contact
                ]
            ),
            "bound": torch.tensor(
                [
                    [True, True, False, False],  # Phase 1: Front legs in contact
                    [False, False, True, True],  # Phase 2: Back legs in contact
                ]
            ),
            "random1": torch.tensor(
                [
                    [True, False, False, False],
                    [False, True, True, True],
                ]
            ),
            "random2": torch.tensor(
                [
                    [False, True, False, False],
                    [True, False, True, True],
                ]
            ),
            "random3": torch.tensor(
                [
                    [False, False, True, False],
                    [True, True, False, True],
                ]
            ),
            "random4": torch.tensor(
                [
                    [False, False, False, True],
                    [True, True, True, False],
                ]
            ),
            "jump": torch.tensor(
                [
                    [False, False, False, False],  # Phase 2: All legs in flight
                    [True, True, True, True],  # Phase 1: All legs in contact
                ]
            ),
            "crawl-anticlockwise": torch.tensor(
                [
                    [True, True, True, False],  # Phase 1: FL, FR, RL in contact, RR moving
                    [True, True, False, True],  # Phase 2: FL, FR, RR in contact, RL moving
                    [True, False, True, True],  # Phase 3: FL, RL, RR in contact, FR moving
                    [False, True, True, True],  # Phase 4: FR, RL, RR in contact, FL moving
                ]
            ),
            "crawl-anticlockwise2": torch.tensor(
                [
                    [True, True, True, False],  # Phase 1: FL, FR, RL in contact, RR moving
                    [True, True, False, True],  # Phase 2: FL, FR, RR in contact, RL moving
                    [False, True, True, True],  # Phase 4: FR, RL, RR in contact, FL moving
                    [True, False, True, True],  # Phase 3: FL, RL, RR in contact, FR moving
                ]
            ),
            "crawl-clockwise": torch.tensor(
                [                    
                    [True, True, True, False],  # Phase 2: FL, FR, RL in contact, RR moving
                    [False, True, True, True],  # Phase 1: FR, RL, RR in contact, FL moving
                    [True, False, True, True],  # Phase 4: FL, RL, RR in contact, FR moving
                    [True, True, False, True],  # Phase 3: FL, FR, RR in contact, RL moving
                ]
            ),
            "trot-fly": torch.tensor(
                [
                    [False, True, True, False],  # Phase 2: FR and RL in contact
                    [False, False, False, False],  # Phase 1: FL and RR in contact
                    [True, False, False, True],  # Phase 4: FR and RL in contact
                    [False, False, False, False],  # Phase 3: FL and RR in contact
                ]
            ),
            

        }
        self.current_gait = "stance"  # Default gait
        self.pending_gait_change = None  # Store pending gait change
        self.in_transition = False  # Flag to indicate if we're in a transition phase
        self.transition_counter = 0  # Counter to track resample steps during transition
        self.transition_duration = configs["controller_config"]["transition_duration"]  # Number of resample steps for transition (increased from 2)
        self.transition_progress = 0.0
        self.transition_start_gait = None  # Starting gait for transition
        self.transition_end_gait = None  # Ending gait for transition
        self.time_left = self.command_duration
        self.current_contact_plan = self.gait_patterns[self.current_gait]
        self.current_goal_idx = 0
        self.goal_completion_counter = 0

        self.init_completed = True
        
    def register_observations(self):
        """
        Register observations for contact-conditioned locomotion.
        Includes contact pattern and timing information.
        """
        from state_manager.observations import (  
            joint_pos_rel,
            joint_vel,
            lin_vel_b,
            ang_vel_b,
            last_action,
            projected_gravity_b,
            contact_locations_b,
            contact_plan,
            contact_time_left,
            ee_pos_rel_b,
        )
        self.obs_manager.register("lin_vel_b", ObsTerm(lin_vel_b))
        self.obs_manager.register("ang_vel_b", ObsTerm(ang_vel_b))
        self.obs_manager.register("projected_gravity", ObsTerm(projected_gravity_b))
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
        self.obs_manager.register(
            "last_action",
            ObsTerm(last_action, params={"last_action": lambda: self.raw_action}),
        )
        
    def register_commands(self):
        """Register contact command parameters."""
        self.command_manager.register(
            "gait",
            CommandTerm(
                type=str,
                name="gait",
                description="Gait pattern (trot, pace, bound, jump)",
                min_value=0,  # Not used for string type
                max_value=1,  # Not used for string type
                default_value="trot",
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
            self.feet_step_size = 0.0
            self.command_duration = 0.35
            self.heading_command = 0.0
            self.pending_heading = 0.0
            self.pending_step_size = 0.0
            self.pending_lateral_pos = 0.0

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
                # self._update_desired_ee_positions()
                self._update_action_scale()
                
                if self.visualize["feet_error"]:
                    self.pub_feet_error()
                    self.pub_feet_error_norm()

        if self.visualize["current_contact_locations"]:
            self.pub_current_contact_locations()

        if self.visualize["foot_forces"]:
            self.pub_foot_forces(state["foot_forces"])
            self.pub_contact_status(state["foot_forces"])
        if self.visualize["contact_plan"]:
            self.pub_contact_plan()

        # self.command_manager.logger.debug(f"Time left: {self.time_left:.2f} seconds")
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
            self.command_manager.logger.error(f"Error computing torques: {e}")
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
    
    def _update_action_scale(self):
        """Update the action scale based on the current gait."""
        
        # self.action_scale = 0.4 if self.current_gait in ["jump", "bound", "trot-jump1", "trot-jump2", "trot-fly", "pace-jump1", "pace-jump2", "crawl-anticlockwise", "crawl-clockwise", "random1", "random2", "random3", "random4"] else 0.35
        self.action_scale = 0.4

    def _resample_commands(self):
        """Resample the commands for the controller with thread safety."""
        try:
            with self._gait_lock:
                # Apply pending step size change if any
                
                if hasattr(self, 'step_size_change_pending') and self.step_size_change_pending:
                    self.feet_step_size = self.pending_step_size
                    self.step_size_change_pending = False
                    if self.command_manager and self.command_manager.logger:
                        self.command_manager.logger.debug(f"Applied step size change: {self.feet_step_size:.2f} meters")
                    # Regenerate future feet positions with new step size
                    avg_base_xy = self.future_feet_positions_w[:, self.current_goal_idx].mean(dim=0).clone()
                    self.generate_future_feet_positions(pos=avg_base_xy)
                    
                
                # Apply pending heading change if any
                if self.heading_change_pending:
                    self.heading_command = self.pending_heading
                    self.heading_change_pending = False
                    if self.command_manager and self.command_manager.logger:
                        self.command_manager.logger.debug(f"Applied heading change: {self.heading_command:.2f} radians")
                    # Regenerate future feet positions with new heading
                    self.generate_future_feet_positions()

                
                if self.lateral_pos_change_pending:
                    self.lateral_pos = self.pending_lateral_pos
                    self.lateral_pos_change_pending = False
                    if self.command_manager and self.command_manager.logger:
                        self.command_manager.logger.debug(f"Applied lateral pos change: {self.lateral_pos:.2f} meters")
                    # Regenerate future feet positions with new heading
                    avg_base_xy = self.future_feet_positions_w[:, self.current_goal_idx].mean(dim=0).clone()
                    avg_base_xy[1] = self.lateral_pos
                    self.generate_future_feet_positions(pos=avg_base_xy)
                
                # Apply pending command duration change if any
                if self.command_duration_change_pending:
                    self.command_duration = self.pending_command_duration
                    self.command_duration_change_pending = False
                    if self.command_manager and self.command_manager.logger:
                        self.command_manager.logger.debug(f"Applied command duration change: {self.command_duration:.2f} seconds")
                
                # Apply pending offset change if any
                if self.stance_width_change_pending:
                    self.current_offset[0, 1] = self.pending_stance_width_front
                    self.current_offset[1, 1] = -self.pending_stance_width_front
                    self.current_offset[2, 1] = self.pending_stance_width_rear
                    self.current_offset[3, 1] = -self.pending_stance_width_rear

                    self.stance_width_change_pending = False
                    if self.command_manager and self.command_manager.logger:
                        self.command_manager.logger.debug(f"Applied stance width change: {self.current_offset}")
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

                            self.command_manager.logger.debug(f"Transition complete, applied gait: {self.current_gait}")

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

                            self.command_manager.logger.debug(f"Transition complete, applied gait: {self.current_gait}")


                # Normal gait progression (only if not in transition phase)
                if not self.in_transition and self.current_gait not in ["stance", "transition"]:
                    self.goal_completion_counter += 1
                    
                    # Handle different gait phases
                    if "crawl" in self.current_gait:
                        # For crawl gait or trot fly, cycle through 4 phases
                        self.current_goal_idx += 1 if (self.goal_completion_counter) % 4 == 0 else 0
                        self.current_contact_plan = torch.stack([
                            self.gait_patterns[self.current_gait][(self.goal_completion_counter) % 4],
                            self.gait_patterns[self.current_gait][(self.goal_completion_counter + 1) % 4]    
                        ])
                    # elif self.current_gait == "jump":
                    #     self.current_goal_idx += 1 if (self.goal_completion_counter) % 4 == 0 else 0
                    #     self.current_contact_plan = torch.stack(
                    #         [
                    #             self.current_contact_plan[1],
                    #             self.current_contact_plan[0],
                    #         ]
                    #     )
                        
                        
                    else:
                        # For other gaits (2 phases), alternate between phases
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
                        
            # if self.current_gait == "jump" and self.current_contact_plan[0].sum() == 4:
            #     self.command_duration = 0.35
            # else:
            #     self.command_duration = 0.35
            
            
                        
        except Exception as e:
            if hasattr(self, "command_manager") and self.command_manager and hasattr(self.command_manager, "logger"):
                self.command_manager.logger.error(f"Error resampling commands: {e}")
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
            "B": lambda: self.change_commands({"gait": "pace"}),
            "X": lambda: self.change_commands({"gait": "bound"}),
            "Y": lambda: self.change_commands({"gait": "jump"}),
            # Crawl Gaits
            "L2-Y": lambda: self.change_commands({"gait": "crawl-anticlockwise"}),
            "L2-A": lambda: self.change_commands({"gait": "crawl-clockwise"}),
            
            # Pace-Jump Gaits
            "R1-X": lambda: self.change_commands({"gait": "pace-jump1"}),
            "R1-B": lambda: self.change_commands({"gait": "pace-jump2"}),
            # Trot-Jump Gaits
            "R1-Y": lambda: self.change_commands({"gait": "trot-jump1"}),
            "R1-A": lambda: self.change_commands({"gait": "trot-jump2"}),
            # "L1-X": lambda: self.change_commands({"gait": "trot-fly"}),
            # OOD Gaits
            "R2-Y": lambda: self.change_commands({"gait": "random1"}),
            "R2-X": lambda: self.change_commands({"gait": "random2"}),
            "R2-A": lambda: self.change_commands({"gait": "random3"}),
            "R2-B": lambda: self.change_commands({"gait": "random4"}),
            # Step Size
            "up": lambda: self._handle_step_size_change("up"),
            "down": lambda: self._handle_step_size_change("down"),
            # Lateral Position 
            "left": lambda: self._handle_lateral_pos_change("left"),
            "right": lambda: self._handle_lateral_pos_change("right"),
            # Command Duration
            "L1-up": lambda: self._handle_command_duration_change("increase"),
            "L1-down": lambda: self._handle_command_duration_change("decrease"),
            "L1-R1": lambda: self._handle_reset(),
            # Offset Adjustment
            "L1-left": lambda: self._handle_stance_width_change("front", "decrease"),
            "L1-right": lambda: self._handle_stance_width_change("front", "increase"),
            "R1-left": lambda: self._handle_stance_width_change("rear", "decrease"),
            "R1-right": lambda: self._handle_stance_width_change("rear", "increase"),
        }
        
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
            if "gait" in new_commands:
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
            self.command_manager.logger.error(f"Contact command update failed: {e}")
        
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

    def _handle_heading_change(self, direction: str):
        """
        Handle heading changes using directional buttons.
        
        Args:
            direction: String indicating the direction to turn ("left" or "right")
        """
        # Define heading change amount in radians
        heading_change = 0.05  # About 11.5 degrees
        
        with self._gait_lock:
            if direction == "left":
                # Turn left (positive heading)
                self.pending_heading = self.heading_command + heading_change
            elif direction == "right":
                # Turn right (negative heading)
                self.pending_heading = self.heading_command - heading_change
                
            # # Keep heading within -pi to pi range
            # self.pending_heading = (self.pending_heading + np.pi) % (2 * np.pi) - np.pi
            
            # Set the pending flag
            self.heading_change_pending = True

    def _handle_lateral_pos_change(self, direction: str):
        """Handle lateral step size changes."""
        with self._gait_lock:
            if direction == "left":
                self.pending_lateral_pos = self.lateral_pos + 0.015
            elif direction == "right":
                self.pending_lateral_pos = self.lateral_pos - 0.015
            self.lateral_pos_change_pending = True

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
            
            if self.command_manager and self.command_manager.logger:
                self.command_manager.logger.debug(f"Pending command duration change: {self.pending_command_duration:.2f} seconds")

    def _handle_stance_width_change(self, side: str, direction: str):
        """Handle offset changes."""
        # Define offset change amount in meters
        stance_width_change = 0.4  # 2cm change
        
        with self._gait_lock:
            if direction == "increase":
                # Increase offset values
                if side == "front":
                    self.pending_stance_width_front = self.current_offset[0, 1] + stance_width_change  # Modify y-coordinate (stance width)
                elif side == "rear":
                    self.pending_stance_width_rear = self.current_offset[2, 1] + stance_width_change  # Modify y-coordinate (stance width)
            elif direction == "decrease":
                # Decrease offset values
                if side == "front":
                    self.pending_stance_width_front = self.current_offset[0, 1] - stance_width_change  # Modify y-coordinate (stance width)
                elif side == "rear":
                    self.pending_stance_width_rear = self.current_offset[2, 1] - stance_width_change  # Modify y-coordinate (stance width)
                
            # Set bounds for stance width (0.1m to 0.3m)
            self.pending_stance_width_front = max(0.02, min(0.3, self.pending_stance_width_front))
            self.pending_stance_width_rear = max(0.02, min(0.3, self.pending_stance_width_rear))
                
            # Set the pending flag
            self.stance_width_change_pending = True
            
            if self.command_manager and self.command_manager.logger:
                self.command_manager.logger.debug(f"Pending stance width change: {self.pending_stance_width_front:.3f} meters (front) and {self.pending_stance_width_rear:.3f} meters (rear)")

    def _handle_reset(self):
        """Reset command duration and offset values to default."""
        with self._gait_lock:
            # Reset command duration
            self.pending_command_duration = 0.35
            self.command_duration_change_pending = True
            
            # Reset offset values
            self.pending_stance_width_front = self.default_offset[0, 1]
            self.pending_stance_width_rear = self.default_offset[0, 1]
            self.stance_width_change_pending = True
            
            if self.command_manager and self.command_manager.logger:
                self.command_manager.logger.debug("Reset command duration and stance width values to default")

    """
    Helper functions to initialize publishers for visualization.
    """

    def _init_publishers(self, configs):
        """
        Initialize publishers for visualization.
        """
        # Initialize visualization flags from configs
        self.visualize = {key: configs["controller_config"]["visualize"][key] for key in configs["controller_config"]["visualize"]}
        
        if self.visualize["future_feet_positions"]:
            self.feet_trajectory_pub = self.create_publisher(MarkerArray, "feet_trajectories", 10)
        if self.visualize["current_contact_locations"]:
            self.contact_locations_pub = self.create_publisher(MarkerArray, "contact_locations", 10)
        if self.visualize["feet_error"]:
            self.feet_error_pub = self.create_publisher(MarkerArray, "feet_error", 10)
            self.feet_error_norm_pub = self.create_publisher(Float32MultiArray, "feet_error_norm", 10)
        if self.visualize["contact_plan"]:
            self.contact_plan_pub = self.create_publisher(Float32MultiArray, "contact_plan", 10)
        if self.visualize["foot_forces"]:
            self.foot_forces_pub = self.create_publisher(Float32MultiArray, "foot_forces", 10)
        if self.visualize["contact_status"]:
            self.contact_status_pub = self.create_publisher(Float32MultiArray, "contact_status", 10)
            
    def pub_feet_error(self):
        """
        Visualizes the feet error in the world frame.
        """
        try:
            # Publish feet locations
            feet_errors_msg = MarkerArray()
            feet_names = ["FL", "FR", "RL", "RR"]
            colors = [
                ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # Red for FL
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),  # Green for FR
                ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # Blue for RL
                ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),  # Yellow for RR
            ]

            # Get feet positions from the state manager
            feet_errors = self.robot.mj_model.get_feet_positions_world() - self.future_feet_positions_w[:, self.current_goal_idx].numpy()
            if feet_errors is not None:
                for i, (name, color) in enumerate(zip(feet_names, colors)):
                    marker = Marker()
                    marker.header.frame_id = "world"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = "feet_locations"
                    marker.id = i
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD

                    # Set the scale of the sphere
                    marker.scale.x = 0.05  # radius
                    marker.scale.y = 0.05
                    marker.scale.z = 0.05

                    # Set color
                    marker.color = color

                    # Set position
                    marker.pose.position.x = float(feet_errors[i][0])
                    marker.pose.position.y = float(feet_errors[i][1])
                    marker.pose.position.z = float(feet_errors[i][2])

                    # Set orientation (identity quaternion)
                    marker.pose.orientation.w = 1.0
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0

                    feet_errors_msg.markers.append(marker)

            # Publish the marker array
            self.feet_error_pub.publish(feet_errors_msg)
        except Exception as e:
            self.command_manager.logger.error(f"Failed to publish feet error: {e}")

    def pub_future_feet_positions(self):
        """
        Visualizes the future feet positions in the world frame.
        """
        try:
            marker_array = MarkerArray()

            # Colors for each foot trajectory
            colors = [
                ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # Red for FL
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),  # Green for FR
                ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # Blue for RL
                ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),  # Yellow for RR
            ]

            foot_names = ["FL", "FR", "RL", "RR"]

            # Transpose the tensor to match the expected shape (num_steps, 4, 3)
            positions_for_viz = self.future_feet_positions_w.permute(1, 0, 2).numpy()

            for foot_idx in range(len(foot_names)):
                # Create a marker for each future position of each foot
                for step in range(positions_for_viz.shape[0]):
                    marker = Marker()
                    marker.header.frame_id = "world"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = f"foot_trajectory_{foot_names[foot_idx]}"
                    marker.id = foot_idx * positions_for_viz.shape[0] + step
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD

                    # Set the scale of the sphere
                    marker.scale.x = 0.02  # radius
                    marker.scale.y = 0.02
                    marker.scale.z = 0.02

                    # Set the color
                    marker.color = colors[foot_idx]

                    # Set position
                    marker.pose.position.x = float(positions_for_viz[step, foot_idx, 0])
                    marker.pose.position.y = float(positions_for_viz[step, foot_idx, 1])
                    marker.pose.position.z = float(positions_for_viz[step, foot_idx, 2])

                    # Set orientation (identity quaternion)
                    marker.pose.orientation.w = 1.0
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0

                    marker_array.markers.append(marker)

            # Publish the marker array
            self.feet_trajectory_pub.publish(marker_array)
            if self.command_manager and self.command_manager.logger:
                self.command_manager.logger.debug("Published future feet positions marker array")

        except Exception as e:
            if self.command_manager and self.command_manager.logger:
                self.command_manager.logger.error(f"Failed to publish future feet positions: {e}")

    def pub_current_contact_locations(self):
        """
        Visualizes the current contact locations in the world frame.
        Colors the markers based on the current contact plan:
        - Green: for feet that are not in contact (contact_plan is False)
        - Black: for feet that are in contact (contact_plan is True)
        """
        try:
            marker_array = MarkerArray()

            # Get current desired positions for each foot
            current_positions = self.future_feet_positions_w[:, self.current_goal_idx]

            # Create markers for each foot
            for foot_idx in range(4):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "current_contact_locations"
                marker.id = foot_idx
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                # Set the scale of the sphere
                marker.scale.x = 0.05  # radius
                marker.scale.y = 0.05
                marker.scale.z = 0.05

                # Set color based on contact plan
                if self.current_contact_plan[0, foot_idx]:
                    # Black for feet in contact
                    marker.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0)
                else:
                    # Green for feet not in contact
                    marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)

                # Set position using desired positions
                marker.pose.position.x = float(current_positions[foot_idx, 0])
                marker.pose.position.y = float(current_positions[foot_idx, 1])
                marker.pose.position.z = float(current_positions[foot_idx, 2])

                # Set orientation (identity quaternion)
                marker.pose.orientation.w = 1.0
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0

                marker_array.markers.append(marker)

            # Publish the marker array
            self.contact_locations_pub.publish(marker_array)

        except Exception as e:
            if self.command_manager and self.command_manager.logger:
                self.command_manager.logger.error(f"Failed to publish current contact locations: {e}")

    def pub_feet_error_norm(self):
        """
        Publishes the error norm (magnitude) for each foot.
        """
        try:
            # Calculate error norms directly
            feet_errors = np.linalg.norm(self.robot.mj_model.get_feet_positions_world() - self.future_feet_positions_w[:, self.current_goal_idx].numpy(), axis=-1)
            
            # Create Float32MultiArray message
            msg = Float32MultiArray()
            msg.layout.dim = [MultiArrayDimension()]
            msg.layout.dim[0].label = "feet"
            msg.layout.dim[0].size = 4
            msg.layout.dim[0].stride = 4
            msg.data = [float(x) for x in feet_errors]

            # Publish the message
            self.feet_error_norm_pub.publish(msg)
        except Exception as e:
            self.command_manager.logger.error(f"Failed to publish feet error norm: {e}")

    def pub_contact_plan(self):
        """
        Publishes the boolean contact plan values.
        The contact plan is a boolean tensor indicating which feet are in contact.
        """
        try:
            # Create Float32MultiArray message
            msg = Float32MultiArray()
            msg.layout.dim = [MultiArrayDimension()]
            msg.layout.dim[0].label = "feet"
            msg.layout.dim[0].size = 4
            msg.layout.dim[0].stride = 4
            
            # Convert boolean contact plan to float values (0.0 for False, 1.0 for True)
            contact_plan_values = [float(x) for x in self.current_contact_plan[0]]
            msg.data = contact_plan_values

            # Publish the message
            self.contact_plan_pub.publish(msg)
            
        except Exception as e:
            if self.command_manager and self.command_manager.logger:
                self.command_manager.logger.error(f"Failed to publish contact plan: {e}")
        
    def pub_contact_status(self, foot_forces):
        """
        Publishes the foot forces in the world frame.
        """
        try:
            # Calculate foot forces directly
            msg = Float32MultiArray()
            msg.layout.dim = [MultiArrayDimension()]
            msg.layout.dim[0].label = "feet"
            msg.layout.dim[0].size = 4
            msg.layout.dim[0].stride = 4
            mapped_foot_forces = [foot_forces[1], foot_forces[0], foot_forces[3], foot_forces[2]]
            msg.data = [float(x > 30) for x in mapped_foot_forces]
            # self.command_manager.logger.debug(f"Publishing foot forces: {foot_forces}")

            # Publish the message
            self.contact_status_pub.publish(msg)
        except Exception as e:
            self.command_manager.logger.error(f"Failed to publish contact status: {e}")
            
    def pub_foot_forces(self, foot_forces):
        """
        Publishes the foot forces in the world frame.
        """
        try:
            # Calculate foot forces directly
            msg = Float32MultiArray()
            msg.layout.dim = [MultiArrayDimension()]
            msg.layout.dim[0].label = "feet"
            msg.layout.dim[0].size = 4
            msg.layout.dim[0].stride = 4
            mapped_foot_forces = [foot_forces[1], foot_forces[0], foot_forces[3], foot_forces[2]]
            msg.data = [float(x) for x in mapped_foot_forces]
            # self.command_manager.logger.debug(f"Publishing foot forces: {foot_forces}")

            # Publish the message
            self.foot_forces_pub.publish(msg)
        except Exception as e:
            self.command_manager.logger.error(f"Failed to publish foot forces: {e}")

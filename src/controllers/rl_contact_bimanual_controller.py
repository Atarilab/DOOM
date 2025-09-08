import threading
import time
from typing import TYPE_CHECKING, Any, Dict

import torch

from controllers.rl_controller_base import RLControllerBase
from state_manager.obs_manager import ObsTerm
from utils.math import combine_frame_transforms, quat_error_magnitude
from utils.frequency_tracker import FrequencyTracker, MultiFrequencyTracker

# ROS2 imports for visualization
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

if TYPE_CHECKING:
    from robots.robot_base import RobotBase

class RLHumanoidBimanualContactController(RLControllerBase):
    """
    Contact-conditioned RL Bimanual Controller
    Uses contact-explicit reinforcement learning policy
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any]):

        super().__init__(robot=robot, configs=configs)
        
        # Initialize publishers for visualization
        self._init_publishers(configs)

        self.policy_joint_names = [
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
        
        self.policy_joint_indices = [self.robot.actuated_joint_names.index(joint_name) for joint_name in self.policy_joint_names]
        self.non_policy_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ]
        self.non_policy_joint_indices = [self.robot.actuated_joint_names.index(joint_name) for joint_name in self.non_policy_joint_names if joint_name in self.robot.actuated_joint_names + self.robot.non_actuated_joint_names]
        self.non_policy_joint_Kps = [50, 50, 50, 75, 20, 20, 50, 50, 50, 75, 20, 20, 150, 150, 150, 150]
        self.non_policy_joint_Kds = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 3, 3, 3]
        self.non_policy_default_angles = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0]
        # Contact command parameters
        self.command_duration = 1.3  # Duration of each contact plan in seconds
        self.object_size = torch.tensor([0.14, 0.105, 0.14], dtype=torch.float32, device=self.device) * 2.0 

        self.repose_contact_plan_ = torch.tensor(
            [
                [False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                [False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
            ], 
            dtype=torch.bool, device=self.device
        )
        
        self.reorientation_contact_plan_ = torch.tensor(
            [
                [False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True],
                [False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True],
            ], 
            dtype=torch.bool, device=self.device
        )
        
        # self.repose_contact_plan_ = torch.tensor(
        #     [
        #         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #     ], 
        #     dtype=torch.bool, device=self.device
        # )

        self.command_start_time = time.time()  # When the current plan started
        self._command_lock = threading.RLock()  # Reentrant lock for command changes

        self.time_left = self.command_duration
        self.current_goal_idx = 0
        # self.current_contact_plan = self.repose_contact_plan_[:, self.current_goal_idx : 2]
        self.current_contact_plan = self.reorientation_contact_plan_[:, self.current_goal_idx : 2]
        self.goal_completion_counter = 0
        
        # Task change management
        self.pending_task_change = None  # Store pending task change
        self.task_change_pending = False  # Flag to indicate pending task change
        
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
        
        self.reorientation_contact_pos_o = torch.tensor([
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],  # y-axis top/bottom
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],  # y-axis top/bottom
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],  # y-axis top/bottom
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],  # x-axis sides
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],  # x-axis sides
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],  # x-axis sides
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],  # x-axis sides
                [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],  # y-axis bottom/top
                [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],  # y-axis bottom/top
                [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],  # y-axis bottom/top
                [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],  # y-axis bottom/top
                [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # x-axis sides
                [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # x-axis sides
                [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # x-axis sides
                [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # x-axis sides
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],  # y-axis top/bottom
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],  # y-axis top/bottom
            ], 
            dtype=torch.float32, device=self.device
        )
        self.reorientation_contact_pos_o[:, :, 0] *= self.object_size[0] / 2
        self.reorientation_contact_pos_o[:, :, 1] *= self.object_size[1] / 2
        
        self.reorientation_desired_object_quat_ = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.9238795325, 0.0, 0.0, 0.3826834324],
                [0.9238795325, 0.0, 0.0, 0.3826834324],
                [0.7071067812, 0.0, 0.0, 0.7071067812],
                [0.7071067812, 0.0, 0.0, 0.7071067812],
                [0.3826834324, 0.0, 0.0, 0.9238795325],
                [0.3826834324, 0.0, 0.0, 0.9238795325],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [-0.3826834324, 0.0, 0.0, 0.9238795325],
                [-0.3826834324, 0.0, 0.0, 0.9238795325],
                [-0.7071067812, 0.0, 0.0, 0.7071067812],
                [-0.7071067812, 0.0, 0.0, 0.7071067812],
                [-0.9238795325, 0.0, 0.0, 0.3826834324],
                [-0.9238795325, 0.0, 0.0, 0.3826834324],
                # [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32, device=self.device,
        )
        
        self.contact_pose_o = torch.zeros(2, 7, dtype=torch.float32, device=self.device)
        
        # self.contact_pose_o[:, :3] = self.repose_contact_pos_o[self.current_goal_idx, :, :3]
        self.contact_pose_o[:, :3] = self.reorientation_contact_pos_o[self.current_goal_idx, :, :3]
        self.contact_pose_o[:, 3] = 1.0
        
        # self.contact_pose_b = torch.zeros_like(self.contact_pose_o)
        self.contact_pose_w = torch.zeros_like(self.contact_pose_o)
        
        self.actions_mapping = torch.arange(len(self.policy_joint_indices), dtype=torch.int32, device=self.device)
        self.init_goal_pos_w = torch.tensor([0.35, 0.0, 0.75], dtype=torch.float32, device=self.device)
        self.object_goal_pose_w = torch.zeros(7, dtype=torch.float32, device=self.device)
        self.object_goal_pose_w[:3] = self.init_goal_pos_w.clone()
        self.object_goal_pose_w[2] += self.object_size[2] / 2
        self.object_goal_pose_w[3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        
        self.object_rpy_ranges = torch.tensor([[-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]], dtype=torch.float32, device=self.device)
        # Frequency tracking (logger will be set later in set_cmd_manager)
        self._frequency_tracker = FrequencyTracker(
            name="compute_lowlevelcmd",
            log_interval=2.0,  # Reduced for easier testing
            logger=self.logger  # Will be updated when logger is available
        )
        
        self.task = "repose"

    def register_observations(self):
        """
        Register observations for contact-conditioned locomotion.
        Includes contact pattern and timing information.
        """
        from state_manager.observations import (
            contact_plan,
            contact_time_left,
            object_pos_robot_xy_frame,
            root_quat_w,
            contact_pos_error,
            lin_vel_w,
            ang_vel_w,
            joint_pos_rel,
            joint_vel,
            last_action,
            goal_pose_diff,
        )

        # - Joint observations
        self.obs_manager.register(
            "joint_pos",
            ObsTerm(
                joint_pos_rel,
                params={
                    "default_joint_pos": self.default_joint_pos,
                    "mapping": self.policy_joint_indices,
                },
                obs_dim=len(self.policy_joint_names),
                device=self.device,
            ),
        )
        self.obs_manager.register(
            "joint_vel",
            ObsTerm(
                joint_vel,
                params={
                    "mapping": self.policy_joint_indices,
                    # "scale": 0.2,
                },
                obs_dim=len(self.policy_joint_names),
                device=self.device,
            ),
        )
        
        # - Object observations
        self.obs_manager.register(
            "object_pos",
            ObsTerm(
                object_pos_robot_xy_frame,
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

        # - Contact observations
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
        self.obs_manager.register(
            "contact_command",
            ObsTerm(
                contact_pos_error,
                params={"contact_pose_w": lambda: self.contact_pose_w[:, :3], "mj_model": self.robot.mj_model},
                obs_dim=6,
                device=self.device,
            ),
        )
        
        # - Object command observations
        self.obs_manager.register("goal_pose_diff", ObsTerm(goal_pose_diff, params={"asset_name": "object", "goal_pose_w": lambda: self.object_goal_pose_w}, obs_dim=7, device=self.device))
        
        # - Action observation
        self.obs_manager.register(
            "last_action",
            ObsTerm(last_action, params={"last_action": lambda: self.raw_action}, obs_dim=len(self.policy_joint_indices), device=self.device),
        )

    def register_commands(self):
        """Register contact command parameters."""
        # Register task selection as button group
        task_options = ["repose", "reorientation"]
        self.command_manager.register_button_command(
            name="task",
            description="Select Task",
            options=task_options,
            default_value=self.task,
        )

    def set_mode(self):
        """Runs when the mode is changed in the UI.
        Generates the future feet positions in the init frame for horizon planning.
        """
        # Call the base class set_mode to activate the controller
        super().set_mode()

        # Set default task when switching back to RL controller
        with self._command_lock:
            self.current_goal_idx = 0
            
            # Clear any pending task changes
            self.pending_task_change = None
            self.task_change_pending = False
            
            # Set contact plan based on current task
            if self.task == "repose":
                self.current_contact_plan = self.repose_contact_plan_[:, 0:2]
                self.contact_pose_o[:, :3] = self.repose_contact_pos_o[0, :, :3]
            else:  # reorientation
                self.current_contact_plan = self.reorientation_contact_plan_[:, 0:2]
                self.contact_pose_o[:, :3] = self.reorientation_contact_pos_o[0, :, :3]
            
            self.contact_pose_o[:, 3] = 1.0  # Set quaternion w component to 1 (identity)
            
            self.object_goal_pose_w[:3] = self.init_goal_pos_w.clone()
            self.object_goal_pose_w[2] += self.object_size[2] / 2
            self.object_goal_pose_w[3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

            self.command_duration = 1.3

    def compute_lowlevelcmd(self, state):
        """
        Compute motor commands using the learned policy.

        :param state: Current robot state
        :return: Motor commands dictionary
        """
        # Frequency tracking
        # frequency = self._frequency_tracker.tick()
        # self.logger.debug(f"compute_lowlevelcmd frequency: {self._frequency_tracker.get_statistics()['current_frequency']:.2f} Hz")

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
        self._update_commands()
        
        # Publish visualizations to RViz
        self.pub_all_visualizations()

        # If we do not use threading, we need to compute the obs first, pass it to the policy, and then compute the joint pos targets
        # else we can compute the joint pos targets directly from the obs tensor
        if self.counter % self.decimation == 0:
            if not self.use_threading:
                obs_tensor = self.obs_manager.compute_full_tensor(self.latest_state, batch_idx=0)
                self.joint_pos_targets[self.policy_joint_indices] = self.compute_joint_pos_targets_from_policy(obs_tensor)
            else:
                self.joint_pos_targets[self.policy_joint_indices] = self.compute_joint_pos_targets()
                            
        # # Clip the joint pos targets for safety
        if hasattr(self, "soft_dof_pos_limit"):
            self.joint_pos_targets = self._clip_dof_pos(self.joint_pos_targets)

    
        for idx, joint_idx in enumerate(self.policy_joint_indices):
            self.cmd[f"motor_{joint_idx}"] = {
                "q": self.joint_pos_targets[joint_idx].item(),
                # "q": self.default_joint_pos_np[idx],
                "kp": self.Kp[idx],
                "dq": 0.0,
                "kd": self.Kd[idx],
                "tau": 0.0,
            }
        for idx, joint_idx in enumerate(self.non_policy_joint_indices):
            self.cmd[f"motor_{joint_idx}"] = {
                "q": self.non_policy_default_angles[idx],
                "kp": self.non_policy_joint_Kps[idx],
                "dq": 0.0,
                "kd": self.non_policy_joint_Kds[idx],
                "tau": 0.0,
            }

        # Track command preparation time
        self.cmd_preparation_time = time.perf_counter() - start_time

                
        self.cmd["mode_pr"] = self.robot.MotorMode.PR
        self.cmd["mode_machine"] = state["mode_machine"]

        return self.cmd
    
    def _resample_commands(self):
        """Resample the commands for the controller with thread safety."""
        try:
            with self._command_lock:
                
                # Handle pending task change
                if self.task_change_pending and self.pending_task_change is not None:
                    self._handle_task_change()
                
                dtheta = quat_error_magnitude(self.latest_state["object/base_quat"], self.object_goal_pose_w[3:7])
                if dtheta < 0.3:
                    
                    if self.current_goal_idx + 2 > self.repose_contact_pos_o.shape[0]:
                        self.current_goal_idx = 2
                        
                    # Update contact location on object based on current goal index
                    self._update_contact_pose()
                    # Update object goal pose
                    self._update_object_goal_pose()
                    
                    # update contact plan based on current task
                    if self.task == "repose":
                        self.current_contact_plan = self.repose_contact_plan_[:, self.current_goal_idx : self.current_goal_idx + 2]
                    else:  # reorientation
                        self.current_contact_plan = self.reorientation_contact_plan_[:, self.current_goal_idx : self.current_goal_idx + 2]
                    
                    self.current_goal_idx += 1
                    
                self.logger.debug(f"Current goal index: {self.current_goal_idx}")
                self.logger.debug(f"Current contact plan: {self.current_contact_plan}")
                self.logger.debug(f"Object goal pose: {self.object_goal_pose_w}")
                self.logger.debug("--------------------------------")

        except Exception as e:
            if hasattr(self, "command_manager") and self.command_manager and hasattr(self.command_manager, "logger"):
                self.logger.error(f"Error resampling commands: {e}")
            else:
                print(f"Error resampling commands: {e}")

    def _handle_task_change(self):
        """Handle the pending task change by applying it and resetting state."""
        try:
            if self.pending_task_change is not None:
                old_task = self.task
                self.task = self.pending_task_change
                
                # Reset to initial state for the new task
                self.current_goal_idx = 0
                
                if self.task == "repose":
                    self.current_contact_plan = self.repose_contact_plan_[:, 0:2]
                    self.contact_pose_o[:, :3] = self.repose_contact_pos_o[0, :, :3]
                else:  # reorientation
                    self.current_contact_plan = self.reorientation_contact_plan_[:, 0:2]
                    self.contact_pose_o[:, :3] = self.reorientation_contact_pos_o[0, :, :3]
                
                self.contact_pose_o[:, 3] = 1.0  # Set quaternion w component to 1 (identity)
                
                # Reset object goal pose
                self.object_goal_pose_w[:3] = self.init_goal_pos_w.clone()
                self.object_goal_pose_w[2] += self.object_size[2] / 2
                self.object_goal_pose_w[3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
                
                # Clear pending change
                self.pending_task_change = None
                self.task_change_pending = False
                
                if self.logger is not None:
                    self.logger.debug(f"Task changed from {old_task} to {self.task}")
                    
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Error handling task change: {e}")

    def _update_commands(self):
        """Update contact poses in world and base frames."""
        try:
            # Get object pose from state
            object_pos_w = self.latest_state["object/base_pos_w"]
            object_quat_w = self.latest_state["object/base_quat"]

            # Contact poses in the world frame
            self.contact_pose_w[:, :3], self.contact_pose_w[:, 3:7] = combine_frame_transforms(
                t01=object_pos_w.unsqueeze(0).expand(2, -1),
                q01=object_quat_w.unsqueeze(0).expand(2, -1),
                t12=self.contact_pose_o[:, :3],
                q12=self.contact_pose_o[:, 3:7],
            )
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"Error updating contact poses: {e}")
            else:
                print(f"Error updating contact poses: {e}")

    def _update_contact_pose(self):
        """Update contact pose_o based on current goal index and task."""
        try:
            # Update contact pose_o based on current goal index and task
            if self.task == "repose" and self.current_goal_idx < self.repose_contact_pos_o.shape[0]:
                self.contact_pose_o[:, :3] = self.repose_contact_pos_o[self.current_goal_idx, :, :3]
            elif self.task == "reorientation" and self.current_goal_idx < self.reorientation_contact_pos_o.shape[0]:
                self.contact_pose_o[:, :3] = self.reorientation_contact_pos_o[self.current_goal_idx, :, :3]
            
            self.contact_pose_o[:, 3] = 1.0  # Set quaternion w component to 1 (identity)
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"Error updating contact pose_o: {e}")
            else:
                print(f"Error updating contact pose_o: {e}")
                
    def _update_object_goal_pose(self):
        """Update object goal pose based on current goal index and task."""
        try:
            self.object_goal_pose_w[:3] = self.init_goal_pos_w.clone()
            self.object_goal_pose_w[2] += self.object_size[2] / 2
            
            if self.task == "repose":
                # For repose task, keep object in default orientation
                self.object_goal_pose_w[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
            else:  # reorientation
                # For reorientation task, use the desired quaternion sequence
                if self.current_goal_idx < self.reorientation_desired_object_quat_.shape[0]:
                    self.object_goal_pose_w[3:7] = self.reorientation_desired_object_quat_[self.current_goal_idx, :]
                else:
                    self.object_goal_pose_w[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"Error updating object goal pose: {e}")
    
    """
    Joystick mappings and callbacks        
    """

    def get_joystick_mappings(self):
        """
        Define joystick button mappings for task changes and control.

        Returns:
            Dict mapping button names to callback functions.
        """
        return {
            # Task selection
            "A": lambda: self.change_commands({"task": "repose"}),
            "B": lambda: self.change_commands({"task": "reorientation"}),
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

        This method handles changes to the robot's task. When a new task is requested,
        it is stored as a pending change rather than applied immediately. The actual task
        transition happens during the next resampling phase to ensure smooth transitions.

        Args:
            new_commands: Dictionary containing command updates. Currently supports:
                - 'task': String specifying the new task ('repose' or 'reorientation')

        Raises:
            ValueError: If an invalid task is specified
        """
        try:
            if "task" in new_commands:
                new_task = new_commands["task"].lower()
                if new_task in ["repose", "reorientation"] and new_task != self.task:
                    with self._command_lock:
                        # Only set pending change if it's different from current task
                        if self.pending_task_change != new_task:
                            self.pending_task_change = new_task
                            self.task_change_pending = True
                            
                            if self.logger is not None:
                                self.logger.debug(f"Pending task change to: {new_task}")

        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Task command update failed: {e}")


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
                self.pending_command_duration = min(1.5, self.command_duration + duration_change)
            elif direction == "decrease":
                # Decrease command duration (with a minimum value)
                self.pending_command_duration = max(1.0, self.command_duration - duration_change)
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
            self.pending_command_duration = 1.3
            self.command_duration_change_pending = True

            if self.logger is not None:
                self.logger.debug("Reset command duration to default")


    """
    Helper functions to initialize publishers for visualization.
    """

    def _init_publishers(self, configs):
        """
        Initialize publishers for visualization.
        """
        # Initialize visualization flags from configs
        self.visualize = {
            key: configs["controller_config"]["visualize"][key] for key in configs["controller_config"]["visualize"]
        }

        if self.visualize.get("object_pose", False):
            # Add marker publisher for cuboid visualization
            self.object_marker_pub = self.create_publisher(Marker, "object_marker", 10)
        if self.visualize.get("contact_positions", False):
            # Add marker publisher for contact positions
            self.contact_markers_pub = self.create_publisher(MarkerArray, "contact_markers", 10)
        if self.visualize.get("desired_object_pose", False):
            # Add marker publisher for desired object pose
            self.desired_object_marker_pub = self.create_publisher(Marker, "desired_object_marker", 10)

    def pub_object_pose(self):
        """
        Publish the current object pose as a cuboid marker in RViz.
        """
        try:
            if not hasattr(self, 'latest_state') or "object/base_pos_w" not in self.latest_state:

                return
                
            # Check if publisher is initialized
            if not hasattr(self, 'object_marker_pub'):
                return
                
            # Get object pose from state
            object_pos_w = torch.tensor(self.latest_state["object/base_pos_w"], dtype=torch.float32, device=self.device)
            object_quat_w = torch.tensor(self.latest_state["object/base_quat"], dtype=torch.float32, device=self.device)
            
            # self.logger.debug(f"Object pose: {object_pos_w}, {object_quat_w}")
            # Create cuboid marker
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "object"
            marker.id = 0
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Set position and orientation
            marker.pose.position.x = float(object_pos_w[0])
            marker.pose.position.y = float(object_pos_w[1])
            marker.pose.position.z = float(object_pos_w[2])
            marker.pose.orientation.w = float(object_quat_w[0])
            marker.pose.orientation.x = float(object_quat_w[1])
            marker.pose.orientation.y = float(object_quat_w[2])
            marker.pose.orientation.z = float(object_quat_w[3])
            
            # Set scale (cuboid size) - ensure minimum size for visibility
            marker.scale.x = float(self.object_size[0])
            marker.scale.y = float(self.object_size[1])
            marker.scale.z = float(self.object_size[2])
            
            # Set color (blue with transparency)
            marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=1.0)  # Increased alpha to 1.0 for better visibility
            
            # Publish marker
            self.object_marker_pub.publish(marker)
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"Failed to publish object pose: {e}")

    def pub_contact_positions(self):
        """
        Publish the contact positions as sphere markers in RViz.
        """
        try:
            # Check if publisher is initialized
            if not hasattr(self, 'contact_markers_pub'):
                return
                
            # Create marker array for contact positions
            marker_array = MarkerArray()
            
            
            for i in range(2):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "contact_positions"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                # Set position from contact_pose_w
                marker.pose.position.x = float(self.contact_pose_w[i, 0])
                marker.pose.position.y = float(self.contact_pose_w[i, 1])
                marker.pose.position.z = float(self.contact_pose_w[i, 2])
                
                # Set orientation (identity quaternion)
                marker.pose.orientation.w = 1.0
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                
                # Set scale (sphere radius)
                marker.scale.x = 0.03
                marker.scale.y = 0.03
                marker.scale.z = 0.03
                
                # Set color based on contact plan
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) if self.current_contact_plan[i, 0] else ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)         
                marker_array.markers.append(marker)
            
            # Publish marker array
            self.contact_markers_pub.publish(marker_array)
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"Failed to publish contact positions: {e}")

    def pub_desired_object_pose(self):
        """
        Publish the desired object pose as a wireframe cuboid marker in RViz.
        """
        try:
            # Check if publisher is initialized
            if not hasattr(self, 'desired_object_marker_pub'):
                return
                
            # Create wireframe cuboid marker
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "desired_object"
            marker.id = 0
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Set position and orientation from object_goal_pose_w
            marker.pose.position.x = float(self.object_goal_pose_w[0])
            marker.pose.position.y = float(self.object_goal_pose_w[1])
            marker.pose.position.z = float(self.object_goal_pose_w[2])
            marker.pose.orientation.w = float(self.object_goal_pose_w[3])
            marker.pose.orientation.x = float(self.object_goal_pose_w[4])
            marker.pose.orientation.y = float(self.object_goal_pose_w[5])
            marker.pose.orientation.z = float(self.object_goal_pose_w[6])
            
            # Set scale (cuboid size)
            marker.scale.x = float(self.object_size[0])
            marker.scale.y = float(self.object_size[1])
            marker.scale.z = float(self.object_size[2])
            
            # Set color (green wireframe for desired pose)
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.3)
            
            # Publish marker
            self.desired_object_marker_pub.publish(marker)

            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"Failed to publish desired object pose: {e}")

    def pub_all_visualizations(self):
        """
        Publish all visualization data to RViz.
        """
        try:
            if hasattr(self, 'visualize'):
                if self.visualize.get("object_pose", False):
                    self.pub_object_pose()
                if self.visualize.get("contact_positions", False):
                    self.pub_contact_positions()
                if self.visualize.get("desired_object_pose", False):
                    self.pub_desired_object_pose()

        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"Failed to publish visualizations: {e}")
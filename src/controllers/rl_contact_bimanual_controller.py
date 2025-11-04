import threading
import time
from typing import TYPE_CHECKING, Any, Dict

from std_msgs.msg import ColorRGBA
import torch

# ROS2 imports for visualization
from visualization_msgs.msg import Marker, MarkerArray

from controllers.action_terms import EMAJointPositionToLimitsAction, JointPositionAction
from controllers.rl_controller_base import RLControllerBase
from state_manager.obs_manager import ObsTerm
from utils.frequency_tracker import FrequencyTracker, MultiFrequencyTracker
from utils.math import combine_frame_transforms, euler_to_quaternion, quat_error_magnitude, quaternion_to_euler

if TYPE_CHECKING:
    from robots.robot_base import RobotBase

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 5 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 1.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


class RLHumanoidBimanualContactController(RLControllerBase):
    """
    Contact-conditioned RL Bimanual Controller
    Uses contact-explicit reinforcement learning policy
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):

        super().__init__(robot=robot, configs=configs, debug=debug)

        # Initialize publishers for visualization
        self._init_publishers(configs)

        self.policy_joint_names = [
            # "waist_yaw_joint",
            # "waist_roll_joint",
            # "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            # "left_wrist_roll_joint",
            # "left_wrist_pitch_joint",
            # "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            # "right_wrist_roll_joint",
            # "right_wrist_pitch_joint",
            # "right_wrist_yaw_joint",
        ]

        # self.policy_activated_indices = torch.tensor([0,1,2,3,4,5,6,10,11,12,13], dtype=torch.int32, device=self.device)
        self.policy_activated_indices = torch.tensor([0, 1, 2, 3, 7, 8, 9, 10], dtype=torch.int32, device=self.device)

        self.policy_stiffness = torch.tensor(
            [
                # STIFFNESS_7520_14 * 2.0, 2.0 * STIFFNESS_5020 * 2.0 , 2.0 * STIFFNESS_5020 * 2.0,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_4010,
                STIFFNESS_4010,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_4010,
                STIFFNESS_4010,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.policy_damping = torch.tensor(
            [
                # DAMPING_7520_14 * 2.0, 2.0 * DAMPING_5020 * 2.0 , 2.0 * DAMPING_5020 * 2.0,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_4010,
                DAMPING_4010,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_4010,
                DAMPING_4010,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.policy_joint_indices = [
            self.robot.actuated_joint_names.index(joint_name) for joint_name in self.policy_joint_names
        ]

        self.non_policy_leg_joint_names = [
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
        ]
        self.non_policy_waist_joint_names = [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ]
        self.non_policy_leg_joint_indices = [
            self.robot.actuated_joint_names.index(joint_name)
            for joint_name in self.non_policy_leg_joint_names
            if joint_name in self.robot.actuated_joint_names + self.robot.non_actuated_joint_names
        ]
        self.non_policy_leg_joint_stiffness = torch.tensor(
            [350.0, 200.0, 200.0, 300.0, 300.0, 150.0, 350.0, 200.0, 200.0, 300.0, 300.0, 150.0],
            dtype=torch.float32,
            device=self.device,
        )
        self.non_policy_leg_joint_damping = torch.tensor(
            [5.0, 5.0, 5.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 10.0, 5.0, 5.0], dtype=torch.float32, device=self.device
        )
        self.non_policy_leg_default_angles = [
            -0.312,
            0.0,
            0.0,
            0.669,
            -0.33,
            0.0,
            -0.312,
            0.0,
            0.0,
            0.669,
            -0.33,
            0.0,
        ]

        self.non_policy_waist_joint_indices = [
            self.robot.actuated_joint_names.index(joint_name)
            for joint_name in self.non_policy_waist_joint_names
            if joint_name in self.robot.actuated_joint_names + self.robot.non_actuated_joint_names
        ]
        self.non_policy_waist_joint_stiffness = torch.tensor(
            [200.0, 200.0, 200.0], dtype=torch.float32, device=self.device
        )
        self.non_policy_waist_joint_damping = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32, device=self.device)
        self.non_policy_waist_default_angles = [0.0, 0.0, 0.0]

        if self.robot.interface == "real":
            self.non_policy_leg_joint_stiffness[self.non_policy_leg_joint_indices] = 0.0
            self.non_policy_leg_joint_damping[self.non_policy_leg_joint_indices] = 0.00

        self.effort_limit = torch.tensor(self.robot.effort_limit, dtype=torch.float32, device=self.device)

        self.policy_deactivate_joint_names = [
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
        ]
        self.policy_deactivate_joint_indices = [
            self.robot.actuated_joint_names.index(joint_name)
            for joint_name in self.policy_deactivate_joint_names
            if joint_name in self.robot.actuated_joint_names + self.robot.non_actuated_joint_names
        ]

        # Pre-compute the combined joint indices to avoid repeated set operations
        self.combined_joint_indices = list(set(self.policy_joint_indices + self.policy_deactivate_joint_indices))
        self.action_scale = torch.zeros(len(self.combined_joint_indices), dtype=torch.float32, device=self.device)

        # Contact command parameters
        self.init_command_duration = 0.5
        self.command_duration = self.init_command_duration
        self.object_size = torch.tensor([0.1, 0.1, 0.115], dtype=torch.float32, device=self.device) * 2.0

        self.detach_contact_plan_ = torch.tensor(
            [
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ],
            dtype=torch.bool,
            device=self.device,
        )

        self.repose_contact_plan_ = torch.tensor(
            [
                [False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                [False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
            ],
            dtype=torch.bool,
            device=self.device,
        )

        self.reorientation_contact_plan_ = torch.tensor(
            [
                [
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                ],
                [
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                ],
            ],
            dtype=torch.bool,
            device=self.device,
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

        self.repose_contact_pos_o = torch.tensor(
            [
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
            dtype=torch.float32,
            device=self.device,
        )
        self.repose_contact_pos_o[:, :, 0] *= self.object_size[0] / 2 + 0.005
        self.repose_contact_pos_o[:, :, 1] *= self.object_size[1] / 2 + 0.005

        self.reorientation_contact_pos_o = torch.tensor(
            [
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
            dtype=torch.float32,
            device=self.device,
        )
        self.reorientation_contact_pos_o[:, :, 0] *= self.object_size[0] / 2 + 0.005
        self.reorientation_contact_pos_o[:, :, 1] *= self.object_size[1] / 2 + 0.005

        self.reorientation_desired_object_quat_ = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                # [1.0, 0.0, 0.0, 0.0],
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
            dtype=torch.float32,
            device=self.device,
        )

        self.contact_pose_o = torch.zeros(2, 7, dtype=torch.float32, device=self.device)

        # self.contact_pose_o[:, :3] = self.repose_contact_pos_o[self.current_goal_idx, :, :3]
        self.contact_pose_o[:, :3] = self.reorientation_contact_pos_o[self.current_goal_idx, :, :3]
        self.contact_pose_o[:, 3] = 1.0

        # self.contact_pose_b = torch.zeros_like(self.contact_pose_o)
        self.contact_pose_w = torch.zeros_like(self.contact_pose_o)

        self.actions_mapping = torch.arange(self.action_dim, dtype=torch.int32, device=self.device)
        # self.init_goal_pos_w = torch.tensor([0.35, 0.0, 0.75], dtype=torch.float32, device=self.device)
        # self.init_goal_pos_w = torch.tensor([0.3, 0.0, 0.75], dtype=torch.float32, device=self.device)
        self.init_goal_pos_w = None
        self.object_goal_pose_w = torch.zeros(7, dtype=torch.float32, device=self.device)
        self.object_goal_pose_w[3] = 1.0

        self.object_rpy_ranges = torch.tensor(
            [[-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]], dtype=torch.float32, device=self.device
        )

        # Joystick control parameters for object goal pose
        self.position_step_size = 0.05  # 1cm steps for position
        self.orientation_step_size = 0.15  # ~3 degrees for orientation (in radians)

        # Pose queue system for future poses
        self.num_future_poses = 2  # current and next pose
        self.object_goal_poses_w = torch.zeros(self.num_future_poses, 7, dtype=torch.float32, device=self.device)

        # Initialize pose queue with initial pose
        for i in range(self.num_future_poses):
            self.object_goal_poses_w[i] = self.object_goal_pose_w.clone()

        # Pending pose updates from joystick
        self.pending_pose_update = None  # Store pending pose update
        self.pose_update_pending = False  # Flag to indicate pending pose update

        # Frequency tracking (logger will be set later in set_cmd_manager)
        self._frequency_tracker = FrequencyTracker(
            name="compute_lowlevelcmd",
            log_interval=2.0,  # Reduced for easier testing
            logger=self.logger,  # Will be updated when logger is available
        )
        # self.action_term = JointPositionAction(configs=configs, action_scale=self.action_scale, default_joint_pos=self.default_joint_pos, actions_mapping=self.actions_mapping)
        self.action_term = EMAJointPositionToLimitsAction(
            configs=configs,
            action_scale=self.action_scale,
            actions_mapping=self.actions_mapping,
            default_joint_pos=self.default_joint_pos,
            soft_joint_pos_limits=self.soft_dof_pos_limit,
            joint_ids=torch.tensor(self.combined_joint_indices, dtype=torch.int32, device=self.device),
        )
        self.task = "detach"

    def get_current_goal_pose(self):
        """Get the current goal pose from the pose queue (always index 0)."""
        return self.object_goal_poses_w[0].clone()

    def register_observations(self):
        """
        Register observations for contact-conditioned locomotion.
        Includes contact pattern and timing information.
        """
        from state_manager.observations import (
            ang_vel_w,
            contact_plan,
            contact_pos_error,
            contact_time_left,
            goal_pose_diff,
            joint_pos_rel,
            joint_vel,
            last_action,
            lin_vel_w,
            object_pos_robot_xy_frame,
            projected_gravity_b,
            root_quat_w,
        )

        # - Joint observations
        self.obs_manager.register(
            "joint_pos",
            ObsTerm(
                joint_pos_rel,
                params={
                    "default_joint_pos": self.default_joint_pos[self.policy_activated_indices],
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

        # self.obs_manager.register("projected_gravity", ObsTerm(projected_gravity_b, obs_dim=3, device=self.device))

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
        self.obs_manager.register(
            "object_lin_vel_w", ObsTerm(lin_vel_w, params={"asset_name": "object"}, obs_dim=3, device=self.device)
        )
        self.obs_manager.register(
            "object_ang_vel_w", ObsTerm(ang_vel_w, params={"asset_name": "object"}, obs_dim=3, device=self.device)
        )

        # - Contact observations
        self.obs_manager.register(
            "contact_time_left",
            ObsTerm(
                contact_time_left, params={"contact_time_left": lambda: self.time_left}, obs_dim=1, device=self.device
            ),
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
        self.obs_manager.register(
            "goal_pose_diff",
            ObsTerm(
                goal_pose_diff,
                params={"asset_name": "object", "goal_poses_w": lambda: self.object_goal_poses_w},
                obs_dim=self.num_future_poses * 7,
                device=self.device,
            ),
        )

        # # - Action observation
        self.obs_manager.register(
            "last_action",
            ObsTerm(
                last_action,
                params={"last_action": lambda: self.action_term.raw_action},
                obs_dim=len(self.combined_joint_indices),
                device=self.device,
            ),
        )

    def register_commands(self):
        """Register contact command parameters."""
        # Register task selection as button group
        task_options = ["repose", "reorientation", "detach"]
        self.command_manager.register_button_command(
            name="task",
            description="Select Task",
            options=task_options,
            default_value=self.task,
        )

    # def set_mode(self):
    #     """Runs when the mode is changed in the UI.
    #     Generates the future feet positions in the init frame for horizon planning.
    #     """
    #     # Call the base class set_mode to activate the controller
    #     super().set_mode()

    #     # Set default task when switching back to RL controller
    #     with self._command_lock:
    #         self.current_goal_idx = 0

    #         # Clear any pending task changes
    #         self.pending_task_change = None
    #         self.task_change_pending = False

    #         # Set contact plan based on current task
    #         if self.task == "repose":
    #             self.current_contact_plan = self.repose_contact_plan_[:, 0:2]
    #             self.contact_pose_o[:, :3] = self.repose_contact_pos_o[0, :, :3]
    #         else:  # reorientation
    #             self.current_contact_plan = self.reorientation_contact_plan_[:, 0:2]
    #             self.contact_pose_o[:, :3] = self.reorientation_contact_pos_o[0, :, :3]

    #         self.contact_pose_o[:, 3] = 1.0  # Set quaternion w component to 1 (identity)

    #         self.object_goal_pose_w[:3] = self.init_goal_pos_w.clone()
    #         self.object_goal_pose_w[2] += self.object_size[2] / 2
    #         self.object_goal_pose_w[3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

    #         # Initialize pose queue with initial pose
    #         for i in range(self.num_future_poses):
    #             self.object_goal_poses_w[i] = self.object_goal_pose_w.clone()

    #         self.command_duration = self.init_command_duration

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

        if self.init_goal_pos_w is None:
            self.init_goal_pos_w = state["object/base_pos_w"]
            self.object_goal_pose_w[:3] = self.init_goal_pos_w.clone()
            self.object_goal_pose_w[3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

            # Reset pose queue with initial pose
            for i in range(self.num_future_poses):
                self.object_goal_poses_w[i] = self.object_goal_pose_w.clone()

        start_time = time.perf_counter()

        # Update the latest state for the observation processing thread
        with self._lock:
            self.latest_state = state

        # Update contact plan timing with thread safety
        with self._command_lock:
            current_time = time.time()
            elapsed = current_time - self.command_start_time
            # self.time_left = max(0, self.command_duration - elapsed)
            self.time_left = self.command_duration
            if elapsed >= self.command_duration:
                self.command_start_time = current_time
                self._resample_commands()

        # Update object goal pose
        self._update_object_goal_pose()

        # Compute contact poses
        self._update_commands()

        # Publish visualizations to RViz
        self.pub_all_visualizations()

        # If we do not use threading, we need to compute the obs first, pass it to the policy, and then compute the joint pos targets
        # else we can compute the joint pos targets directly from the obs tensor
        if self.counter % self.decimation == 0:
            if not self.use_threading:
                obs_tensor = self.obs_manager.compute_full_tensor(self.latest_state, batch_idx=0)
                self.joint_pos_targets[self.combined_joint_indices] = self.compute_joint_pos_targets_from_policy(
                    obs_tensor
                )
            else:
                self.joint_pos_targets[self.combined_joint_indices] = self.compute_joint_pos_targets()

        # # Clip the joint pos targets for safety based on effort limits
        # self.joint_pos_targets[self.combined_joint_indices] = self._clip_joint_pos_by_effort_limit(
        #     joint_pos_targets=self.joint_pos_targets[self.combined_joint_indices],
        #     joint_pos=state["robot/joint_pos"][self.combined_joint_indices],
        #     joint_vel=state["robot/joint_vel"][self.combined_joint_indices],
        # )

        # Log each observation with key and value from
        # if self.debug:
        # for name, obs_term in self.obs_manager.obs_terms.items():
        #     self.logger.debug(f"{name}: {obs_term(self.latest_state)}")
        # self.logger.debug("--------------------------------")
        # self.logger.debug(f"Joint pos targets: {self.joint_pos_targets[self.robot.actuated_joint_indices]}")

        for idx, joint_idx in enumerate(self.combined_joint_indices):
            self.cmd[f"motor_{joint_idx}"] = {
                "q": self.joint_pos_targets[joint_idx].item(),
                # "q": self.default_joint_pos_np[idx],
                "kp": self.policy_stiffness[idx],
                "dq": 0.0,
                "kd": self.policy_damping[idx],
                "tau": 0.0,
            }
        for idx, joint_idx in enumerate(self.non_policy_leg_joint_indices):
            self.cmd[f"motor_{joint_idx}"] = {
                "q": self.non_policy_leg_default_angles[idx],
                "kp": self.non_policy_leg_joint_stiffness[idx],
                "dq": 0.0,
                "kd": self.non_policy_leg_joint_damping[idx],
                "tau": 0.0,
            }
        for idx, joint_idx in enumerate(self.non_policy_waist_joint_indices):
            self.cmd[f"motor_{joint_idx}"] = {
                "q": self.non_policy_waist_default_angles[idx],
                "kp": self.non_policy_waist_joint_stiffness[idx],
                "dq": 0.0,
                "kd": self.non_policy_waist_joint_damping[idx],
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

                current_goal_pose = self.get_current_goal_pose()
                dtheta = quat_error_magnitude(self.latest_state["object/base_quat"], current_goal_pose[3:7])
                # if dtheta < 0.3:
                if self.current_goal_idx + 2 > self.repose_contact_pos_o.shape[0]:
                    self.current_goal_idx = 2

                # Update contact location on object based on current goal index
                self._update_contact_pose()

                # update contact plan based on current task
                if self.task == "repose":
                    self.current_contact_plan = self.repose_contact_plan_[
                        :, self.current_goal_idx : self.current_goal_idx + 2
                    ]
                elif self.task == "detach":
                    self.current_contact_plan = self.detach_contact_plan_[
                        :, self.current_goal_idx : self.current_goal_idx + 2
                    ]
                elif self.task == "reorientation":  # reorientation
                    self.current_contact_plan = self.reorientation_contact_plan_[
                        :, self.current_goal_idx : self.current_goal_idx + 2
                    ]
                self.current_goal_idx += 1

                # self.logger.debug(f"Current goal index: {self.current_goal_idx}")
                # self.logger.debug(f"Current contact plan: {self.current_contact_plan}")
                # self.logger.debug(f"Current goal pose: {self.get_current_goal_pose()}")
                # self.logger.debug("--------------------------------")

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
                # self.object_goal_pose_w[2] += self.object_size[2] / 2
                self.object_goal_pose_w[3:] = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
                )

                # Reset pose queue with initial pose
                for i in range(self.num_future_poses):
                    self.object_goal_poses_w[i] = self.object_goal_pose_w.clone()

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
            if hasattr(self, "logger") and self.logger is not None:
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
            elif self.task == "detach" and self.current_goal_idx < self.repose_contact_pos_o.shape[0]:
                self.contact_pose_o[:, :3] = self.repose_contact_pos_o[self.current_goal_idx, :, :3]

            self.contact_pose_o[:, 3] = 1.0  # Set quaternion w component to 1 (identity)
        except Exception as e:
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.error(f"Error updating contact pose_o: {e}")
            else:
                print(f"Error updating contact pose_o: {e}")

    def _update_object_goal_pose(self):
        """Update object goal pose based on current goal index and task."""
        try:
            self.object_goal_poses_w = torch.roll(self.object_goal_poses_w, shifts=-1, dims=0)
            if self.pose_update_pending and self.pending_pose_update is not None:
                self.object_goal_poses_w[-1] = self.pending_pose_update.clone()
                self.pending_pose_update = None
                self.pose_update_pending = False
            else:
                self.object_goal_poses_w[-1] = self.object_goal_poses_w[-2].clone()

            if self.task == "reorientation":
                # For reorientation task, use the desired quaternion sequence
                if self.current_goal_idx < self.reorientation_desired_object_quat_.shape[0]:
                    self.object_goal_poses_w[0, 3:7] = self.reorientation_desired_object_quat_[self.current_goal_idx, :]
                else:
                    self.object_goal_poses_w[0, 3:7] = torch.tensor(
                        [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
                    )

        except Exception as e:
            if hasattr(self, "logger") and self.logger is not None:
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
            # "B": lambda: self.change_commands({"task": "reorientation"}),
            "X": lambda: self.change_commands({"task": "detach"}),
            # Command Duration
            "L2-up": lambda: self._handle_command_duration_change("increase"),
            "L2-down": lambda: self._handle_command_duration_change("decrease"),
            "L2-R2": lambda: self._handle_reset(),
            # Toggle action scale
            "L1-R1": lambda: self._handle_action_scale_change(),
            # Object goal pose position control (only for repose task)
            "L1-up": lambda: (
                self._update_object_goal_position("x", "increase") if self.task in ["repose", "reorientation"] else None
            ),
            "L1-down": lambda: (
                self._update_object_goal_position("x", "decrease") if self.task in ["repose", "reorientation"] else None
            ),
            "left": lambda: (
                self._update_object_goal_position("y", "increase") if self.task in ["repose", "reorientation"] else None
            ),
            "right": lambda: (
                self._update_object_goal_position("y", "decrease") if self.task in ["repose", "reorientation"] else None
            ),
            "up": lambda: self._update_object_goal_position("z", "increase") if self.task == "repose" else None,
            "down": lambda: self._update_object_goal_position("z", "decrease") if self.task == "repose" else None,
            # Object goal pose orientation control (only for repose task)
            "R1-left": lambda: (
                self._update_object_goal_orientation("roll", "increase") if self.task == "repose" else None
            ),
            "R1-right": lambda: (
                self._update_object_goal_orientation("roll", "decrease") if self.task == "repose" else None
            ),
            "R2-left": lambda: (
                self._update_object_goal_orientation("pitch", "increase") if self.task == "repose" else None
            ),
            "R2-right": lambda: (
                self._update_object_goal_orientation("pitch", "decrease") if self.task == "repose" else None
            ),
            "L2-left": lambda: (
                self._update_object_goal_orientation("yaw", "increase") if self.task == "repose" else None
            ),
            "L2-right": lambda: (
                self._update_object_goal_orientation("yaw", "decrease") if self.task == "repose" else None
            ),
        }

    def _handle_action_scale_change(self):
        """
        Handle action scale changes.
        """
        if self.action_scale.sum() == 0.0:

            # Compute base action scale for all policy joints
            self.action_scale = 0.15 * self.effort_limit[self.combined_joint_indices] / self.policy_stiffness

            # Indices in the global actuated joint list
            policy_deactivated_joint_indices = [
                self.robot.actuated_joint_names.index(joint_name)
                for joint_name in self.policy_deactivate_joint_names
                if joint_name in self.robot.actuated_joint_names + self.robot.non_actuated_joint_names
            ]
            # Build a mask over policy joints for those that are policy_deactivated
            if len(policy_deactivated_joint_indices) > 0:
                policy_deactivated_set = set(policy_deactivated_joint_indices)
                policy_policy_deactivated_mask = torch.tensor(
                    [idx in policy_deactivated_set for idx in self.combined_joint_indices],
                    dtype=torch.bool,
                    device=self.device,
                )
                self.action_scale[policy_policy_deactivated_mask] = 0.0
                if self.robot.interface == "real":
                    self.policy_damping[policy_policy_deactivated_mask] = 0.0
                    self.policy_stiffness[policy_policy_deactivated_mask] = 0.0

            self.action_term.action_scale = self.action_scale
        else:
            self.action_scale = torch.zeros(len(self.combined_joint_indices), dtype=torch.float32, device=self.device)
            self.action_term.action_scale = self.action_scale
        if self.logger is not None:
            self.logger.debug(f"Action scale changed to: {self.action_scale}")

    def _update_object_goal_position(self, axis, direction):
        """
        Update object goal position along specified axis.

        Args:
            axis: 'x', 'y', or 'z'
            direction: 'increase' or 'decrease'
        """
        with self._command_lock:
            if axis == "x":
                idx = 0
            elif axis == "y":
                idx = 1
            elif axis == "z":
                idx = 2
            else:
                return

            step = self.position_step_size if direction == "increase" else -self.position_step_size

            # Get current pose and update the specified axis
            current_pose = self.get_current_goal_pose()
            current_pose[idx] += step

            # Set as pending update
            self.pending_pose_update = current_pose
            self.pose_update_pending = True

            if self.logger is not None:
                self.logger.debug(f"Pending object goal position {axis} update: {current_pose[idx]:.3f}")

    def _update_object_goal_orientation(self, axis, direction):
        """
        Update object goal orientation around specified axis.

        Args:
            axis: 'roll', 'pitch', or 'yaw'
            direction: 'increase' or 'decrease'
        """
        with self._command_lock:
            # Get current pose
            current_pose = self.get_current_goal_pose()
            current_quat = current_pose[3:7].clone()

            # Convert current quaternion to RPY using torch operations
            current_rpy = quaternion_to_euler(current_quat, order="wxyz")

            # Update the specified axis
            step = self.orientation_step_size if direction == "increase" else -self.orientation_step_size

            if axis == "roll":
                current_rpy[0] += step
            elif axis == "pitch":
                current_rpy[1] += step
            elif axis == "yaw":
                current_rpy[2] += step
            else:
                return

            # Convert back to quaternion using torch operations
            new_quat = euler_to_quaternion(
                current_rpy[0].item(), current_rpy[1].item(), current_rpy[2].item(), order="wxyz"
            ).to(device=self.device)
            current_pose[3:7] = new_quat

            # Set as pending update
            self.pending_pose_update = current_pose
            self.pose_update_pending = True

            if self.logger is not None:
                self.logger.debug(
                    f"Pending object goal orientation {axis} update: RPY = [{current_rpy[0]:.3f}, {current_rpy[1]:.3f}, {current_rpy[2]:.3f}]"
                )

    def _handle_command_duration_change(self, direction: str):
        """
        Handle command duration changes.

        Args:
            direction: String indicating whether to increase or decrease the duration
        """
        # Define duration change amount in seconds
        duration_change = 0.1  # 100ms change

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

    """
    Function handlers for changing the task for the rl-contact-bimanual controller
    - task, command
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
                if new_task in ["repose", "reorientation", "detach"] and new_task != self.task:
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

    def _handle_reset(self):
        """Reset command duration and offset values to default."""
        with self._command_lock:
            # Reset command duration
            self.pending_command_duration = self.init_command_duration
            self.command_duration_change_pending = True

            self.object_goal_pose_w[:3] = self.init_goal_pos_w.clone()
            self.object_goal_pose_w[2] += self.object_size[2] / 2
            self.object_goal_pose_w[3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

            # Reset pose queue with initial pose
            for i in range(self.num_future_poses):
                self.object_goal_poses_w[i] = self.object_goal_pose_w.clone()

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
        if self.visualize.get("ee_pose", False):
            # Add marker publisher for ee pose visualization
            self.ee_marker_pub = self.create_publisher(MarkerArray, "ee_marker", 10)
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
            if not hasattr(self, "latest_state") or "object/base_pos_w" not in self.latest_state:

                return

            # Check if publisher is initialized
            if not hasattr(self, "object_marker_pub"):
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
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.error(f"Failed to publish object pose: {e}")

    def pub_contact_positions(self):
        """
        Publish the contact positions as sphere markers in RViz.
        """
        try:
            # Check if publisher is initialized
            if not hasattr(self, "contact_markers_pub"):
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
                marker.color = (
                    ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
                    if self.current_contact_plan[i, 0]
                    else ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
                )
                marker_array.markers.append(marker)

            # Publish marker array
            self.contact_markers_pub.publish(marker_array)

        except Exception as e:
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.error(f"Failed to publish contact positions: {e}")

    def pub_desired_object_pose(self):
        """
        Publish the desired object pose as a wireframe cuboid marker in RViz.
        """
        try:
            # Check if publisher is initialized
            if not hasattr(self, "desired_object_marker_pub"):
                return

            # Create wireframe cuboid marker
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "desired_object"
            marker.id = 0
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Set position and orientation from current goal pose
            current_goal_pose = self.get_current_goal_pose()
            marker.pose.position.x = float(current_goal_pose[0])
            marker.pose.position.y = float(current_goal_pose[1])
            marker.pose.position.z = float(current_goal_pose[2])
            marker.pose.orientation.w = float(current_goal_pose[3])
            marker.pose.orientation.x = float(current_goal_pose[4])
            marker.pose.orientation.y = float(current_goal_pose[5])
            marker.pose.orientation.z = float(current_goal_pose[6])

            # Set scale (cuboid size)
            marker.scale.x = float(self.object_size[0])
            marker.scale.y = float(self.object_size[1])
            marker.scale.z = float(self.object_size[2])

            # Set color (green wireframe for desired pose)
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.3)

            # Publish marker
            self.desired_object_marker_pub.publish(marker)

        except Exception as e:
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.error(f"Failed to publish desired object pose: {e}")

    def _clip_joint_pos_by_effort_limit(
        self, joint_pos_targets: torch.Tensor, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> torch.Tensor:
        """
        Clip joint position targets based on 0.9 * effort_limit for PD-like controller.

        This method clips joint positions to ensure the resulting torques from the PD controller
        don't exceed 90% of the motor effort limits, providing safety margins.

        :param joint_pos_targets: Desired joint positions for policy joints
        :return: Clipped joint positions constrained by effort limits
        """
        # Get current joint positions for the policy joints
        current_joint_pos = joint_pos

        # Calculate position error
        pos_error = joint_pos_targets - current_joint_pos

        # Calculate maximum allowed position error based on effort limit
        # For a PD controller: tau = kp * pos_error + kd * vel_error
        # We want: |kp * pos_error + kd * vel| <= 0.9 * effort_limit

        # Calculate current damping torque component (can be positive or negative)
        damping_torque = self.policy_damping * joint_vel

        # Compute available positive and negative torque budgets separately
        available_torque_pos = 0.9 * self.effort_limit[self.combined_joint_indices] - (-damping_torque)
        available_torque_neg = 0.9 * self.effort_limit[self.combined_joint_indices] - (damping_torque)

        # Clamp to nonnegative values
        available_torque_pos = torch.clamp(available_torque_pos, min=0.0)
        available_torque_neg = torch.clamp(available_torque_neg, min=0.0)

        # Compute max error for positive and negative torque directions
        max_pos_error = available_torque_pos / self.policy_stiffness
        max_neg_error = available_torque_neg / self.policy_stiffness

        # Clip accordingly
        clipped_pos_error = torch.max(torch.min(pos_error, max_pos_error), -max_neg_error)

        # Return clipped joint positions
        return current_joint_pos + clipped_pos_error

    def pub_ee_pose(self):
        """
        Publish the ee pose as a sphere marker in RViz.
        """
        try:
            # Check if publisher is initialized
            if not hasattr(self, "ee_marker_pub"):
                return

            # Create marker array for ee pose
            marker_array = MarkerArray()
            ee_pose_w = self.robot.mj_model.get_ee_positions_w()
            for i in range(2):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "ee_pose"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                # Set position from ee pose
                marker.pose.position.x = float(ee_pose_w[i, 0])
                marker.pose.position.y = float(ee_pose_w[i, 1])
                marker.pose.position.z = float(ee_pose_w[i, 2])

                # Set orientation (identity quaternion)
                marker.pose.orientation.w = 1.0
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0

                # Set scale (sphere radius)
                marker.scale.x = 0.06
                marker.scale.y = 0.06
                marker.scale.z = 0.06

                # Set color (red for ee pose)
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

                marker_array.markers.append(marker)

            # Publish marker array
            self.ee_marker_pub.publish(marker_array)

        except Exception as e:
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.error(f"Failed to publish ee pose: {e}")

    def pub_all_visualizations(self):
        """
        Publish all visualization data to RViz.
        """
        try:
            if hasattr(self, "visualize"):
                if self.visualize.get("object_pose", False):
                    self.pub_object_pose()
                if self.visualize.get("contact_positions", False):
                    self.pub_contact_positions()
                if self.visualize.get("desired_object_pose", False):
                    self.pub_desired_object_pose()
                if self.visualize.get("ee_pose", False):
                    self.pub_ee_pose()

        except Exception as e:
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.error(f"Failed to publish visualizations: {e}")

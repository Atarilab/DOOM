import os
import threading
import time
from typing import Any, Dict
from queue import Queue

import numpy as np
import torch
from commands.command_manager import CommandTerm
from controllers.controller_base import ControllerBase
from state_manager.obs_manager import ObsTerm
from state_manager.observations import (
    ang_vel_b,
    joint_pos_rel,
    joint_vel,
    last_action,
    lin_vel_b,
    projected_gravity_b,
    velocity_commands,
)
from utils.helpers import ObservationHistoryStorage
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped


class BaseRLLocomotionController(ControllerBase, Node):
    """
    Base Reinforcement Learning Locomotion Controller

    Provides core infrastructure for neural network-based robot locomotion control
    with concurrent observation processing and policy inference.
    """

    def __init__(self, mj_model_wrapper: "MjQuadRobotWrapper", configs: Dict[str, Any]):
        """
        Initialize the RL locomotion controller with model and configuration.

        :param mj_model_wrapper: Mujoco model wrapper for kinematics
        :param configs: Configuration dictionary
        """
        # Initialize ROS2 node
        Node.__init__(self, 'rl_locomotion_controller')
        # Initialize controller base
        ControllerBase.__init__(self, mj_model_wrapper=mj_model_wrapper, configs=configs)

        # Create publisher for future feet positions
        self.feet_pos_pub = self.create_publisher(MarkerArray, 'future_feet_positions', 10)
        
        # Create static transform publisher for map frame
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        
        # Create and publish static transform from map to world
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'world'
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        transform.transform.rotation.w = 1.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        self.tf_broadcaster.sendTransform(transform)
        
        self.active = False
        # Load and prepare policy model
        self._load_policy_model(configs)

        # Initialize controller-specific parameters
        self._initialize_controller_parameters(configs)

        # Set up observation and action processing
        self._configure_processing_infrastructure(configs)

    def _load_policy_model(self, configs: Dict[str, Any]):
        """
        Load and prepare the neural network policy model.

        :param configs: Configuration dictionary
        """
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            configs["controller_config"]["policy_path"],
        )

        # Use TorchScript for optimized inference
        self.policy = torch.jit.load(model_path).to("cpu")
        self.policy.eval()

        # Precompute static configurations
        self.action_scale = torch.tensor(configs["controller_config"]["action_scale"], dtype=torch.float32)

    def _initialize_controller_parameters(self, configs: Dict[str, Any]):
        """
        Initialize controller-specific parameters.

        :param configs: Configuration dictionary
        """
        controller_config = configs["controller_config"]

        self.default_joint_pos = torch.tensor(controller_config["ISAAC_LAB_DEFAULT_JOINT_POS"], dtype=torch.float32)

        self.actions_isaac_to_unitree_mapping = np.array(controller_config["JOINT_ACTION_ISAAC_LAB_TO_UNITREE_MAPPING"])

        self.joint_obs_unitree_to_isaac_mapping = torch.tensor(
            controller_config["JOINT_OBSERVATION_UNITREE_TO_ISAAC_LAB_MAPPING"]
        )

        # Controller gains and configuration
        self.Kp = controller_config["stiffness"]
        self.Kd = controller_config["damping"]
        self.action_dim = controller_config["action_dim"]
        self.control_dt = controller_config["control_dt"]
        self.decimation = controller_config["decimation"]

        # Filter coefficient (0 < alpha < 1), lower values = more smoothing
        self.action_filter_alpha = controller_config["action_filter_alpha"] if "action_filter_alpha" in controller_config else 1.0
        self.filtered_action = torch.zeros(self.action_dim, dtype=torch.float32)
        self.is_first_action = True

        # Initial state and commands
        self.latest_state = None
        self.cmd = {}

    def _configure_processing_infrastructure(self, configs: Dict[str, Any]):
        """
        Set up concurrent processing infrastructure for observations and policy.

        :param configs: Configuration dictionary
        """
        # Performance optimization: Preallocate tensors
        action_dim = configs["controller_config"]["action_dim"]
        obs_dim = configs["controller_config"]["obs_dim"]
        self.raw_action = torch.zeros(action_dim, dtype=torch.float32, device="cpu")

        # Observation history storage
        self.obs_buffer = ObservationHistoryStorage(num_envs=1, num_obs=obs_dim, max_length=1, device="cpu")

        # Start concurrent processing threads
        self._init_processing_threads()

    def _init_processing_threads(self):
        """Initialize and start concurrent processing threads."""
        self.obs_processing_thread = threading.Thread(target=self._process_observations, daemon=True)
        self.policy_inference_thread = threading.Thread(target=self._run_policy_inference, daemon=True)
        self.obs_processing_thread.start()
        self.policy_inference_thread.start()

    def _process_observations(self):
        """Continuously process observations in a separate thread"""
        while True:
            try:
                with self._lock:
                    current_state = self.latest_state

                if current_state is None or not self.active:
                    time.sleep(0.01)  # Prevent busy waiting
                    continue

                # Compute and store observations
                with torch.no_grad():
                    try:
                        obs = self.obs_manager.compute(current_state)
                        obs_tensor = torch.cat([v.reshape(-1) for v in obs.values()])
                        self.obs_buffer.add(obs_tensor.unsqueeze(0))
                    except Exception as e:
                        print(f"Error converting observations to tensor: {e}")
                        print("Observations that caused the error:")
                        for obs_name, obs_value in obs.items():
                            print(f"{obs_name}: {type(obs_value)}")
                            if isinstance(obs_value, (np.ndarray, torch.Tensor)):
                                print(f"  Shape: {obs_value.shape}")
                            else:
                                print(f"  Value: {obs_value}")
                        time.sleep(0.1)  # Prevent rapid error loops
                        continue

            except Exception as e:
                print(f"Observation processing error: {e}")
                print(f"Error type: {type(e)}")
                import traceback

                print(f"Traceback: {traceback.format_exc()}")
                time.sleep(0.1)  # Prevent rapid error loops

    def _run_policy_inference(self):
        """Continuously run policy inference in a separate thread at a fixed dt of 0.02 seconds"""
        dt = self.control_dt * self.decimation  # Fixed time step in seconds (0.02)
        last_time = time.time()
        
        while True:
            try:
                if not self.active:
                    time.sleep(0.01)
                    continue

                # Calculate time since last iteration
                current_time = time.time()
                elapsed = current_time - last_time
                
                # Sleep if we're ahead of schedule
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    current_time = time.time()  # Update current time after sleep
                
                # Update last time for next iteration
                last_time = current_time

                try:
                    obs = self.obs_buffer.get()

                    # Policy inference
                    with torch.no_grad():
                        raw_action = self.policy(obs.unsqueeze(0))

                    self.raw_action.copy_(raw_action[0][0])
                except Exception as e:
                    print(f"Policy inference error: {e}")
                    time.sleep(0.1)  # Prevent rapid error loops
                    continue

            except Exception as e:
                print(f"Policy inference thread error: {e}")
                time.sleep(0.1)  # Prevent rapid error loops

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
            # Ensure we have processed observations and have a valid action
            if self.obs_buffer.get().numel() == 0 or self.obs_buffer.get().sum() == 0:
                # If no observations yet, use default joint positions
                joint_pos_targets = self.default_joint_pos.cpu().numpy()[self.actions_isaac_to_unitree_mapping]
            else:
                # Apply exponential moving average filter to smooth actions
                if self.is_first_action:
                    self.filtered_action = self.raw_action
                    self.is_first_action = False
                else:
                    # EMA filter: filtered = alpha * new + (1 - alpha) * previous
                    self.filtered_action = (
                        self.action_filter_alpha * self.raw_action 
                        + (1 - self.action_filter_alpha) * self.filtered_action
                    )

                # Compute joint position targets from the filtered policy output
                joint_pos_targets = (
                    (self.filtered_action * self.action_scale + self.default_joint_pos)
                    .cpu()
                    .numpy()[self.actions_isaac_to_unitree_mapping]
                )

            # Clip the joint pos targets for safety
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
                for i in range(12)
            }

            # Track command preparation time
            self.cmd_preparation_time = time.perf_counter() - start_time

            return self.cmd

        except Exception as e:
            print(f"Command preparation error: {e}")
            return self.cmd

    def set_mode(self):
        self.active = True


class RLLocomotionVelocityController(BaseRLLocomotionController):
    """
    Velocity-conditioned RL Locomotion Controller
    Uses contact-implicit reinforcement learning policy
    """

    def __init__(self, mj_model_wrapper: "MjQuadRobotWrapper", configs: Dict[str, Any]):
        super().__init__(mj_model_wrapper=mj_model_wrapper, configs=configs)

        # Default velocity commands
        self.velocity_commands = torch.tensor([0.0, 0.0, 0.0])

    def change_commands(self, new_commands: Dict[str, Any]):
        """
        Change velocity commands with validation.

        :param new_commands: Dictionary of new command values
        """

        try:
            self.velocity_commands = self.command_manager.validate_and_change_commands(
                self.velocity_commands, new_commands
            )
            self.command_manager.logger.debug(f"Command Updated: {new_commands}")
        except ValueError as e:
            # Log error or handle validation failure
            if self.command_manager.logger:
                self.command_manager.logger.error(f"Command update failed: {e}")

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

    def set_mode(self):
        """Runs when the mode is changed in the UI."""
        super().set_mode()


class RLLocomotionContactController(BaseRLLocomotionController):
    """
    Contact-conditioned RL Locomotion Controller
    Uses contact-explicit reinforcement learning policy
    """

    def __init__(self, mj_model_wrapper: "MjQuadRobotWrapper", configs: Dict[str, Any]):
        self.init_completed = False
        super().__init__(mj_model_wrapper=mj_model_wrapper, configs=configs)

        # Contact command parameters
        self.command_duration = 0.35  # Duration of each contact plan in seconds
        self.current_contact_plan = torch.ones((2, 4), dtype=torch.bool)  # Current contact pattern
        self.command_start_time = time.time()  # When the current plan started
        self.visualize_future_feet_positions = configs["controller_config"]["visualize"]["future_feet_positions"]
        self.visualize_current_contact_locations = configs["controller_config"]["visualize"]["current_contact_locations"]

        # Thread synchronization for gait changes
        self._gait_lock = threading.RLock()  # Reentrant lock for gait changes
        self._gait_change_event = threading.Event()  # Event to signal gait changes

        # Horizon planning
        self.future_feet_positions_init_frame = None
        self.horizon_length = configs["controller_config"]["horizon_length"]
        self.feet_step_size = configs["controller_config"]["feet_step_size"]
        self.future_feet_positions_init_frame = None
        self.future_feet_positions_w = torch.zeros(4, self.horizon_length, 3)
        self.future_feet_positions_b = torch.zeros(4, self.horizon_length, 3)

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
            "pace": torch.tensor(
                [
                    [True, False, True, False],  # Phase 1: FL and RL in contact
                    [False, True, False, True],  # Phase 2: FR and RR in contact
                ]
            ),
            "bound": torch.tensor(
                [
                    [True, True, False, False],  # Phase 1: Front legs in contact
                    [False, False, True, True],  # Phase 2: Back legs in contact
                ]
            ),
            "jump": torch.tensor(
                [
                    [True, True, True, True],  # Phase 1: All legs in contact
                    [False, False, False, False],  # Phase 2: All legs in flight
                ]
            ),
        }
        self.current_gait = "transition"  # Default gait
        self.pending_gait_change = None  # Store pending gait change
        self.in_transition = False  # Flag to indicate if we're in a transition phase
        self.transition_counter = 0  # Counter to track resample steps during transition
        self.transition_duration = 2  # Number of resample steps for transition (increased from 2)
        self.transition_progress = 0.0
        self.transition_start_gait = None  # Starting gait for transition
        self.transition_end_gait = None  # Ending gait for transition
        self.time_left = self.command_duration
        self.current_contact_plan = self.gait_patterns[self.current_gait]
        self.current_goal_idx = 0
        self.goal_completion_counter = 0

        self.init_completed = True
        self.active = True

    def compute_torques(self, state, desired_goal):
        """
        Compute motor commands using the learned policy.

        :param state: Current robot state
        :param desired_goal: Desired goal state (not used in this implementation)
        :return: Motor commands dictionary
        """
        if self.mj_model_wrapper is not None:
            self.mj_model_wrapper.update(state)

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

        if self.visualize_current_contact_locations:
            self.viz_current_contact_locations()

        # Use the base class implementation to compute torques
        # This will use the pre-computed raw_action from the policy inference thread
        return super().compute_torques(state, desired_goal)

    def _resample_commands(self):
        """Resample the commands for the controller with thread safety."""
        try:
            with self._gait_lock:
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
                        if self.pending_gait_change in ["trot", "pace", "bound", "jump"]:
                            self.current_gait = "transition"
                            self.current_contact_plan = self.gait_patterns["transition"]
                            
                        if (
                            hasattr(self, "command_manager")
                            and self.command_manager
                            and hasattr(self.command_manager, "logger")
                        ):
                            self.command_manager.logger.debug(
                                f"Starting transition phase from {self.transition_start_gait} to {self.transition_end_gait}"
                            )
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
                                self.generate_future_feet_positions()
                                
                            self.pending_gait_change = None
                            self.transition_start_gait = None
                            self.transition_end_gait = None
                            
                            # Signal that the gait change is complete
                            self._gait_change_event.set()
                            
                            self.command_manager.logger.debug(f"Transition complete, applied gait: {self.current_gait}")
                        else:
                            # Log transition progress for debugging
                            if (self.transition_counter % 2 == 0 ):
                                self.command_manager.logger.debug(
                                    f"Transition progress: {self.transition_progress:.2f}"
                                )

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
        except Exception as e:
            if hasattr(self, "command_manager") and self.command_manager and hasattr(self.command_manager, "logger"):
                self.command_manager.logger.error(f"Error resampling commands: {e}")
            else:
                print(f"Error resampling commands: {e}")

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

    def register_observations(self):
        """
        Register observations for contact-conditioned locomotion.
        Includes contact pattern and timing information.
        """
        from state_manager.observations import (
            # base_height,
            contact_locations_b,
            contact_plan,
            # contact_status,
            contact_time_left,
            ee_pos_rel_b,
        )

        self.obs_manager.register("lin_vel_b", ObsTerm(lin_vel_b))
        self.obs_manager.register("ang_vel_b", ObsTerm(ang_vel_b))
        # self.obs_manager.register(
        #     "base_height", ObsTerm(base_height, params={"mj_model_wrapper": self.mj_model_wrapper})
        # )
        self.obs_manager.register("projected_gravity", ObsTerm(projected_gravity_b))
        # self.obs_manager.register("contact_status", ObsTerm(contact_status))
        # self.obs_manager.register(
        #     "contact_locations",
        #     ObsTerm(
        #         contact_locations,
        #         params={
        #             "mj_model_wrapper": self.mj_model_wrapper,
        #             "future_feet_positions_init_frame": lambda: self.future_feet_positions_init_frame,
        #             "obs_horizon": 2,
        #             "current_goal_idx": lambda: self.current_goal_idx
        #         }
        #     )
        # )
        self.obs_manager.register(
            "contact_locations",
            ObsTerm(
                contact_locations_b,
                params={
                    "mj_model_wrapper": self.mj_model_wrapper,
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
                params={"mj_model_wrapper": self.mj_model_wrapper,
                        "future_feet_positions_w": lambda: self.future_feet_positions_w,
                        "current_goal_idx": lambda: self.current_goal_idx
                }
            )
        )
        self.obs_manager.register(
            "last_action",
            ObsTerm(last_action, params={"last_action": lambda: self.raw_action}),
        )

    def set_mode(self):
        """Runs when the mode is changed in the UI.
        Generates the future feet positions in the init frame for horizon planning.
        """
        # Call the base class set_mode to activate the controller
        # super().set_mode()

        self.generate_future_feet_positions()
        if self.visualize_future_feet_positions:
            self.viz_future_feet_positions()

    def generate_future_feet_positions(self):
        """
        Generates future feet positions for the controller.
        """
        
        self.goal_completion_counter = 0
        self.current_goal_idx = 0
        offset = torch.tensor(
            [(0.2334, 0.1865, 0.0), (0.2334, -0.1865, 0.0), (-0.2334, 0.1865, 0.0), (-0.2334, -0.1865, 0.0)],
            dtype=torch.float32
        )
        #
        base_pos_w = torch.tensor(self.mj_model_wrapper.get_body_position_world("base_link"), dtype=torch.float32)
        
        # Get robot yaw from base orientation matrix
        base_rot = torch.tensor(self.mj_model_wrapper.get_body_orientation_world("base_link"), dtype=torch.float32)
        yaw = torch.atan2(base_rot[1, 0], base_rot[0, 0])
        
        # Create rotation matrix for yaw
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rotation_matrix = torch.tensor([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # Rotate the offset positions
        rotated_offset = torch.matmul(rotation_matrix, offset.T).T
        
        # Apply the rotated offset to base position
        self.future_feet_positions_w[:] = (base_pos_w + rotated_offset).unsqueeze(1)
        
        stride_offsets = torch.arange(self.horizon_length, dtype=torch.float32).unsqueeze(0) * torch.tensor(self.feet_step_size, dtype=torch.float32).unsqueeze(
            -1
        )
        
        # Apply stride in the direction of robot's orientation
        direction_x = cos_yaw
        direction_y = sin_yaw
        self.future_feet_positions_w[:, :, 0] += stride_offsets.squeeze(-1) * direction_x
        self.future_feet_positions_w[:, :, 1] += stride_offsets.squeeze(-1) * direction_y
        self.future_feet_positions_w[:, :, 2] = 0.05
        
        self.future_feet_positions_init_frame = self.mj_model_wrapper.transform_world_to_init_frame(self.future_feet_positions_w.numpy())
            
                
    def viz_future_feet_positions(self):
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
            
            foot_names = ['FL', 'FR', 'RL', 'RR']
            
            # Transpose the tensor to match the expected shape (num_steps, 4, 3)
            positions_for_viz = self.future_feet_positions_w.permute(1, 0, 2).numpy()
            
            for foot_idx in range(len(foot_names)):
                # Create a marker for the trajectory of each foot
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f"foot_trajectory_{foot_names[foot_idx]}"
                marker.id = foot_idx
                marker.type = Marker.POINTS
                marker.action = Marker.ADD
                
                # Set the scale of the points
                marker.scale.x = 0.02  # Point size
                marker.scale.y = 0.02
                marker.scale.z = 0.02
                
                # Set the color
                marker.color = colors[foot_idx]
                
                # Add all future positions for this foot
                for step in range(positions_for_viz.shape[0]):
                    point = Point()
                    point.x = float(positions_for_viz[step, foot_idx, 0])
                    point.y = float(positions_for_viz[step, foot_idx, 1])
                    point.z = float(positions_for_viz[step, foot_idx, 2])
                    marker.points.append(point)
                
                marker_array.markers.append(marker)
            
            # Publish the marker array
            self.feet_pos_pub.publish(marker_array)
            self.command_manager.logger.debug('Published future feet positions marker array')
            
        except Exception as e:
            self.command_manager.logger.error(f'Failed to publish future feet positions: {e}')

    def viz_current_contact_locations(self):
        """
        Visualizes the current contact locations in the world frame.
        Colors the markers based on the current contact plan:
        - Green: for feet that are not in contact (contact_plan is False)
        - Black: for feet that are in contact (contact_plan is True)
        """
        try:
            marker_array = MarkerArray()
            
            # Get current contact locations from future feet positions
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
                
                # Set position
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
            self.feet_pos_pub.publish(marker_array)
            
        except Exception as e:
            self.command_manager.logger.error(f'Failed to publish current contact locations: {e}')


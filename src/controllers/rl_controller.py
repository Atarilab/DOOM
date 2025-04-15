import os
import threading
import time
from typing import Any, Dict

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


class BaseRLLocomotionController(ControllerBase):
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
        super().__init__(mj_model_wrapper=mj_model_wrapper, configs=configs)

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
        """Continuously run policy inference in a separate thread"""
        while True:
            try:
                if not self.active:
                    time.sleep(0.01)
                    continue

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
                # Compute joint position targets from the policy output
                joint_pos_targets = (
                    (self.raw_action * self.action_scale + self.default_joint_pos)
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

        # Horizon planning
        self.future_feet_positions_init_frame = None
        self.horizon_length = 1000
        self.feet_step_size = 0.2  # meters
        self.future_feet_positions_init_frame = None
        self.future_feet_positions_w = torch.zeros(4, self.horizon_length, 3)
        self.future_feet_positions_b = torch.zeros(4, self.horizon_length, 3)

        # Define gait patterns (FL, FR, RL, RR)
        self.gait_patterns = {
            "stance": torch.tensor(
                [
                    [True, True, True, True],  # Phase 1: FL and RR in contact
                    [True, True, True, True],  # Phase 2: FR and RL in contact
                ]
            ),
            "trot": torch.tensor(
                [
                    [True, False, False, True],  # Phase 1: FL and RR in contact
                    [False, True, True, False],  # Phase 2: FR and RL in contact
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
        self.current_gait = "stance"  # Default gait
        self.pending_gait_change = None  # Store pending gait change
        self.in_transition = False  # Flag to indicate if we're in a transition phase
        self.time_left = self.command_duration
        self.current_contact_plan = self.gait_patterns[self.current_gait]
        self.current_goal_idx = 0
        self.goal_completion_counter = 0

        self.init_completed = True
        self.active = False

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

        # Update contact plan timing
        current_time = time.time()
        elapsed = current_time - self.command_start_time
        self.time_left = max(0, self.command_duration - elapsed)
        if elapsed >= self.command_duration:
            self.command_start_time = current_time
            self._resample_commands()

        # Use the base class implementation to compute torques
        # This will use the pre-computed raw_action from the policy inference thread
        return super().compute_torques(state, desired_goal)

    def _update_commands(self):
        """Update the commands for the controller."""
        # This method is called within the _process_commands thread with the lock already held
        # It can be extended to update any command-related variables that need to be updated
        # For now, we'll just log that it's being called for debugging purposes
        pass

    def _resample_commands(self):
        """Resample the commands for the controller."""
        try:
            # Handle gait transitions
            if self.pending_gait_change is not None:
                with self._lock:
                    if not self.in_transition:
                        # Start transition phase - switch to stance
                        self.in_transition = True
                        self.current_gait = "stance"
                        self.current_contact_plan = self.gait_patterns["stance"]
                        if (
                            hasattr(self, "command_manager")
                            and self.command_manager
                            and hasattr(self.command_manager, "logger")
                        ):
                            self.command_manager.logger.debug(
                                f"Starting transition phase to: {self.pending_gait_change}"
                            )
                    else:
                        # Transition phase complete - apply the pending gait
                        self.in_transition = False
                        self.current_gait = self.pending_gait_change
                        self.current_contact_plan = self.gait_patterns[self.pending_gait_change]
                        self.pending_gait_change = None
                        if (
                            hasattr(self, "command_manager")
                            and self.command_manager
                            and hasattr(self.command_manager, "logger")
                        ):
                            self.command_manager.logger.debug(f"Transition complete, applied gait: {self.current_gait}")

            # Normal gait progression (only if not in transition phase)
            if not self.in_transition and self.current_gait != "stance":
                self.goal_completion_counter += 1
                self.current_goal_idx += (
                    1 if self.goal_completion_counter % 2 == 0 and self.goal_completion_counter > 0 else 0
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
        """
        Change contact commands with validation.

        :param new_commands: Dictionary containing contact pattern and timing information
        """

        try:
            if "gait" in new_commands:
                new_gait = new_commands["gait"].lower()
                if new_gait in self.gait_patterns:
                    # Store the requested gait change instead of applying it immediately
                    with self._lock:
                        # Only set pending change if it's different from current gait
                        if new_gait != self.current_gait:
                            self.pending_gait_change = new_gait
                            if (
                                hasattr(self, "command_manager")
                                and self.command_manager
                                and hasattr(self.command_manager, "logger")
                            ):
                                self.command_manager.logger.debug(f"Stored pending gait change to: {new_gait}")
                else:
                    raise ValueError(f"Invalid gait: {new_gait}. Must be one of {list(self.gait_patterns.keys())}")

            if hasattr(self, "command_manager") and self.command_manager and hasattr(self.command_manager, "logger"):
                self.command_manager.logger.debug(f"Contact Command Updated: {new_commands}")
        except ValueError as e:
            if hasattr(self, "command_manager") and self.command_manager and hasattr(self.command_manager, "logger"):
                self.command_manager.logger.error(f"Contact command update failed: {e}")
            else:
                print(f"Contact command update failed: {e}")

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
            base_height,
            contact_locations,
            contact_locations_b,
            contact_plan,
            contact_status,
            contact_time_left,
            ee_pos_rel,
        )

        self.obs_manager.register("lin_vel_b", ObsTerm(lin_vel_b))
        self.obs_manager.register("ang_vel_b", ObsTerm(ang_vel_b))
        # self.obs_manager.register(
        #     "base_height", ObsTerm(base_height, params={"mj_model_wrapper": self.mj_model_wrapper})
        # )
        self.obs_manager.register("projected_gravity", ObsTerm(projected_gravity_b))
        self.obs_manager.register("contact_status", ObsTerm(contact_status))
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
        # self.obs_manager.register(
        #    "ee_pos_rel",
        #     ObsTerm(
        #         ee_pos_rel,
        #         params={"mj_model_wrapper": self.mj_model_wrapper,
        #                 "future_feet_positions_init_frame": lambda: self.future_feet_positions_init_frame,
        #                 "current_goal_idx": lambda: self.current_goal_idx
        #         }
        #     )
        # )
        # self.obs_manager.register(
        #    "ee_pos_rel_b",
        #     ObsTerm(
        #         ee_pos_rel_b,
        #         params={"mj_model_wrapper": self.mj_model_wrapper,
        #                 "future_feet_positions_w": lambda: self.future_feet_positions_w,
        #                 "current_goal_idx": lambda: self.current_goal_idx
        #         }
        #     )
        # )
        self.obs_manager.register(
            "last_action",
            ObsTerm(last_action, params={"last_action": lambda: self.raw_action}),
        )

    def set_mode(self):
        """Runs when the mode is changed in the UI.
        Generates the future feet positions in the init frame for horizon planning.
        """
        # Call the base class set_mode to activate the controller
        super().set_mode()

        # # Initialize the processing threads if they haven't been started yet
        # if not hasattr(self, 'obs_processing_thread') or not self.obs_processing_thread.is_alive():
        #     self._init_processing_threads()

        # compute the feet positions in the init frame
        # if self.mj_model_wrapper.initial_feet_positions_init_frame is None:
        #     raise RuntimeError("Initial feet positions not set. Call set_initial_world_frame() first.")

        # self.feet_pos_init_frame = self.mj_model_wrapper.get_feet_positions_init_frame() +

        # # Get current feet positions (already a numpy array)
        # current_feet_pos = self.feet_pos_init_frame

        # # Create offset array for x coordinates: shape (horizon_length,)
        # x_offsets = np.arange(self.horizon_length, dtype=np.float32) * self.feet_step_size

        # # Expand current feet positions and x_offsets for broadcasting
        # # current_feet_pos: (4,3), x_offsets: (horizon_length,)
        # expanded_feet_pos = current_feet_pos[:, np.newaxis, :]  # Shape: (4,1,3)
        # expanded_offsets = x_offsets[np.newaxis, :, np.newaxis]  # Shape: (1,horizon_length,1)

        # # Create the horizon positions using broadcasting
        # self.future_feet_positions_init_frame = np.tile(expanded_feet_pos, (1, self.horizon_length, 1))  # Shape: (4,horizon_length,3)
        # self.future_feet_positions_init_frame[:,:,0] += expanded_offsets.squeeze(-1)  # Add offsets to x coordinates only

        offset = torch.tensor(
            [(0.2334, 0.1865, 0.0), (0.2334, -0.1865, 0.0), (-0.2334, 0.1865, 0.0), (-0.2334, -0.1865, 0.0)]
        )
        #
        base_pos_w = torch.tensor(self.mj_model_wrapper.get_body_position_world("base_link"))
        self.future_feet_positions_w[:] = (base_pos_w + offset).unsqueeze(1)
        stride_offsets = torch.arange(self.horizon_length).unsqueeze(0) * torch.tensor(self.feet_step_size).unsqueeze(
            -1
        )
        # get robot yaw
        # Get robot yaw from base orientation matrix
        base_rot = torch.tensor(self.mj_model_wrapper.get_body_orientation_world("base_link"))
        yaw = torch.atan2(base_rot[1, 0], base_rot[0, 0])
        # yaw = torch.tensor([0.0])
        direction_x = torch.cos(yaw)
        direction_y = torch.sin(yaw)
        self.future_feet_positions_w[:, :, 0] += stride_offsets.squeeze(-1) * direction_x
        self.future_feet_positions_w[:, :, 1] += stride_offsets.squeeze(-1) * direction_y
        self.future_feet_positions_w[:, :, 2] = 0.2
        # self.future_feet_positions_w[:,:,2] = torch.tensor(self.mj_model_wrapper.get_feet_positions_world()[:, 2]).unsqueeze(-1)
        # self.future_feet_positions_w[:, :, 2] =
        # self.future_feet_positions_b = self.mj_model_wrapper.transform_world_to_base(self.future_feet_positions_w)
        # self.future_feet_positions_init_frame = base_pos_w + offset

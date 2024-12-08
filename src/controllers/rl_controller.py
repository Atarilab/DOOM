import os
import time
import torch
import threading
import numpy as np
from itertools import chain
from typing import Dict, Any


from controllers.controller_base import ControllerBase
from utils.helpers import ObservationHistoryStorage
from state_manager.obs_manager import ObsTerm
from state_manager.observations import (
    joint_pos_rel,
    joint_vel,
    lin_vel_b,
    ang_vel_b,
    last_action,
    projected_gravity_b,
    velocity_commands,
    feet_pos,
)


class BaseRLLocomotionController(ControllerBase):
    """
    Base Reinforcement Learning Locomotion Controller

    Provides core infrastructure for neural network-based robot locomotion control
    with concurrent observation processing and policy inference.
    """

    def __init__(self, pin_model_wrapper, command_manager, configs: Dict[str, Any]):
        """
        Initialize the RL locomotion controller with model and configuration.

        :param pin_model_wrapper: Pinocchio model wrapper for kinematics
        :param configs: Configuration dictionary
        """
        super().__init__(pin_model_wrapper=pin_model_wrapper, 
                         command_manager=command_manager, 
                         configs=configs)

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
        self.action_scale = torch.tensor(
            configs["controller_config"]["action_scale"], dtype=torch.float32
        )

    def _initialize_controller_parameters(self, configs: Dict[str, Any]):
        """
        Initialize controller-specific parameters.

        :param configs: Configuration dictionary
        """
        controller_config = configs["controller_config"]

        self.default_joint_pos = torch.tensor(
            controller_config["ISAAC_LAB_DEFAULT_JOINT_POS"], dtype=torch.float32
        )

        self.actions_isaac_to_unitree_mapping = np.array(
            controller_config["JOINT_ACTION_ISAAC_LAB_TO_UNITREE_MAPPING"]
        )

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
        self.raw_action = torch.zeros(action_dim, dtype=torch.float32, device="cpu")

        # Observation history storage
        self.obs_buffer = ObservationHistoryStorage(
            num_envs=1, num_obs=48, max_length=1, device="cpu"
        )

        # Start concurrent processing threads
        self._init_processing_threads()

    def _init_processing_threads(self):
        """Initialize and start concurrent processing threads."""
        self.obs_processing_thread = threading.Thread(
            target=self._process_observations, daemon=True
        )
        self.policy_inference_thread = threading.Thread(
            target=self._run_policy_inference, daemon=True
        )

        self.obs_processing_thread.start()
        self.policy_inference_thread.start()

    def _process_observations(self):
        """Continuously process observations in a separate thread"""
        while True:
            try:
                with self._lock:
                    current_state = self.latest_state

                if current_state is None:
                    time.sleep(0.01)  # Prevent busy waiting
                    continue

                # Prepare Pinocchio model state
                q_pin = np.concatenate(
                    [
                        current_state["base_pos_w"],
                        current_state["base_quat"],
                        current_state["joint_pos"][
                            np.array(self.unitree_pin_joint_mappings)
                        ],
                    ]
                )
                v_pin = current_state["joint_vel"][
                    np.array(self.unitree_pin_joint_mappings)
                ]

                self.pin_model_wrapper.update(q_pin, v_pin)

                # Compute and store observations
                with torch.no_grad():
                    obs = self.obs_manager.compute_observations(current_state)
                    obs_tensor = torch.tensor(
                        list(chain.from_iterable(obs.values())),
                        dtype=torch.float32,
                        device="cpu",
                    )
                    self.obs_buffer.add(obs_tensor.unsqueeze(0))

            except Exception as e:
                print(f"Observation processing error: {e}")
                time.sleep(0.1)  # Prevent rapid error loops

    def _run_policy_inference(self):
        """Continuously run policy inference in a separate thread"""
        while True:
            try:
                obs = self.obs_buffer.get()

                # Policy inference
                with torch.no_grad():
                    raw_action = self.policy(obs.unsqueeze(0))

                self.raw_action.copy_(raw_action[0][0])

            except Exception as e:
                print(f"Policy inference error: {e}")
                time.sleep(0.1)  # Prevent rapid error loops

    def compute_torques(self, state, desired_goal):
        """
        Compute motor commands using the learned policy.

        :param state: Current robot state
        :param desired_goal: Desired goal state (not used in this implementation)
        :return: Motor commands dictionary
        """
        start_time = time.perf_counter()

        try:
            # Ensure we have processed observations
            if self.obs_buffer.get().numel() == 0:
                self.raw_action.zero_()

            # Compute joint position targets
            joint_pos_targets = (
                (self.raw_action * self.action_scale + self.default_joint_pos)
                .cpu()
                .numpy()[self.actions_isaac_to_unitree_mapping]
            )

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


class RLLocomotionVelocityController(BaseRLLocomotionController):
    """
    Velocity-conditioned RL Locomotion Controller
    Uses contact-implicit reinforcement learning policy
    """
    
    def __init__(self, pin_model_wrapper, configs: Dict[str, Any], command_manager):
        super().__init__(pin_model_wrapper=pin_model_wrapper, 
                         command_manager=command_manager, 
                         configs=configs)
        
        # Default velocity commands
        self.velocity_commands = np.array([0.0, 0.0, 0.0])
        
        # command manager
        self.command_manager = command_manager
        
    def update_commands(self, new_commands: Dict[str, Any]):
        """
        Update velocity commands with validation.
        
        :param new_commands: Dictionary of new command values
        """
        print("HI")
        print(self.velocity_commands)
        if self.command_manager:
            try:
                self.velocity_commands = self.command_manager.validate_and_update_commands(
                    "RLLocomotionVelocityController", 
                    self.velocity_commands, 
                    new_commands
                )
            except ValueError as e:
                # Log error or handle validation failure
                if self.obs_manager.logger:
                    self.obs_manager.logger.error(f"Command update failed: {e}")
        print(self.velocity_commands)

    def register_observations(self):
        """
        Register observations for velocity-conditioned locomotion.
        Maintains a specific order for direct policy input.
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
            ObsTerm(
                joint_vel, params={"mapping": self.joint_obs_unitree_to_isaac_mapping}
            ),
        )
        self.obs_manager.register(
            "last_action",
            ObsTerm(last_action, params={"last_action": lambda: self.raw_action}),
        )
        self.obs_manager.register(
            "feet_pos",
            ObsTerm(
                feet_pos,
                params={"pin_model_wrapper": self.pin_model_wrapper},
                include=False,
            ),
        )


class RLLocomotionContactController(BaseRLLocomotionController):
    """
    Contact-conditioned RL Locomotion Controller
    Uses contact-explicit reinforcement learning policy
    """

    def register_observations(self):
        """
        Register observations for contact-conditioned locomotion.
        Maintains a specific order for direct policy input.
        """
        # Observation registration identical to velocity controller
        # (You might want to add contact-specific observations in the future)
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
            ObsTerm(
                joint_vel, params={"mapping": self.joint_obs_unitree_to_isaac_mapping}
            ),
        )
        self.obs_manager.register(
            "last_action",
            ObsTerm(last_action, params={"last_action": lambda: self.raw_action}),
        )

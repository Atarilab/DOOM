import os
import time
import torch
import threading
import traceback
import numpy as np
from typing import Any, Dict

from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

from utils.helpers import ObservationHistoryStorage
from controllers.controller_base import ControllerBase


class RLControllerBase(ControllerBase, Node):
    """
    Base Reinforcement Learning Locomotion Controller

    Provides core infrastructure for neural network-based robot locomotion control
    with concurrent observation processing and policy inference.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any]):
        """
        Initialize the RL locomotion controller with model and configuration.

        :param robot: Robot model
        :param configs: Configuration dictionary
        """
        # Initialize ROS2 node
        Node.__init__(self, "rl_locomotion_controller")
        # Initialize controller base
        ControllerBase.__init__(self, robot=robot, configs=configs)

        # Create static transform publisher for map frame
        self.tf_broadcaster = StaticTransformBroadcaster(self)

        # Create and publish static transform from map to world
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "map"
        transform.child_frame_id = "world"
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        transform.transform.rotation.w = 1.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        self.tf_broadcaster.sendTransform(transform)

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
            "policies",
            configs["controller_config"]["policy_path"],
        )

        # Use TorchScript for optimized inference
        self.policy = torch.jit.load(model_path).to("cpu")
        self.policy.eval()

        # Precompute static configurations
        self.action_scale = torch.tensor(
            configs["controller_config"]["action_scale"], dtype=torch.float32
        )
        self.policy_architecture = configs["controller_config"].get("policy_architecture", "mlp")

    def _initialize_controller_parameters(self, configs: Dict[str, Any]):
        """
        Initialize controller-specific parameters.

        :param configs: Configuration dictionary
        """
        controller_config = configs["controller_config"]

        # Controller gains and configuration
        self.Kp = controller_config.get("stiffness", None)
        self.Kd = controller_config.get("damping", None)
        self.action_dim = controller_config.get("action_dim", None)
        self.control_dt = controller_config.get("control_dt", None)
        self.decimation = controller_config.get("decimation", None)

        self.default_joint_pos = (
            torch.tensor(
                controller_config.get("ISAAC_LAB_DEFAULT_JOINT_POS", None), dtype=torch.float32
            )
            if controller_config.get("ISAAC_LAB_DEFAULT_JOINT_POS", None) is not None
            else torch.tensor(controller_config["default_joint_pos"], dtype=torch.float32)
        )

        self.actions_mapping = np.array(
            controller_config.get(
                "JOINT_ACTION_ISAAC_LAB_TO_UNITREE_MAPPING", np.arange(self.action_dim)
        ))

        self.joint_obs_unitree_to_isaac_mapping = (
            torch.tensor(
                controller_config.get("JOINT_OBSERVATION_UNITREE_TO_ISAAC_LAB_MAPPING", None)
            )
            if controller_config.get("JOINT_OBSERVATION_UNITREE_TO_ISAAC_LAB_MAPPING", None)
            is not None
            else None
        )

        # Filter coefficient (0 < alpha < 1), lower values = more smoothing
        self.action_filter_alpha = controller_config.get("action_filter_alpha", 1.0)
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
        action_dim = configs["controller_config"].get("action_dim", None)
        obs_dim = configs["controller_config"].get("obs_dim", None)
        self.raw_action = torch.zeros(action_dim, dtype=torch.float32, device="cpu")

        # Observation history storage
        self.obs_buffer = ObservationHistoryStorage(
            num_envs=1,
            policy_architecture=self.policy_architecture,
            num_obs=obs_dim,
            max_length=1,
            device="cpu",
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

                print(f"Traceback: {traceback.format_exc()}")
                time.sleep(0.1)  # Prevent rapid error loops

    def _run_policy_inference(self):
        """Continuously run policy inference in a separate thread at a fixed dt of 0.02 seconds"""
        dt = self.control_dt * self.decimation  # Fixed time step in seconds (0.02)
        last_time = time.time()

        while True:
            try:

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
                        raw_action = self.policy(obs)

                    self.raw_action.copy_(raw_action[0][0])
                except Exception as e:
                    self.logger.error(f"Policy inference error: {e}")
                    time.sleep(0.1)  # Prevent rapid error loops
                    continue

            except Exception as e:
                self.logger.error(f"Policy inference thread error: {e}")
                time.sleep(0.1)  # Prevent rapid error loops

    def compute_joint_pos_targets(self):
        """
        Compute joint position targets based on the policy output.
        """
        try:
            # Ensure we have processed observations and have a valid action
            if self.obs_buffer.get().numel() == 0 or self.obs_buffer.get().sum() == 0:
                # If no observations yet, use default joint positions
                joint_pos_targets = self.default_joint_pos.cpu().numpy()[self.actions_mapping]
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
                    .numpy()[self.actions_mapping]
                )

            # # Clip the joint pos targets for safety
            # if hasattr(self, "soft_dof_pos_limit"):
            #     joint_indices = self.policy_joint_indices if hasattr(self, "policy_joint_indices") else None
            #     joint_pos_targets = self._clip_dof_pos(joint_pos_targets, joint_indices=joint_indices)

            return joint_pos_targets
        except Exception as e:
            self.logger.error(f"Error computing joint pos targets: {e}")

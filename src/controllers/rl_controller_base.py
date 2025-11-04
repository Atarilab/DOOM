import os
import time
import traceback
from typing import TYPE_CHECKING, Any, Dict

from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
import torch

from controllers.action_terms import JointPositionAction
from controllers.controller_base import ControllerBase
from utils.helpers import EMAFilter
from utils.thread_manager import ThreadManager

if TYPE_CHECKING:
    from robots.robot_base import RobotBase


class RLControllerBase(ControllerBase, Node):
    """
    Base Reinforcement Learning Locomotion Controller

    Provides core infrastructure for neural network-based robot locomotion control
    with concurrent observation processing and policy inference.
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        """
        Initialize the RL locomotion controller with model and configuration.

        :param robot: Robot model
        :param configs: Configuration dictionary
        """
        # Initialize ROS2 node
        Node.__init__(self, "rl_locomotion_controller")
        # Initialize controller base
        ControllerBase.__init__(self, robot=robot, configs=configs, debug=debug)

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

        # Thread control variables
        self._previous_active_state = False

    @property
    def active(self):
        """Get the active state of the controller."""
        return self._active

    @active.setter
    def active(self, value):
        """Set the active state and manage threads accordingly."""
        # Store the previous state to detect changes
        self._previous_active_state = getattr(self, "_active", False)
        self._active = value

        # If threading is not required, skip starting/stopping the threads
        if not hasattr(self, "use_threading"):
            self.use_threading = False
        if not self.use_threading:
            return

        # If transitioning from active to inactive, stop threads
        if self._previous_active_state and not value:
            if hasattr(self, "thread_manager"):
                self.thread_manager.stop()
        # If transitioning from inactive to active, start threads
        elif not self._previous_active_state and value:
            if hasattr(self, "thread_manager"):
                self.thread_manager.start()

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
        if model_path.endswith(".onnx"):
            from controllers.policies.policy import ONNXPolicy

            self.policy = ONNXPolicy(model_path, configs["controller_config"]["device"])
        else:
            from controllers.policies.policy import TorchScriptPolicy

            self.policy = TorchScriptPolicy(model_path, configs["controller_config"]["device"])

        # # Use TorchScript for optimized inference
        # self.policy = torch.jit.load(model_path).to(configs["controller_config"]["device"])
        # self.policy.eval()

        # Flatten RNN parameters to avoid memory fragmentation warnings
        self._flatten_rnn_parameters()

        # # Suppress the RNN memory warning for TorchScript models
        # # This warning is common with compiled models and doesn't affect functionality
        # warnings.filterwarnings("ignore", message="RNN module weights are not part of single contiguous chunk of memory")

        if self.logger:
            self.logger.info(f"Policy loaded from {model_path} on device {self.policy.device}")

        # Precompute static configurations
        self.action_scale = configs["controller_config"]["action_scale"]
        self.policy_architecture = configs["controller_config"].get("policy_architecture", "mlp")

    def _flatten_rnn_parameters(self):
        """
        Flatten RNN parameters to avoid memory fragmentation warnings.
        This is needed when using RNN/LSTM/GRU modules to ensure weights are contiguous in memory.
        """
        try:
            # For TorchScript models, we need to access the underlying modules
            if hasattr(self.policy, "graph"):
                # This is a TorchScript model, we can't directly access RNN modules
                # Try to access the original module if available
                if hasattr(self.policy, "original_module"):
                    for module in self.policy.original_module.modules():
                        if hasattr(module, "flatten_parameters"):
                            module.flatten_parameters()
                # For TorchScript models, we can try to access the code object
                elif hasattr(self.policy, "code"):
                    # This is a more advanced approach for TorchScript models
                    # The flatten_parameters() should ideally be called during model creation
                    pass
            else:
                # For regular PyTorch models, recursively find and flatten RNN parameters
                for module in self.policy.modules():
                    if hasattr(module, "flatten_parameters"):
                        module.flatten_parameters()
        except Exception as e:
            # Silently handle the exception to avoid spamming logs
            # The warning will still appear but won't be repeated in our logs
            pass

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
        self.use_buffer = controller_config.get("use_buffer", False)
        self.clip_actions = controller_config.get("clip_actions", False)
        self.policy_dt = self.control_dt * self.decimation
        self.counter = 0

        self.obs_buffer_length = controller_config.get("obs_buffer_length", 1)

        self.default_joint_pos = (
            torch.tensor(
                controller_config.get("ISAAC_LAB_DEFAULT_JOINT_POS", None), dtype=torch.float32, device=self.device
            )
            if controller_config.get("ISAAC_LAB_DEFAULT_JOINT_POS", None) is not None
            else torch.tensor(controller_config.get("default_joint_pos", None), dtype=torch.float32, device=self.device)
        )
        self.default_joint_pos_np = self.default_joint_pos.cpu().numpy()

        if hasattr(self.robot, "joints_isaac2unitree"):
            self.actions_mapping = torch.tensor(self.robot.joints_isaac2unitree)
        else:
            self.actions_mapping = torch.arange(self.robot.num_joints)

        if hasattr(self.robot, "joints_unitree2isaac"):
            self.joint_obs_unitree_to_isaac_mapping = torch.tensor(self.robot.joints_unitree2isaac)
        else:
            self.joint_obs_unitree_to_isaac_mapping = torch.arange(self.robot.num_joints)

        # Initialize raw_action on GPU if not already done
        # self.raw_action = torch.zeros(self.action_dim, dtype=torch.float32, device=self.device)
        # Filter coefficient (0 < alpha < 1), lower values = more smoothing
        self.filtered_action = EMAFilter(
            controller_config.get("action_filter_alpha", 1.0), self.action_dim, device=self.device
        )

        self.action_term = JointPositionAction(configs, self.action_scale, self.default_joint_pos, self.actions_mapping)
        # Initial state and commands
        self.latest_state = None
        self.joint_pos_targets = torch.zeros(self.robot.num_joints, dtype=torch.float32, device=self.device)
        self.cmd = {}

    def _configure_processing_infrastructure(self, configs: Dict[str, Any]):
        """
        Set up concurrent processing infrastructure for observations and policy.

        :param configs: Configuration dictionary
        """
        self.use_threading = configs["controller_config"].get("use_threading", False)

        # Initialize thread manager if threading is enabled
        if self.use_threading:
            self.thread_manager = ThreadManager(logger=self.logger, debug=self.debug)
            self.thread_manager.add_thread("observation_processing", self._process_observations)
            self.thread_manager.add_thread("policy_inference", self._run_policy_inference)

    def _initialize_obs_buffer(self):
        """
        Initialize the observation buffer once the obs_manager is available.
        This is called by set_obs_manager in the base class.
        """
        if hasattr(self, "obs_manager") and self.obs_manager is not None:
            self.obs_manager.initialize_obs_buffer(
                max_buffer_length=self.obs_buffer_length, policy_architecture=self.policy_architecture
            )

    def set_obs_manager(self, obs_manager):
        """
        Override set_obs_manager to initialize obs_buffer after obs_manager is set.
        """
        super().set_obs_manager(obs_manager)
        if self.use_buffer:
            self._initialize_obs_buffer()

    def set_mode(self):
        """
        Override set_mode to initialize the controller when mode is set.
        Thread management is handled automatically by the active property setter.
        """
        super().set_mode()
        self.counter = 0
        self.action_term.raw_action.zero_()
        if self.use_buffer and self.obs_manager is not None:
            self.obs_manager.reset_buffer(done=torch.tensor([False]))

    def _process_observations(self):
        """Continuously process observations in a separate thread"""
        while hasattr(self, "thread_manager") and self.thread_manager.should_continue():
            try:
                # Check if controller is active
                if not self.active:
                    time.sleep(0.01)  # Sleep when inactive
                    continue

                with self._lock:
                    current_state = self.latest_state

                if current_state is None:
                    time.sleep(0.01)  # Prevent busy waiting
                    continue

                # Compute and store observations
                with torch.no_grad():
                    try:
                        # Check if obs_manager is available
                        if self.obs_manager is None:
                            time.sleep(0.01)  # Wait for obs_manager to be set
                            continue

                        # Use compute_full_tensor for efficient observation processing
                        obs_tensor = self.obs_manager.compute_full_tensor(current_state, batch_idx=0)

                    except Exception as e:
                        print(f"Error computing full observation tensor: {e}")
                        time.sleep(0.1)  # Prevent rapid error loops
                        continue

            except Exception as e:
                print(f"Observation processing error: {e}")
                print(f"Error type: {type(e)}")

                print(f"Traceback: {traceback.format_exc()}")
                time.sleep(0.1)  # Prevent rapid error loops

    def _run_policy_inference(self):
        """Continuously run policy inference in a separate thread.

        NOTE: I have commented out the sleep to make the policy inference run at the same rate as the control frequency.
        Need to confirm if this is a good idea.
        """
        last_time = time.time()

        while hasattr(self, "thread_manager") and self.thread_manager.should_continue():
            try:
                # Check if controller is active
                if not self.active:
                    time.sleep(0.01)  # Sleep when inactive
                    continue

                # Calculate time since last iteration
                current_time = time.time()
                elapsed = current_time - last_time

                # Sleep if we're ahead of schedule
                if elapsed < self.policy_dt:
                    time.sleep(self.policy_dt - elapsed)
                    current_time = time.time()  # Update current time after sleep

                # Update last time for next iteration
                last_time = current_time

                try:
                    if self.use_buffer:
                        obs = self.obs_manager.get_from_buffer()
                    else:
                        obs = self.obs_manager.get_latest_full_obs_tensor()

                    # Check if we have valid observations
                    if obs is None or obs.numel() == 0 or obs.sum() == 0:
                        time.sleep(0.01)  # Wait for observations
                        continue

                    # Policy inference
                    with torch.no_grad():
                        raw_action = self.policy(obs).squeeze(0).squeeze(0)
                    self.action_term.raw_action.copy_(raw_action)

                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Policy inference error: {e}")
                    time.sleep(0.1)  # Prevent rapid error loops
                    continue

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Policy inference thread error: {e}")
                time.sleep(0.1)  # Prevent rapid error loops

    def compute_joint_pos_targets(self):
        """
        Compute joint position targets based from pre-computed policy output.
        We also apply an exponential moving average filter to smooth the actions.

        Note: We use this when we run with threading, where policy inference and observation processing are done in separate threads.
        """
        try:
            # Ensure we have processed observations and have a valid action
            if self.action_term.raw_action.sum() == 0:
                # If no observations yet, use default joint positions
                joint_pos_targets = self.default_joint_pos_np[self.actions_mapping]
            else:
                joint_pos_targets = self.action_term.process_actions(self.action_term.raw_action)

            # Clip the joint pos targets for safety
            if hasattr(self, "soft_dof_pos_limit"):
                joint_pos_targets = self._clip_dof_pos(joint_pos_targets)

            return joint_pos_targets
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error computing joint pos targets: {e}")
            return self.default_joint_pos_np[self.actions_mapping]

    def compute_joint_pos_targets_from_policy(self, obs_tensor: torch.Tensor):
        """
        Compute joint position targets directly from the policy output.
        Compute full tensor obs, pass it to the policy, and then compute the joint pos targets.
        We also apply an exponential moving average filter to smooth the actions.

        Note: We use this when we do not run with threading, where policy inference and observation processing are done here.
        """
        try:
            with torch.no_grad():
                raw_action = self.policy(obs_tensor).detach().squeeze(0)

            processed_action = self.action_term.process_actions(raw_action)
            return processed_action

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error computing joint pos targets from policy: {e}")
            return self.default_joint_pos[self.actions_mapping]

    #####################
    # Thread management #
    #####################

    def __del__(self):
        """Cleanup method to ensure threads are stopped when controller is destroyed."""
        if hasattr(self, "thread_manager"):
            self.thread_manager.stop()

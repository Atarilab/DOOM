import json
import os
import threading
import time
from typing import Any, Callable, Dict

import numpy as np
import torch
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster

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


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


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
        Node.__init__(self, "rl_locomotion_controller")
        # Initialize controller base
        ControllerBase.__init__(self, mj_model_wrapper=mj_model_wrapper, configs=configs)

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
        self.action_filter_alpha = (
            controller_config["action_filter_alpha"] if "action_filter_alpha" in controller_config else 1.0
        )
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
        self.obs_buffer = ObservationHistoryStorage(num_envs=1, num_obs=obs_dim, max_length=5, device="cpu")

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
                        raw_action = torch.clamp(raw_action, min=-3.5, max=3.5)
                        #self.command_manager.logger.debug(f"Raw actions: {raw_action}")

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
        
        loc_from_vid_paper_01_log = {}
        
        loc_from_vid_paper_01_log["state"] = state

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
                
                
            loc_from_vid_paper_01_log["joint_pos_targets"] = joint_pos_targets


            # Clip the joint pos targets for safety
            joint_pos_targets = self._clip_dof_pos(joint_pos_targets)

            loc_from_vid_paper_01_log["joint_pos_targets_clipped"] = joint_pos_targets


            # self.command_manager.logger.debug(f"joint_pos_targets: {joint_pos_targets}")

            # efforts = [joint_pos_targets[i] * self.Kp for i in range(12)]

            # self.command_manager.logger.debug(f"efforts calculated only kp: {efforts}")

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
            loc_from_vid_paper_01_log["cmd"] = self.cmd
            
            self.command_manager.logger.debug(
                f"LOC-FROM-VID-PAPER-EXP-01: %s",
                json.dumps(loc_from_vid_paper_01_log, default=to_serializable, separators=(",", ":"))
            )

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
            # Create a new tensor with the updated values
            new_velocity_commands = self.velocity_commands.clone()
            
            if "x_velocity" in new_commands:
                new_velocity_commands[0] = new_commands["x_velocity"]
            if "y_velocity" in new_commands:
                new_velocity_commands[1] = new_commands["y_velocity"]
            if "yaw_rate" in new_commands:
                new_velocity_commands[2] = new_commands["yaw_rate"]
                
            self.velocity_commands = new_velocity_commands
            
            if self.command_manager and self.command_manager.logger:
                self.command_manager.logger.debug(f"Command Updated: {new_commands}")
                self.command_manager.logger.debug(
                    f"LOC-FROM-VID-PAPER-EXP-01 Command Updated: %s",
                json.dumps(new_commands, default=to_serializable, separators=(",", ":"))
                )
        except ValueError as e:
            # Log error or handle validation failure
            if self.command_manager and self.command_manager.logger:
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
        self.obs_manager.register(
            "velocity_commands",
            ObsTerm(
                velocity_commands,
                params={"velocity_commands": lambda: self.velocity_commands},
            ),
        )
        self.obs_manager.register("projected_gravity", ObsTerm(projected_gravity_b))
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
        
    def get_joystick_mappings(self):
        """
        Define joystick button mappings for gait changes and heading control.
        
        Returns:
            Dict mapping button names to callback functions.
        """
        return {

            # Step Size
            "up": lambda: self.change_commands({"x_velocity": self.velocity_commands[0] + 0.1, "y_velocity": self.velocity_commands[1]}),
            "down": lambda: self.change_commands({"x_velocity": self.velocity_commands[0] - 0.1, "y_velocity": self.velocity_commands[1]}),
            # Lateral Position 
            "left": lambda: self.change_commands({"x_velocity": self.velocity_commands[0], "y_velocity": self.velocity_commands[1] + 0.1}),
            "right": lambda: self.change_commands({"x_velocity": self.velocity_commands[0], "y_velocity": self.velocity_commands[1] - 0.1}),
        }

    def set_mode(self):
        """Runs when the mode is changed in the UI."""
        super().set_mode()
        self.velocity_commands = torch.zeros_like(self.velocity_commands)
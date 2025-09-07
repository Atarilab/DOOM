from typing import Dict, List, Optional

import mujoco
import torch

from utils.math import quat_to_rotmatrix
from utils.helpers import tensorify


class MjRobotWrapper:
    """MuJoCo wrapper for quadruped robot simulation and computation."""

    def __init__(self, xml_path: str, ee_names: List[str], base_link: str, device: Optional[torch.device] = None):
        """
        Initialize MuJoCo model from XML.

        Args:
            xml_path: Path to the robot's XML file
            ee_names: Names of the end effectors of the robot
            base_link: Name of the base link of the robot
            device: Device to use for torch operations
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)  # type: ignore
        self.data = mujoco.MjData(self.model)  # type: ignore

        # Cache commonly used body and joint indices
        self.body_names = {}
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)  # type: ignore
            if name:
                self.body_names[name] = i

        self.joint_names = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)  # type: ignore
            if name and name != "floating_base_joint":
                self.joint_names[name] = i
                
        self.num_joints = len(self.joint_names)

        # Debug print to see what bodies are available
        print(f"Available bodies: {list(self.body_names.keys())}")
        print(f"Available joints: {list(self.joint_names.keys())}")

        self.base_link = base_link
        self.ee_indices = [self.body_names.get(name, -1) for name in ee_names]
        self.base_idx = self.body_names.get(self.base_link, -1)
        
        # Store device for torch operations
        self.device = device if device is not None else torch.device('cpu')

        # Pre-allocate tensors for performance optimization
        self.num_ee = len(self.ee_indices)
        self._ee_positions_w = torch.zeros((self.num_ee, 3), dtype=torch.float32, device=self.device)
        self._ee_positions_b = torch.zeros((self.num_ee, 3), dtype=torch.float32, device=self.device)
        self._feet_positions_init = torch.zeros((self.num_ee, 3), dtype=torch.float32, device=self.device)
        
        # Pre-compute valid indices for performance
        self.valid_ee_indices = [i for i, idx in enumerate(self.ee_indices) if idx >= 0]
        self.valid_mj_indices = [self.ee_indices[i] for i in self.valid_ee_indices]

        # Initialize world frame transform
        self._init_world_frame = False
        self.world_to_init = torch.eye(4, dtype=torch.float32, device=self.device)  # Will store the transform from world to initial pose

        # Store initial feet positions
        self.initial_feet_positions_init_frame = None

    def set_initial_world_frame(self, state: Optional[Dict] = None, caller: Optional[str] = None) -> None:
        """
        Set the current robot pose as the fixed world frame.

        Args:
            state: Dictionary containing robot state. Uses state["base_pos_w"]
                  and state["base_quat"] for the initial frame.
            caller: Name of the controller calling this method. Must be "ZeroTorqueController".
        """
        if caller != "ZeroTorqueController":
            raise RuntimeError("set_initial_world_frame can only be called from ZeroTorqueController")

        try:
            if state is None:
                raise ValueError("State cannot be None")
            # Create 4x4 transform matrix
            self.world_to_init[:3, :3] = quat_to_rotmatrix(state["robot/base_quat"], order="wxyz")
            self.world_to_init[:3, 3] = state["robot/base_pos_w"]
            self._init_world_frame = True

        except Exception as e:
            print(f"Error setting init frame: {e}")
            return


        # Update the robot state with the current state
        self.update(state)

        # Save the initial feet positions in the init frame
        self.initial_feet_positions_init_frame = self.get_feet_positions_init_frame()
        # Calculate the ground height as the average of the initial feet positions in the init frame (minus the foot radius)
        self.ground_height_init_frame = torch.mean(self.initial_feet_positions_init_frame, dim=0)[2] - 0.025

    def update(self, state: Optional[Dict] = None) -> None:
        """
        Reset robot state and simulation state.

        Args:
            - state (Dict): Complete state dictionary containing robot state

        """
        if state is not None:
            # Extract joint positions and velocities from state
            q = state["robot/joint_pos"]
            v = state["robot/joint_vel"]

            # Set base pose in world frame
            if "robot/base_pos_w" in state:
                self.data.qpos[0:3] = state["robot/base_pos_w"].cpu().numpy()
            if "robot/base_quat" in state:
                self.data.qpos[3:7] = state["robot/base_quat"].cpu().numpy()
            self.data.qpos[-self.num_joints :] = q.cpu().numpy()
            self.data.qvel[-self.num_joints :] = v.cpu().numpy()

        mujoco.mj_forward(self.model, self.data)  # type: ignore

    def get_ee_positions_w(self) -> torch.Tensor:
        """Get end effector positions in world frame."""
        # Reset to zeros for invalid indices
        self._ee_positions_w.zero_()
        self._ee_positions_w[self.valid_ee_indices] = torch.from_numpy(self.data.xpos[self.valid_mj_indices]).float().to(self.device)
        return self._ee_positions_w

    def get_ee_positions_b(self) -> torch.Tensor:
        """Get end effector positions in base frame."""
        world_positions = self.get_ee_positions_w()
        if self.base_idx < 0:
            return world_positions  # Return world positions if base not found

        base_pos = tensorify(self.data.xpos[self.base_idx], dtype=torch.float32, device=self.device)
        base_rot = tensorify(self.data.xmat[self.base_idx].reshape(3, 3), dtype=torch.float32, device=self.device)

        # Vectorized transformation from world to base frame using pre-allocated tensor
        # Subtract base position from all positions at once
        rel_pos = world_positions - base_pos
        # Apply rotation transformation to all positions at once
        base_frame_positions = base_rot.T @ rel_pos.T
        self._ee_positions_b.copy_(base_frame_positions.T)
        return self._ee_positions_b

    def base_pos_init_frame(self) -> torch.Tensor:
        """Get the base position in the init frame."""
        if not self._init_world_frame:
            raise RuntimeError("Init frame not set. Call set_initial_world_frame() first.")
        return self.get_body_position_init_frame(self.base_link)

    def transform_init_to_base(self, pos: torch.Tensor) -> torch.Tensor:
        """Transform quantities from the init frame to the base frame.

        Args:
            pos: Tensor of shape (N,3) or (N,M,3) containing positions in the init frame

        Returns:
            Tensor of the same shape as pos but with positions transformed to the base frame
        """
        # Get the base position and orientation
        if self.base_idx < 0:
            raise ValueError("Base body not found in model")

        base_pos = tensorify(self.data.xpos[self.base_idx], dtype=torch.float32, device=self.device)
        base_rot = tensorify(self.data.xmat[self.base_idx].reshape(3, 3), dtype=torch.float32, device=self.device)

        # Get the original shape
        original_shape = pos.shape

        # Reshape to (N*M, 3) if needed
        if len(original_shape) == 3:
            # For (N,M,3) shape
            N, M, _ = original_shape
            pos_reshaped = pos.reshape(-1, 3)
        else:
            # For (N,3) shape
            pos_reshaped = pos

        # Apply transformation to each position
        # First subtract the base position to get relative position
        rel_pos = pos_reshaped - base_pos
        # Then rotate using the transpose of the base rotation matrix
        transformed = base_rot.T @ rel_pos.T
        transformed = transformed.T

        # Reshape back to original shape if needed
        if len(original_shape) == 3:
            transformed = transformed.reshape(original_shape)

        return transformed

    def transform_world_to_base(self, pos: torch.Tensor) -> torch.Tensor:
        """Transform quantities from the world frame to the base frame.

        Args:
            pos: Tensor of shape (N,3) or (N,M,3) containing positions in the world frame

        Returns:
            Tensor of the same shape as pos but with positions transformed to the base frame
        """
        # Get the base position and orientation
        if self.base_idx < 0:
            raise ValueError("Base body not found in model")

        base_pos = tensorify(self.data.xpos[self.base_idx], dtype=torch.float32, device=self.device)
        base_rot = tensorify(self.data.xmat[self.base_idx].reshape(3, 3), dtype=torch.float32, device=self.device)

        # Get the original shape
        original_shape = pos.shape

        # Reshape to (N*M, 3) if needed
        if len(original_shape) == 3:
            # For (N,M,3) shape
            N, M, _ = original_shape
            pos_reshaped = pos.reshape(-1, 3)
        else:
            # For (N,3) shape
            pos_reshaped = pos

        # Apply transformation to each position
        # First subtract the base position to get relative position
        rel_pos = pos_reshaped - base_pos
        # Then rotate using the transpose of the base rotation matrix
        transformed = base_rot.T @ rel_pos.T
        transformed = transformed.T

        # Reshape back to original shape if needed
        if len(original_shape) == 3:
            transformed = transformed.reshape(original_shape)

        return transformed

    def transform_world_to_init_frame(self, pos: torch.Tensor) -> torch.Tensor:
        """Transform quantities from the world frame to the init frame.

        Args:
            pos: Tensor of shape (N,3) or (N,M,3) containing positions in the world frame

        Returns:
            Tensor of the same shape as pos but with positions transformed to the init frame
        """
        if not self._init_world_frame:
            raise RuntimeError("Init frame not set. Call set_initial_world_frame() first.")

        # Get the original shape
        original_shape = pos.shape

        # Reshape to (N*M, 3) if needed
        if len(original_shape) == 3:
            # For (N,M,3) shape
            N, M, _ = original_shape
            pos_reshaped = pos.reshape(-1, 3)
        else:
            # For (N,3) shape
            pos_reshaped = pos

        # Apply transformation to each position
        # First subtract the init frame origin to get relative position
        rel_pos = pos_reshaped - self.world_to_init[:3, 3]
        # Then rotate using the transpose of the init frame rotation matrix
        transformed = self.world_to_init[:3, :3].T @ rel_pos.T
        transformed = transformed.T

        # Reshape back to original shape if needed
        if len(original_shape) == 3:
            transformed = transformed.reshape(original_shape)

        return transformed

    def get_feet_positions_init_frame(self) -> torch.Tensor:
        """Get feet positions in the init frame."""
        if not self._init_world_frame:
            raise RuntimeError("Init frame not set. Call set_initial_world_frame() first.")

        world_positions = self.get_ee_positions_w()
        
        # Vectorized transformation from world to init frame using pre-allocated tensor
        # Subtract init frame origin from all positions at once
        rel_pos = world_positions - self.world_to_init[:3, 3]
        # Apply rotation transformation to all positions at once
        init_frame_positions = self.world_to_init[:3, :3].T @ rel_pos.T
        self._feet_positions_init.copy_(init_frame_positions.T)
        return self._feet_positions_init

    def get_body_position_world(self, body_name: str) -> torch.Tensor:
        """Get position of a specific body in world frame."""
        idx = self.body_names.get(body_name, -1)
        if idx < 0:
            raise ValueError(f"Body '{body_name}' not found in model")
        return tensorify(self.data.xpos[idx], dtype=torch.float32, device=self.device)

    def get_body_position_init_frame(self, body_name: str) -> torch.Tensor:
        """Get position of a specific body in the init frame."""
        if not self._init_world_frame:
            raise RuntimeError("Init frame not set. Call set_initial_world_frame() first.")

        world_pos = self.get_body_position_world(body_name)
        rel_pos = world_pos - self.world_to_init[:3, 3]
        return self.world_to_init[:3, :3].T @ rel_pos

    def get_body_orientation_world(self, body_name: str) -> torch.Tensor:
        """Get orientation matrix of a specific body in world frame."""
        idx = self.body_names.get(body_name, -1)
        if idx < 0:
            raise ValueError(f"Body '{body_name}' not found in model")
        return tensorify(self.data.xmat[idx].reshape(3, 3), dtype=torch.float32, device=self.device)

    def get_com_position_world(self) -> torch.Tensor:
        """Get center of mass position in world frame."""
        mujoco.mj_subtreeVel(self.model, self.data)  # type: ignore
        return tensorify(self.data.subtree_com[0], dtype=torch.float32, device=self.device)

    def get_com_position_init_frame(self) -> torch.Tensor:
        """Get center of mass position in the init frame."""
        if not self._init_world_frame:
            raise RuntimeError("Init frame not set. Call set_initial_world_frame() first.")

        world_com = self.get_com_position_world()
        rel_pos = world_com - self.world_to_init[:3, 3]
        return self.world_to_init[:3, :3].T @ rel_pos

    def get_base_height_init_frame(self) -> float:
        """
        Get the height of the base in the init frame relative to the average
        of the initial feet positions using vectorized operations.

        Returns:
            float: The height of the base relative to the average initial feet position.
        """
        if not self._init_world_frame:
            raise RuntimeError("Init frame not set. Call set_initial_world_frame() first.")

        if self.initial_feet_positions_init_frame is None:
            raise RuntimeError("Initial feet positions not saved. Call set_initial_world_frame() first.")

        # Get base position in world frame
        base_pos_world = self.get_body_position_world("base_link")

        # Transform to initial frame using vectorized operations
        rel_pos = base_pos_world - self.world_to_init[:3, 3]
        base_pos_init = self.world_to_init[:3, :3].T @ rel_pos

        # Calculate average of initial feet positions using vectorized operations
        # This is more efficient than torch.mean(dim=0) for large arrays
        avg_feet_pos_init = (
            torch.sum(self.initial_feet_positions_init_frame, dim=0) / self.initial_feet_positions_init_frame.shape[0]
        )

        # Return the height difference (z-coordinate)
        return float(base_pos_init[2] - avg_feet_pos_init[2] - 0.025)

    def get_initial_feet_positions(self) -> torch.Tensor:
        """
        Get the feet positions that were saved when the init frame was set.

        Returns:
            torch.Tensor: The feet positions in the init frame.
        """
        if not self._init_world_frame:
            raise RuntimeError("Init frame not set. Call set_initial_world_frame() first.")

        if self.initial_feet_positions_init_frame is None:
            raise RuntimeError("Initial feet positions not saved. Call set_initial_world_frame() first.")

        return self.initial_feet_positions_init_frame

    def transform_world_to_base_with_quat(self, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """Transform quantities from the world frame to the base frame using a custom quaternion.

        Args:
            pos: Tensor of shape (N,3) or (N,M,3) containing positions in the world frame
            quat: Quaternion in wxyz format to use for rotation instead of the robot's current orientation

        Returns:
            Tensor of the same shape as pos but with positions transformed to the base frame
        """
        # Get the base position
        if self.base_idx < 0:
            raise ValueError("Base body not found in model")

        base_pos = tensorify(self.data.xpos[self.base_idx], dtype=torch.float32, device=self.device)

        # Convert quaternion to rotation matrix using torch operations
        base_rot = quat_to_rotmatrix(quat, order="wxyz")

        # Get the original shape
        original_shape = pos.shape

        # Reshape to (N*M, 3) if needed
        if len(original_shape) == 3:
            # For (N,M,3) shape
            N, M, _ = original_shape
            pos_reshaped = pos.reshape(-1, 3)
        else:
            # For (N,3) shape
            pos_reshaped = pos

        # Apply transformation to each position
        # First subtract the base position to get relative position
        rel_pos = pos_reshaped - base_pos
        # Then rotate using the transpose of the base rotation matrix
        transformed = base_rot.T @ rel_pos.T
        transformed = transformed.T

        # Reshape back to original shape if needed
        if len(original_shape) == 3:
            transformed = transformed.reshape(original_shape)

        return transformed

    def transform_init_to_base_with_quat(self, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """Transform quantities from the init frame to the base frame using a custom quaternion.

        Args:
            pos: Tensor of shape (N,3) or (N,M,3) containing positions in the init frame
            quat: Quaternion in wxyz format to use for rotation instead of the robot's current orientation

        Returns:
            Tensor of the same shape as pos but with positions transformed to the base frame
        """
        # Get the base position
        if self.base_idx < 0:
            raise ValueError("Base body not found in model")

        base_pos = tensorify(self.data.xpos[self.base_idx], dtype=torch.float32, device=self.device)

        # Convert quaternion to rotation matrix using torch operations
        base_rot = quat_to_rotmatrix(quat, order="wxyz")

        # Get the original shape
        original_shape = pos.shape

        # Reshape to (N*M, 3) if needed
        if len(original_shape) == 3:
            # For (N,M,3) shape
            N, M, _ = original_shape
            pos_reshaped = pos.reshape(-1, 3)
        else:
            # For (N,3) shape
            pos_reshaped = pos

        # Apply transformation to each position
        # First subtract the base position to get relative position
        rel_pos = pos_reshaped - base_pos
        # Then rotate using the transpose of the base rotation matrix
        transformed = base_rot.T @ rel_pos.T
        transformed = transformed.T

        # Reshape back to original shape if needed
        if len(original_shape) == 3:
            transformed = transformed.reshape(original_shape)

        return transformed

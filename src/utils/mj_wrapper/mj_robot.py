from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np
from numpy.typing import NDArray
from utils.math import quat_to_rotmatrix


class MjQuadRobotWrapper:
    """MuJoCo wrapper for quadruped robot simulation and computation."""

    def __init__(self, xml_path: str):
        """
        Initialize MuJoCo model from XML.

        Args:
            xml_path: Path to the robot's XML file
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Cache commonly used body and joint indices
        self.body_names = {}
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                self.body_names[name] = i

        self.joint_names = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_names[name] = i

        # Debug print to see what bodies are available
        print(f"Available bodies: {list(self.body_names.keys())}")

        # Cache feet body indices
        self.feet_bodies = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.feet_indices = [self.body_names.get(name, -1) for name in self.feet_bodies]

        # Initialize world frame transform
        self._init_world_frame = False
        self.world_to_init = np.eye(4)  # Will store the transform from world to initial pose

        # Store initial feet positions
        self.initial_feet_positions_init_frame = None

    def set_initial_world_frame(self, state: Optional[Dict] = None, caller: Optional[str] = None) -> None:
        """
        Set the current robot pose as the fixed world frame.

        Args:
            state: Dictionary containing robot state. Uses state["base_pos_w"]
                  and state["base_quat"] for the initial frame.
            caller: Name of the controller calling this method. Must be "IdleController".
        """
        if caller != "IdleController":
            raise RuntimeError("set_initial_world_frame can only be called from IdleController")

        try:
            # Use the base position and orientation from the state
            base_pos = np.array(state["base_pos_w"])
            # Convert quaternion (w,x,y,z) to rotation matrix using the utility function
            base_rot = quat_to_rotmatrix(np.array(state["base_quat"]), order="wxyz")

        except Exception as e:
            print(f"Error setting init frame: {e}")
            return

        # Create 4x4 transform matrix
        self.world_to_init[:3, :3] = base_rot
        self.world_to_init[:3, 3] = base_pos
        self._init_world_frame = True

        # Update the robot state with the current state
        self.update(state)

        # Save the initial feet positions in the init frame
        self.initial_feet_positions_init_frame = self.get_feet_positions_init_frame()
        # Calculate the ground height as the average of the initial feet positions in the init frame (minus the foot radius)
        self.ground_height_init_frame = np.mean(self.initial_feet_positions_init_frame, axis=0)[2] - 0.025

    def update(self, state: Dict = None) -> None:
        """
        Reset robot state and simulation state.

        Args:
            - state (Dict): Complete state dictionary containing robot state

        """
        if state is not None:
            # Extract joint positions and velocities from state
            q = state["joint_pos"]
            v = state["joint_vel"]

            # Set base pose in world frame
            if "base_pos_w" in state:
                self.data.qpos[0:3] = state["base_pos_w"]
            if "base_quat" in state:
                self.data.qpos[3:7] = state["base_quat"]
        self.data.qpos[-12:] = q
        self.data.qvel[-12:] = v

        mujoco.mj_forward(self.model, self.data)

    def get_feet_positions_world(self) -> np.ndarray:
        """Get feet positions in world frame."""
        positions = []
        for idx in self.feet_indices:
            if idx >= 0:  # Check if the index is valid
                pos = self.data.xpos[idx]
                positions.append(pos)
            else:
                # If foot not found, use a default position
                positions.append(np.zeros(3))
        return np.array(positions)

    def get_feet_positions_base(self) -> np.ndarray:
        """Get feet positions in base frame."""
        world_positions = self.get_feet_positions_world()
        base_idx = self.body_names.get("base", -1)
        if base_idx < 0:
            return world_positions  # Return world positions if base not found

        base_pos = self.data.xpos[base_idx]
        base_rot = self.data.xmat[base_idx].reshape(3, 3)

        positions = []
        for pos in world_positions:
            rel_pos = pos - base_pos
            base_frame_pos = base_rot.T @ rel_pos
            positions.append(base_frame_pos)
        return np.array(positions)

    def base_pos_init_frame(self) -> np.ndarray:
        """Get the base position in the init frame."""
        if not self._init_world_frame:
            raise RuntimeError("Init frame not set. Call set_initial_world_frame() first.")
        return self.get_body_position_init_frame("base_link")

    def transform_init_to_base(self, pos: np.ndarray) -> np.ndarray:
        """Transform quantities from the init frame to the base frame.

        Args:
            pos: Array of shape (N,3) or (N,M,3) containing positions in the init frame

        Returns:
            Array of the same shape as pos but with positions transformed to the base frame
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
        transformed = self.world_to_init[:3, :3].T @ (pos_reshaped - self.world_to_init[:3, 3]).T
        transformed = transformed.T

        # Reshape back to original shape if needed
        if len(original_shape) == 3:
            transformed = transformed.reshape(original_shape)

        return transformed

    def transform_world_to_base(self, pos: np.ndarray) -> np.ndarray:
        """Transform quantities from the world frame to the base frame.

        Args:
            pos: Array of shape (N,3) or (N,M,3) containing positions in the world frame

        Returns:
            Array of the same shape as pos but with positions transformed to the base frame
        """
        # Get the base position and orientation
        base_idx = self.body_names.get("base_link", -1)
        if base_idx < 0:
            raise ValueError("Base body not found in model")

        base_pos = self.data.xpos[base_idx]
        base_rot = self.data.xmat[base_idx].reshape(3, 3)

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

    def get_feet_positions_init_frame(self) -> np.ndarray:
        """Get feet positions in the init frame."""
        if not self._init_world_frame:
            raise RuntimeError("Init frame not set. Call set_initial_world_frame() first.")

        world_positions = self.get_feet_positions_world()
        positions = []
        for pos in world_positions:
            # Transform from current world to init frame
            rel_pos = pos - self.world_to_init[:3, 3]
            init_frame_pos = self.world_to_init[:3, :3].T @ rel_pos
            positions.append(init_frame_pos)
        return np.array(positions)

    def get_body_position_world(self, body_name: str) -> np.ndarray:
        """Get position of a specific body in world frame."""
        idx = self.body_names.get(body_name, -1)
        if idx < 0:
            raise ValueError(f"Body '{body_name}' not found in model")
        return self.data.xpos[idx]

    def get_body_position_init_frame(self, body_name: str) -> np.ndarray:
        """Get position of a specific body in the init frame."""
        if not self._init_world_frame:
            raise RuntimeError("Init frame not set. Call set_initial_world_frame() first.")

        world_pos = self.get_body_position_world(body_name)
        rel_pos = world_pos - self.world_to_init[:3, 3]
        return self.world_to_init[:3, :3].T @ rel_pos

    def get_body_orientation_world(self, body_name: str) -> np.ndarray:
        """Get orientation matrix of a specific body in world frame."""
        idx = self.body_names.get(body_name, -1)
        if idx < 0:
            raise ValueError(f"Body '{body_name}' not found in model")
        return self.data.xmat[idx].reshape(3, 3)

    def get_com_position_world(self) -> np.ndarray:
        """Get center of mass position in world frame."""
        mujoco.mj_subtreeVel(self.model, self.data)
        return self.data.subtree_com[0]

    def get_com_position_init_frame(self) -> np.ndarray:
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
        # This is more efficient than np.mean(axis=0) for large arrays
        avg_feet_pos_init = (
            np.sum(self.initial_feet_positions_init_frame, axis=0) / self.initial_feet_positions_init_frame.shape[0]
        )

        # Return the height difference (z-coordinate)
        return base_pos_init[2] - avg_feet_pos_init[2]

    def get_initial_feet_positions(self) -> np.ndarray:
        """
        Get the feet positions that were saved when the init frame was set.

        Returns:
            np.ndarray: The feet positions in the init frame.
        """
        if not self._init_world_frame:
            raise RuntimeError("Init frame not set. Call set_initial_world_frame() first.")

        if self.initial_feet_positions_init_frame is None:
            raise RuntimeError("Initial feet positions not saved. Call set_initial_world_frame() first.")

        return self.initial_feet_positions_init_frame

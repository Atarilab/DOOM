"""
Unified interface for joint mapping between MuJoCo and Isaac Lab coordinate systems.

This module provides a comprehensive interface to handle joint order conversions
between different robot coordinate systems, specifically MuJoCo and Isaac Lab.
"""

from enum import Enum
from typing import List, Union

import numpy as np


class JointOrder(Enum):
    """Enumeration for different joint ordering schemes."""

    MUJOCO = "mujoco"
    ISAAC_LAB = "isaac_lab"
    UNITREE_DDS = "unitree_dds"


class JointMappingInterface:
    """
    Unified interface for mapping between different joint coordinate systems.

    This class provides methods to convert joint states, positions, velocities,
    and actions between MuJoCo and Isaac Lab coordinate systems.
    """

    def __init__(self, robot_name: str):
        """
        Initialize the joint mapping interface.

        Args:
            robot_name (str): Name of the robot (e.g., "go2", "g1")
        """
        self.robot_name = robot_name.lower()
        self._setup_joint_orders()
        self._setup_mappings()

    def _setup_joint_orders(self):
        """Setup joint orders for different coordinate systems."""
        if self.robot_name == "go2":
            # MuJoCo order: Groups by leg (FR → FL → RR → RL), then by joint type (hip → thigh → calf)
            self.mujoco_joint_names = [
                "FR_hip_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
                "FL_hip_joint",
                "FL_thigh_joint",
                "FL_calf_joint",
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
            ]

            # Isaac Lab order: Groups by joint type (all hips → all thighs → all calves), then by leg (FL → FR → RL → RR)
            self.isaac_lab_joint_names = [
                "FL_hip_joint",
                "FR_hip_joint",
                "RL_hip_joint",
                "RR_hip_joint",
                "FL_thigh_joint",
                "FR_thigh_joint",
                "RL_thigh_joint",
                "RR_thigh_joint",
                "FL_calf_joint",
                "FR_calf_joint",
                "RL_calf_joint",
                "RR_calf_joint",
            ]

            # Use the correct mapping from the existing config
            # JOINT_OBSERVATION_UNITREE_TO_ISAAC_LAB_MAPPING: [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
            self.mujoco_to_isaac_mapping = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
            self.isaac_to_mujoco_mapping = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

        elif self.robot_name == "g1":
            # MuJoCo order for G1 (simplified - you may need to adjust based on your G1 configuration)
            self.mujoco_joint_names = [
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

            # Isaac Lab order for G1
            self.isaac_lab_joint_names = [
                "left_hip_pitch_joint",
                "right_hip_pitch_joint",
                "waist_yaw_joint",
                "left_hip_roll_joint",
                "right_hip_roll_joint",
                "waist_roll_joint",
                "left_hip_yaw_joint",
                "right_hip_yaw_joint",
                "waist_pitch_joint",
                "left_knee_joint",
                "right_knee_joint",
                "left_shoulder_pitch_joint",
                "right_shoulder_pitch_joint",
                "left_ankle_pitch_joint",
                "right_ankle_pitch_joint",
                "left_shoulder_roll_joint",
                "right_shoulder_roll_joint",
                "left_ankle_roll_joint",
                "right_ankle_roll_joint",
                "left_shoulder_yaw_joint",
                "right_shoulder_yaw_joint",
                "left_elbow_joint",
                "right_elbow_joint",
                "left_wrist_roll_joint",
                "right_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "right_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_wrist_yaw_joint",
            ]
        else:
            raise ValueError(f"Unsupported robot: {self.robot_name}")

    def _setup_mappings(self):
        """Setup conversion mappings between coordinate systems."""
        if self.robot_name == "go2":
            # For Go2, use the hardcoded mappings from the existing config
            # These mappings are already validated and working
            pass  # Mappings are already set in _setup_joint_orders
        else:
            # For other robots, create mappings using the existing helper function
            from .helpers import create_joint_mapping

            self.mujoco_to_isaac_mapping = create_joint_mapping(self.mujoco_joint_names, self.isaac_lab_joint_names)
            self.isaac_to_mujoco_mapping = create_joint_mapping(self.isaac_lab_joint_names, self.mujoco_joint_names)

        # Convert to numpy arrays for efficient indexing
        self.mujoco_to_isaac_indices = np.array(self.mujoco_to_isaac_mapping)
        self.isaac_to_mujoco_indices = np.array(self.isaac_to_mujoco_mapping)

        # Validate mappings
        self._validate_mappings()

    def _validate_mappings(self):
        """Validate that all joints can be mapped between coordinate systems."""
        if -1 in self.mujoco_to_isaac_mapping or -1 in self.isaac_to_mujoco_mapping:
            missing_mujoco = [
                self.mujoco_joint_names[i] for i, idx in enumerate(self.mujoco_to_isaac_mapping) if idx == -1
            ]
            missing_isaac = [
                self.isaac_lab_joint_names[i] for i, idx in enumerate(self.isaac_to_mujoco_mapping) if idx == -1
            ]

            raise ValueError(
                f"Invalid joint mapping for {self.robot_name}:\n"
                f"Missing in Isaac Lab: {missing_mujoco}\n"
                f"Missing in MuJoCo: {missing_isaac}"
            )

    def convert_joint_positions(
        self, positions: Union[List[float], np.ndarray], from_order: JointOrder, to_order: JointOrder
    ) -> np.ndarray:
        """
        Convert joint positions between coordinate systems.

        Args:
            positions: Joint positions in the source coordinate system
            from_order: Source coordinate system
            to_order: Target coordinate system

        Returns:
            Joint positions in the target coordinate system
        """
        positions = np.asarray(positions)

        if from_order == to_order:
            return positions

        if from_order == JointOrder.MUJOCO and to_order == JointOrder.ISAAC_LAB:
            return positions[self.mujoco_to_isaac_indices]
        elif from_order == JointOrder.ISAAC_LAB and to_order == JointOrder.MUJOCO:
            return positions[self.isaac_to_mujoco_indices]
        else:
            raise ValueError(f"Unsupported conversion: {from_order} to {to_order}")

    def convert_joint_velocities(
        self, velocities: Union[List[float], np.ndarray], from_order: JointOrder, to_order: JointOrder
    ) -> np.ndarray:
        """
        Convert joint velocities between coordinate systems.

        Args:
            velocities: Joint velocities in the source coordinate system
            from_order: Source coordinate system
            to_order: Target coordinate system

        Returns:
            Joint velocities in the target coordinate system
        """
        return self.convert_joint_positions(velocities, from_order, to_order)

    def convert_joint_actions(
        self, actions: Union[List[float], np.ndarray], from_order: JointOrder, to_order: JointOrder
    ) -> np.ndarray:
        """
        Convert joint actions between coordinate systems.

        Args:
            actions: Joint actions in the source coordinate system
            from_order: Source coordinate system
            to_order: Target coordinate system

        Returns:
            Joint actions in the target coordinate system
        """
        return self.convert_joint_positions(actions, from_order, to_order)

    def get_default_positions(self, order: JointOrder) -> np.ndarray:
        """
        Get default joint positions in the specified coordinate system.

        Args:
            order: Target coordinate system

        Returns:
            Default joint positions
        """
        if self.robot_name == "go2":
            # Default positions in MuJoCo order
            mujoco_default = np.array(
                [
                    0.1,
                    0.8,
                    -1.5,  # FR_hip, FR_thigh, FR_calf
                    -0.1,
                    0.8,
                    -1.5,  # FL_hip, FL_thigh, FL_calf
                    0.1,
                    1.0,
                    -1.5,  # RR_hip, RR_thigh, RR_calf
                    -0.1,
                    1.0,
                    -1.5,  # RL_hip, RL_thigh, RL_calf
                ]
            )

            if order == JointOrder.MUJOCO:
                return mujoco_default
            elif order == JointOrder.ISAAC_LAB:
                return self.convert_joint_positions(mujoco_default, JointOrder.MUJOCO, JointOrder.ISAAC_LAB)
            else:
                raise ValueError(f"Unsupported order: {order}")

        elif self.robot_name == "g1":
            # Default positions in MuJoCo order (all zeros for G1)
            mujoco_default = np.zeros(len(self.mujoco_joint_names))

            if order == JointOrder.MUJOCO:
                return mujoco_default
            elif order == JointOrder.ISAAC_LAB:
                return self.convert_joint_positions(mujoco_default, JointOrder.MUJOCO, JointOrder.ISAAC_LAB)
            else:
                raise ValueError(f"Unsupported order: {order}")

        else:
            raise ValueError(f"Unsupported robot: {self.robot_name}")

    def get_joint_names(self, order: JointOrder) -> List[str]:
        """
        Get joint names in the specified coordinate system.

        Args:
            order: Target coordinate system

        Returns:
            List of joint names
        """
        if order == JointOrder.MUJOCO:
            return self.mujoco_joint_names.copy()
        elif order == JointOrder.ISAAC_LAB:
            return self.isaac_lab_joint_names.copy()
        else:
            raise ValueError(f"Unsupported order: {order}")

    def get_mapping_indices(self, from_order: JointOrder, to_order: JointOrder) -> np.ndarray:
        """
        Get the mapping indices for converting between coordinate systems.

        Args:
            from_order: Source coordinate system
            to_order: Target coordinate system

        Returns:
            Array of indices for reordering
        """
        if from_order == JointOrder.MUJOCO and to_order == JointOrder.ISAAC_LAB:
            return self.mujoco_to_isaac_indices
        elif from_order == JointOrder.ISAAC_LAB and to_order == JointOrder.MUJOCO:
            return self.isaac_to_mujoco_indices
        else:
            raise ValueError(f"Unsupported mapping: {from_order} to {to_order}")

    def print_joint_mapping_info(self):
        """Print detailed information about the joint mappings."""
        print(f"Joint Mapping Information for {self.robot_name.upper()}")
        print("=" * 50)

        print("\nMuJoCo Joint Order:")
        for i, name in enumerate(self.mujoco_joint_names):
            print(f"  {i:2d}: {name}")

        print("\nIsaac Lab Joint Order:")
        for i, name in enumerate(self.isaac_lab_joint_names):
            print(f"  {i:2d}: {name}")

        print("\nMuJoCo → Isaac Lab Mapping:")
        for i, (mujoco_name, isaac_idx) in enumerate(zip(self.mujoco_joint_names, self.mujoco_to_isaac_indices)):
            isaac_name = self.isaac_lab_joint_names[isaac_idx]
            print(f"  {i:2d}: {mujoco_name} → {isaac_idx:2d}: {isaac_name}")

        print("\nIsaac Lab → MuJoCo Mapping:")
        for i, (isaac_name, mujoco_idx) in enumerate(zip(self.isaac_lab_joint_names, self.isaac_to_mujoco_indices)):
            mujoco_name = self.mujoco_joint_names[mujoco_idx]
            print(f"  {i:2d}: {isaac_name} → {mujoco_idx:2d}: {mujoco_name}")


# Convenience functions for easy access
def create_joint_mapper(robot_name: str) -> JointMappingInterface:
    """
    Create a joint mapping interface for the specified robot.

    Args:
        robot_name: Name of the robot ("go2" or "g1")

    Returns:
        JointMappingInterface instance
    """
    return JointMappingInterface(robot_name)


def convert_positions(
    positions: Union[List[float], np.ndarray], robot_name: str, from_order: JointOrder, to_order: JointOrder
) -> np.ndarray:
    """
    Convenience function to convert joint positions.

    Args:
        positions: Joint positions to convert
        robot_name: Name of the robot
        from_order: Source coordinate system
        to_order: Target coordinate system

    Returns:
        Converted joint positions
    """
    mapper = create_joint_mapper(robot_name)
    return mapper.convert_joint_positions(positions, from_order, to_order)


def get_default_positions(robot_name: str, order: JointOrder) -> np.ndarray:
    """
    Convenience function to get default joint positions.

    Args:
        robot_name: Name of the robot
        order: Target coordinate system

    Returns:
        Default joint positions
    """
    mapper = create_joint_mapper(robot_name)
    return mapper.get_default_positions(order)

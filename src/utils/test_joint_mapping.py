#!/usr/bin/env python3
"""
Test script to demonstrate the joint mapping interface.

This script shows how to use the unified joint mapping interface to convert
between MuJoCo and Isaac Lab coordinate systems.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.joint_mapping import JointOrder, convert_positions, create_joint_mapper, get_default_positions


def test_go2_mapping():
    """Test joint mapping for Go2 robot."""
    print("Testing Go2 Joint Mapping")
    print("=" * 40)

    # Create mapper
    mapper = create_joint_mapper("go2")

    # Print mapping information
    mapper.print_joint_mapping_info()

    # Test with your stand_up_joint_pos (MuJoCo order)
    stand_up_joint_pos = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]

    print(f"\nOriginal stand_up_joint_pos (MuJoCo order):")
    for i, pos in enumerate(stand_up_joint_pos):
        print(f"  {i:2d}: {pos:6.3f} - {mapper.get_joint_names(JointOrder.MUJOCO)[i]}")

    # Convert to Isaac Lab order
    isaac_positions = mapper.convert_joint_positions(stand_up_joint_pos, JointOrder.MUJOCO, JointOrder.ISAAC_LAB)

    print(f"\nConverted to Isaac Lab order:")
    for i, pos in enumerate(isaac_positions):
        print(f"  {i:2d}: {pos:6.3f} - {mapper.get_joint_names(JointOrder.ISAAC_LAB)[i]}")

    # Convert back to MuJoCo order
    mujoco_positions = mapper.convert_joint_positions(isaac_positions, JointOrder.ISAAC_LAB, JointOrder.MUJOCO)

    print(f"\nConverted back to MuJoCo order:")
    for i, pos in enumerate(mujoco_positions):
        print(f"  {i:2d}: {pos:6.3f} - {mapper.get_joint_names(JointOrder.MUJOCO)[i]}")

    # Verify round-trip conversion
    if np.allclose(stand_up_joint_pos, mujoco_positions):
        print("\n✅ Round-trip conversion successful!")
    else:
        print("\n❌ Round-trip conversion failed!")
        print("Original:", stand_up_joint_pos)
        print("Converted back:", mujoco_positions)

    # Test default positions
    print(f"\nDefault positions in MuJoCo order:")
    mujoco_default = mapper.get_default_positions(JointOrder.MUJOCO)
    for i, pos in enumerate(mujoco_default):
        print(f"  {i:2d}: {pos:6.3f} - {mapper.get_joint_names(JointOrder.MUJOCO)[i]}")

    print(f"\nDefault positions in Isaac Lab order:")
    isaac_default = mapper.get_default_positions(JointOrder.ISAAC_LAB)
    for i, pos in enumerate(isaac_default):
        print(f"  {i:2d}: {pos:6.3f} - {mapper.get_joint_names(JointOrder.ISAAC_LAB)[i]}")


def test_g1_mapping():
    """Test joint mapping for G1 robot."""
    print("\n\nTesting G1 Joint Mapping")
    print("=" * 40)

    # Create mapper
    mapper = create_joint_mapper("g1")

    # Print mapping information
    mapper.print_joint_mapping_info()

    # Test with some sample positions (MuJoCo order)
    sample_positions = np.random.rand(len(mapper.get_joint_names(JointOrder.MUJOCO)))

    print(f"\nSample positions (MuJoCo order):")
    for i, pos in enumerate(sample_positions):
        print(f"  {i:2d}: {pos:6.3f} - {mapper.get_joint_names(JointOrder.MUJOCO)[i]}")

    # Convert to Isaac Lab order
    isaac_positions = mapper.convert_joint_positions(sample_positions, JointOrder.MUJOCO, JointOrder.ISAAC_LAB)

    print(f"\nConverted to Isaac Lab order:")
    for i, pos in enumerate(isaac_positions):
        print(f"  {i:2d}: {pos:6.3f} - {mapper.get_joint_names(JointOrder.ISAAC_LAB)[i]}")

    # Convert back to MuJoCo order
    mujoco_positions = mapper.convert_joint_positions(isaac_positions, JointOrder.ISAAC_LAB, JointOrder.MUJOCO)

    # Verify round-trip conversion
    if np.allclose(sample_positions, mujoco_positions):
        print("\n✅ Round-trip conversion successful!")
    else:
        print("\n❌ Round-trip conversion failed!")


def test_convenience_functions():
    """Test convenience functions."""
    print("\n\nTesting Convenience Functions")
    print("=" * 40)

    # Test convert_positions function
    positions = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]

    converted = convert_positions(positions, "go2", JointOrder.MUJOCO, JointOrder.ISAAC_LAB)
    print(f"Converted positions using convenience function:")
    for i, pos in enumerate(converted):
        print(f"  {i:2d}: {pos:6.3f}")

    # Test get_default_positions function
    default_mujoco = get_default_positions("go2", JointOrder.MUJOCO)
    default_isaac = get_default_positions("go2", JointOrder.ISAAC_LAB)

    print(f"\nDefault positions (MuJoCo): {default_mujoco}")
    print(f"Default positions (Isaac Lab): {default_isaac}")


def compare_with_existing_config():
    """Compare with existing configuration values."""
    print("\n\nComparing with Existing Configuration")
    print("=" * 40)

    # Your stand_up_joint_pos in MuJoCo order
    stand_up_joint_pos = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]

    # Isaac Lab default positions from config
    isaac_lab_default = [
        0.1000,
        -0.1000,
        0.1000,
        -0.1000,  # hips
        0.8000,
        0.8000,
        1.0000,
        1.0000,  # thighs
        -1.5000,
        -1.5000,
        -1.5000,
        -1.5000,  # calves
    ]

    mapper = create_joint_mapper("go2")

    # Convert your positions to Isaac Lab order
    converted_to_isaac = mapper.convert_joint_positions(stand_up_joint_pos, JointOrder.MUJOCO, JointOrder.ISAAC_LAB)

    print("Your stand_up_joint_pos converted to Isaac Lab order:")
    for i, pos in enumerate(converted_to_isaac):
        print(f"  {i:2d}: {pos:6.3f} - {mapper.get_joint_names(JointOrder.ISAAC_LAB)[i]}")

    print("\nIsaac Lab default positions from config:")
    for i, pos in enumerate(isaac_lab_default):
        print(f"  {i:2d}: {pos:6.3f} - {mapper.get_joint_names(JointOrder.ISAAC_LAB)[i]}")

    # Check if they match
    if np.allclose(converted_to_isaac, isaac_lab_default):
        print("\n✅ Your positions match the Isaac Lab config!")
    else:
        print("\n❌ Your positions do NOT match the Isaac Lab config!")
        print("Differences:")
        for i, (converted, config) in enumerate(zip(converted_to_isaac, isaac_lab_default)):
            if not np.isclose(converted, config):
                print(f"  {i:2d}: {converted:6.3f} vs {config:6.3f} (diff: {converted-config:6.3f})")


if __name__ == "__main__":
    test_go2_mapping()
    # test_g1_mapping()
    test_convenience_functions()
    compare_with_existing_config()

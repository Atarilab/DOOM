# G1 Robot Visualization

This directory contains the G1 robot description and visualization files.

## Quick Start

To visualize the G1 robot in RViz, you can use the launch file:

```bash
# From the workspace root
ros2 launch src/robots/g1/g1_visualization.launch.py

# Or from the g1 directory
cd src/robots/g1
ros2 launch g1_visualization.launch.py
```

## Files

- `g1_29dof_lock_waist_with_hand_rev_1_0.urdf` - The original URDF file for the G1 robot
- `g1_29dof_lock_waist_with_hand_rev_1_0_absolute.urdf` - URDF file with absolute mesh paths (used by default)
- `g1_visualization.launch.py` - Launch file for visualization
- `g1_description/` - Package structure for the G1 description (contains meshes and config)

## Launch Arguments

- `description_file`: URDF file to use (default: g1_29dof_lock_waist_with_hand_rev_1_0.urdf)
- `prefix`: Prefix for robot frames (default: "")
- `use_sim_time`: Use simulation time (default: True)

## Example Usage

```bash
# Use a different URDF file
ros2 launch g1_visualization.launch.py description_file:=g1_29dof_lock_waist_rev_1_0.urdf

# Add a prefix to robot frames
ros2 launch g1_visualization.launch.py prefix:=g1_

# Use real time instead of simulation time
ros2 launch g1_visualization.launch.py use_sim_time:=False
```

## Dependencies

- `robot_state_publisher`
- `rviz2`
- `xacro` (for processing URDF files)

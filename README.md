# DOOM
Repository for running the experiments on the real robots.

## Requirements 
- docker (ros2 container with unitree_sdk for Go2, ros container with unitree_legged_sdk for AlienGo)
- conda (used for setting environment variables, creating aliases etc.)
  
## Installation Instructions
```bash
./doom.sh
```

## Running the container
```bash
dungeon
```

## TODO
- [ ] ROS container with unitree_legged_sdk for AlienGo
- [ ] Get vicon frame from Vicon SDK and transform to robot base
- [ ] Implement safety mechanisms (soft dof pos limits, dof torque limits)
- [ ] Test Velocity-conditioned policy using joystick


## Resources
Unitree Guide: https://support.unitree.com/home/en/developer/Quick_start

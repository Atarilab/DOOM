# DOOM
Repository for running the experiments on the real robots.

## Requirements 
- docker (ros2 container with unitree_sdk for Go2, ros container with unitree_legged_sdk for AlienGo)
- conda (used for setting environment variables, creating aliases etc.)
- unitree Go2 
  
## Installation Instructions
```bash
./doom.sh
```

## Running the container
```bash
dungeon
```
> **Warning:** Before you run the example scripts provided by Unitree from unitree_sdk, make sure to turn off the sports mode using the Go2 app. To do this, toggle off **Device > Service Status > sport_mode** as this will interfere with the additional torque commands passed to the robot.


Once inside the container, you can run the python scripts by:
```bash
python3 filename.py $NETWORK_INTERFACE
```
For example,
```bash
python3 read_lowstate.py $NETWORK_INTERFACE
```

and the C++ scripts by:
```bash
sudo ./filename $NETWORK_INTERFACE
```
Note that `$NETWORK_INTERFACE` is already setup when you install using `./doom.sh -i` and is necessary to identify the robot through the network.

## TODO
- [ ] ROS container with unitree_legged_sdk for AlienGo
- [ ] Get vicon frame from Vicon SDK and transform to robot base
- [ ] Implement safety mechanisms (soft dof pos limits, dof torque limits)
- [ ] Test Velocity-conditioned policy using joystick


## Resources
Unitree Guide: https://support.unitree.com/home/en/developer/Quick_start

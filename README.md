# DOOM
Repository for running the experiments on the real robots.

## Requirements 
- docker (ros2 container with unitree_sdk for Go2, ros container with unitree_legged_sdk for AlienGo)
- conda (used for setting environment variables, creating aliases etc.)
- unitree Go2 
  
## Installation Instructions
```bash
./doom.sh -i
```

## Running the container
```bash
dungeon
```
> **Warning:** Before you run the example scripts provided by Unitree from unitree_sdk, make sure to turn off the sports mode using the Go2 app. To do this, toggle off **Device > Service Status > sport_mode** as this will interfere with the additional torque commands passed to the robot.

## Testing Robot Connection
Once inside the conda env, you can access the Robot's IP address via `$ROBOT_IP`. You can test the connection using:
```bash
ping $ROBOT_IP
```
If the connection is not established, you might need to manually set the IP for the wired connection. You can do so by following the "Configure Network Environment" section [here](https://support.unitree.com/home/en/developer/Quick_start).

---

## Running the scripts
Once inside the container and the connection to the robot has been established, you can run the python scripts by:
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

#### Tested example scripts from `unitree_sdk2_python`
`read_lowstate.py`
`go2_stand_example.py`

## Vicon State Estimation
The vicon receiver client is already installed in the docker container. You can simply launch it using:
```bash
ros2 launch vicon_receiver client.launch.py
```

## TODO
- [ ] ROS container with unitree_legged_sdk for AlienGo
- [ ] Get vicon frame from Vicon SDK and transform to robot base
- [ ] Implement safety mechanisms (soft dof pos limits, dof torque limits)
- [ ] Test Velocity-conditioned policy using joystick


## Resources
Unitree Guide: https://support.unitree.com/home/en/developer/Quick_start

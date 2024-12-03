# DOOM
Repository for running the experiments on the real robots.

## Requirements 
- docker (ros2 container with unitree_sdk)
- unitree Go2 
  
## Installation Instructions
For installation of the DOOM project, run:
```bash
./doom.sh -i
```
The above script, updates all submodules, builds the docker container, and manually sets up a network interface in the same subnetwork as the robot.

## Building and running the container
```bash
./doom.sh -b # build
./doom.sh -e # enter
```

For more helpful functions from `./doom.sh`, run:
```bash
./doom.sh -h
```

> **Warning:** Before you run the example scripts provided by Unitree from unitree_sdk, make sure to turn off the sports mode using the Go2 app. Log in to the app using the ATARI Gmail credentials and toggle off **Device > Service Status > sport_mode** as this will interfere with the additional torque commands passed to the robot. 

## Testing Robot Connection
Once inside the docker container, you can access the Robot's IP address via `$ROBOT_IP`. You can test the connection using:
```bash
ping $ROBOT_IP
```
If the connection is not established, you might need to manually set the IP for the wired connection. You can do so by following the "Configure Network Environment" section [here](https://support.unitree.com/home/en/developer/Quick_start).

---

## VS Code Workspace Setup
Open VS Code with `unitree_mujoco_container` as the project directory and build the docker container.

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

## Simulation Tests
This repository also has a simulation mode which allows you to run your scripts with the `unitree_sdk` to send commands to your robot in MuJoCo. 
To launch the simulator, run:
```bash
python3 simulate.py --task custom-task-name --log test
```
The various tasks are defined in `tasks/task_configs.json`.

Once the simulator is up and running, you can try out the stand up controller using:
```bash
python3 stand_go2.py --task custom-task-name
```

## Convert IDL Messages to Python for DDS Communication
idlc -l py -o output_dir file.idl


## Installed ROS2 Packages
- [unitree_sdk](https://github.com/unitreerobotics/unitree_sdk2)
- [unitree_sdk_python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [ros2-vicon-receiver](https://github.com/Atarilab/ros2-vicon-receiver.git)

## TODO
- [ ] Add support for AlienGo
- [x] Get vicon frame from Vicon SDK and transform to robot base
- [ ] Adds mechanism for state logger and plotter
- [ ] Implement safety mechanisms (soft dof pos limits, dof torque limits)
- [ ] Test Velocity-conditioned policy using joystick


## Resources
Unitree Guide: https://support.unitree.com/home/en/developer/Quick_start

# DOOM
Repository for running the experiments on the real robots.

## Requirements 
- docker (ros2 container with unitree_sdk)
- unitree Go2
- nvidia Graphics card and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  
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

> **Warning:** Before you run anything on the robot, make sure to turn off the sports mode using the Go2 app. Log in to the app using the ATARI Gmail credentials and toggle off **Device > Service Status > sport_mode** as this will interfere with the additional torque commands passed to the robot. 

## Testing Robot Connection
Once inside the docker container, you can access the Robot's IP address via `$ROBOT_IP`. You can test the connection using:
```bash
ping $ROBOT_IP
```
If the connection is not established, you might need to manually set the IP for the wired connection. You can do so by following the "Configure Network Environment" section [here](https://support.unitree.com/home/en/developer/Quick_start).

---

## VS Code Workspace Setup
Open VS Code with `unitree_mujoco_container` as the project directory and build the docker container.
Additionally, after the build, the debugger can also be setup using `Ctrl+Shift+D > Create launch.json`. `launch.json` can be setup using command line args for task and log.
"""

## How to use DOOM to control your robot
The various tasks are defined in `tasks/task_configs.json`. Currently, the following tasks are defined and tested:
- `rl-velocity-sim-go2` (Status: ✅ )
- `rl-velocity-real-go2` (Status: ✅ )
  
Once you've chosen the task you want to run, you can launch the user interface to control the robot using:
```bash
ros2 run master_manager master_node --task custom-task-name --log log_name
```
This repository also has a simulation mode which allows you to run the same scripts with the `unitree_sdk` to send commands to your robot in MuJoCo. Note that MuJoCo is not used as a visualizer for your real robot interface but rather as a sanity test of the same script that you might run on the real robot.
To launch the simulator, run:
```bash
python3 simulate.py --task custom-task-name --log log_name
```
Example Workflow: `Standing` > `Stay_down` > `Stand_up` > `Back to Main Menu` > `RL-Velocity` > `RL-Velocity`


## Vicon State Estimation
The vicon receiver client is already installed in the docker container. You can simply launch it using:
```bash
ros2 launch vicon_receiver client.launch.py
```

## Live Plotting using PlotJuggler
```bash
ros2 run plotjuggler plotjuggler
```

## Installed ROS2 Packages
- [unitree_sdk](https://github.com/unitreerobotics/unitree_sdk2)
- [unitree_sdk_python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [ros2-vicon-receiver](https://github.com/Atarilab/ros2-vicon-receiver.git)

## TODO
- [ ] Implement a real-time High-Level Contact Planner 
- [ ] Integrate Pinocchio for additional states such as feet positions
- [ ] Add support for AlienGo
- [ ] Implement safety mechanisms (soft dof pos limits, dof torque limits)
- [ ] Implement joystick control for velocity commands
- [x] Get vicon frame from Vicon SDK and transform to robot base
- [x] Add mechanism for real-time state logger and plotter
- [x] Test Velocity-conditioned policy

## Known Issues
When using torque control for low-level control, there is a delay (latency), which causes the robot to behave unexpectedly. This could be resolved by training with delayed actuation of joints. However, position control generally seems to be a more stable and recommended approach to sending low-level commands to the robot.


## Resources
Unitree Guide: https://support.unitree.com/home/en/developer/Quick_start

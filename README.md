![ATARI DOOM Banner](assets/banner.jpg)
# DOOM
Repository for running the experiments on the real robots and simulate them on Mujoco using a common interface.

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
> **Check:** Unitree ros topics should appear by default with 'ros2 topic list'. If not, try restarting your computer.

## Testing Robot Connection
Once inside the docker container, you can access the Robot's IP address via `$ROBOT_IP`. You can test the connection using:
```bash
ping $ROBOT_IP
```
If the connection is not established, you might need to manually set the IP for the wired connection. You can do so by following the "Configure Network Environment" section [here](https://support.unitree.com/home/en/developer/Quick_start).

If it still doesn't ping the robot after manually configuring the IP, you can check if the right network interface is chosen. Using the USB-Ethernet Adapter, the network interface should have the ID `enx3c4937046061` by default. You can confirm it using `ifconfig` and checking the network interface ID for the corresponding `$ROBOT_IP`. If not, you should manually change it in `.env.base`, `.env.docker`, then delete the existing container using `./doom.sh -d`, rebuild the container using `./doom.sh -b`, enter inside the container using `./doom.sh -e`, and update the ROS/DDS network interface inside `setup.sh` using the one you found with `ifconfig`. Don't forget to run `source setup.sh` to update them.

---

## VS Code Workspace Setup
Open VS Code with `unitree_mujoco_container` as the project directory and build the docker container. Additionally, debuggers for certain tasks are already defined in `.vscode/launch.json`.

## How to use DOOM to control your robot
The various tasks are defined in `tasks/task_configs.json`. Currently, the following tasks are defined and tested:
- `rl-velocity-sim-go2` (Status: ✅ )
- `rl-velocity-real-go2` (Status: ✅ )
- `rl-contact-sim-go2` (Status: ✅ )
- `rl-contact-real-go2` (Status: ✅ )
- `rl-velocity-sim-g1` (Status: ⚠️ ) [WIP]: Stand Controllers work, not the policy
  
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

## Go2 Blind Locomotion using UI Velocity Commands Example (SIM)
Before you start, make sure there are no other main processes running on your computer. This could cause jittery movements due to imperfect tracking of the PD controllers introduced by the latency from heavy process in the background. Keep an eye out for your CPU utilisation using `htop` to validate this. A simple restart can ensure that you start fresh. 

#### Terminal 1
```bash
./doom -e # enter the container
cd src/
python3 simulate.py --task=rl-velocity-sim-go2
```
#### Terminal 2
```bash
./doom -a # attach a terminal to the existing DOOM container
source setup_local.sh
ros2 run master_manager master_node --task rl-velocity-sim-go2 --enable-ui # use enable-ui for the terminal UI interface (additionally, we can also manage commands to the robot via joystick with/without UI)
```
IDLE mode is a damping mode to gracefully stop commands to the robot.
STAND modes are used to initialise the robot to its default positions using simple PD controllers. Click on `STAND`, and then `STAY_DOWN`. This makes the robot stay in a crouched position close to the ground. After it stabilises, click on `STAND_UP`, which is a phase-based PD controller that makes the robot stand up to the default joint configuration. `STAND_DOWN` is also a phase-based PD controller that moves from the standing up joint configuration to the crouched joint position, as in `STAY_DOWN`.

> Note: Since these `STAND` modes are phase-based PD controllers, allow them to stabilise before switching to other modes.

> Note: The UI terminal window may need to be resized to see the different modes. If using [Terminator](https://gnome-terminator.org/), you can use `Ctrl+Shift+X` for full screen and `Ctrl + MouseScroll` to adjust the window dimensions to fit your screen

Now that the robot is in the `STAND_UP` configuration, go back to the Main Menu from the UI, choose `LOCOMOTION`, and then `RL-VELOCITY`. This will start the velocity command based blind RL locomotion policy at zero velocity. From the UI, you can now enter the X, Y velocities and the yaw rates for fixed command velocities.

## Go2 Blind Locomotion using UI Velocity Commands Example (REAL)
The only change for running on the real robot is launching the Vicon client to get the base states instead of launching the simulator, and using the correct task name for the real experiments. The rest is taken care of by DOOM to maintain the same interface to run experiments in sim (MuJoCo) and on the real robot.

#### Terminal 1
Launch Vicon client that publishes the base states
```bash
./doom -e # enter the container
ros2 launch vicon_receiver client.launch.py
```

#### Terminal 2
```bash
./doom -a # attach a terminal to the existing DOOM container
source setup.sh
ros2 topic list # view available topics, confirm if you can view topics published by the robot and by the vicon
ros2 run master_manager master_node --task rl-velocity-real-go2 --enable-ui # use enable-ui for the terminal UI interface (additionally, we can also manage commands to the robot via joystick with/without UI)
```

#### Terminal 3 (optional)
Use plotjuggler to view topics in real time plots
```bash
./doom -a # attach a terminal to the existing DOOM container
source setup.sh
ros2 run plotjuggler plotjuggler
```

#### Terminal 4 (optional)
Robot Visualization in RViz
```bash
./doom -a # attach a terminal to the existing DOOM container
source setup.sh
ros2 launch go2_description go2_visualization.launch.py
```

## Joystick 
Alternatively, to send commands to the robot, users can also use the joystick. Make sure that the Docker container is launched with the joystick already connected.
There are some common joystick configurations to handle the mode-switches:

Start: `IDLE` -> `STAY_DOWN`

Start: AnyElse -> `IDLE`

Up: `STAY_DOWN`/`STAND_DOWN` -> `STAND_UP`

Down: `STAND_UP` -> `STAND_DOWN`

Down: `STAND_DOWN` -> `STAY_DOWN`

L1+R1: `STAND_UP` -> `RL-CONTACT` 

You can further define controller-specific joystick mappings by defining `get_joystick_mappings` (See RL-Contact Controller for reference).


## Vicon State Estimation
The Vicon receiver client is already installed in the docker container. You can launch it in a new terminal inside existing container (`./doom -a`) using:
```bash
ros2 launch vicon_receiver client.launch.py
```

## Live Plotting using PlotJuggler
```bash
ros2 run plotjuggler plotjuggler
```

## Robot Visualization in RViz
Visualizing in RViz is useful, especially for debugging, to see what the world looks like to the robot. Once the master node is launched in a terminal, you can launch RViz in another terminal using:
```bash
ros2 launch go2_description go2_visualization.launch.py
```

## Code Formatting
This project uses [black](https://github.com/psf/black) as the code formatter and [flake7](https://github.com/PyCQA/flake8) as additional linter to ensure consistent code style across the codebase. You can run the formatter inside the container in VS Code (or other forks like Cursor, Windsurf etc.) using the keyboard shortcut `Ctrl + Shift + B` and running the task `Format and Lint`.

## Installed ROS2 Packages
- [unitree_sdk](https://github.com/unitreerobotics/unitree_sdk2)
- [unitree_sdk_python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [ros2-vicon-receiver](https://github.com/Atarilab/ros2-vicon-receiver.git)

## TODO
- [ ] Test g1 velocity locomotion Policy
- [ ] Test g1 bimanual manipulation policy
- [ ] Add support for AlienGo/Allegro
- [x] Create table with box env for G1
- [x] Implement simple stand up, stand down controllers for G1
- [x] Add G1 in sim
- [x] Optional Terminal UI Interface
- [x] Test RViz on the real robot for contact-conditioned policy (stance, trot in place)
- [x] Implement joystick control for controller-specific commands
- [x] Implement joystick control for switching between common modes
- [x] Implement a real-time High-Level Contact Planner
- [x] Publish robot and joint states to corresponding topics from master node
- [x] Visualize Robot in RViz
- [x] Implement interface to change commands from UI
- [x] Implement safety mechanisms (soft dof pos limits, dof torque limits)
- [x] Get vicon frame from Vicon SDK and transform to robot base (directly using base position from vicon after [**@victorDD1**](https://github.com/victorDD1)'s update)
- [x] Add mechanism for real-time state logger and plotter (debug logger in console and file, plotjuggler for ros topics)
- [x] Test Velocity-conditioned policy

## Known Issues
- When using torque control for low-level control, there is a delay (latency), which causes the robot to behave unexpectedly. This could be resolved by training with delayed actuation of joints. However, position control generally seems to be a more stable and recommended approach to sending low-level commands to the robot.
- 

## Resources
Unitree Guide: https://support.unitree.com/home/en/developer/Quick_start

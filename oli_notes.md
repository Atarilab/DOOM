# Turn of robot sports mode using the app

cd DOOM
colcon build
source install/setup.bash


# connect to robot and vicon using ethernet / usb cable 
# if not connected to icon: comment out subscriper (else doesnt work)

For sim:
python3 simulate.py --task rl-velocity-sim-go2 --log test # start sim
ros2 run master_manager master_node --task rl-velocity-sim-go2 --log test # start controller with config


For real:
ros2 run master_manager master_node --task rl-velocity-real-go2 --log test # start controller with config
# Script for Vicon Datastream client installation

cd ros2-vicon-receiver/
bash ./install_libs.sh
colcon build --symlink-install
source install/setup.bash
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Define launch arguments
    task = LaunchConfiguration("task", default="rl-velocity-sim-go2")
    log = LaunchConfiguration("log", default="test")
    run_simulation = LaunchConfiguration("sim", default="true")

    # Master manager node
    master_manager_node = Node(
        package="master_manager",
        executable="master_node",
        name="robot_controller",
        parameters=[],
        arguments=["--task", task, "--log", log],
        output="screen",
    )

    # Simulation node (conditionally launched)
    simulation_node = ExecuteProcess(
        condition=IfCondition(run_simulation),
        cmd=[
            "python3",
            os.path.join(os.getenv("HOME"), "/home/atari/workspace/DOOM/src/simulate.py"),
            "--task",
            task,
            "--log",
            log,
        ],
        output="screen",
    )

    # Create the launch description and add actions
    ld = LaunchDescription(
        [
            # Declare launch arguments
            simulation_node,
            master_manager_node,
        ]
    )

    return ld

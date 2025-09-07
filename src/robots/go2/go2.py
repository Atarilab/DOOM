from typing import TYPE_CHECKING, Dict, List, Type

from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as Go2LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as Go2LowState_

from controllers.rl_contact_locomotion_controller import RLQuadrupedLocomotionContactController
from controllers.rl_velocity_locomotion_controller import RLQuadrupedLocomotionVelocityController
from controllers.stand_controller import (
    Go2StandDownController,
    Go2StandUpController,
    Go2StayDownController,
    Go2StanceController,
)
from robots.robot_base import RobotBase
from state_manager.msg_handlers import go2_low_state_handler, vicon_handler, sport_mode_state_handler
from state_manager.state_manager import DDSStateSubscriber, ROS2StateSubscriber
from utils.joint_mapping import JointMappingInterface

if TYPE_CHECKING:
    from controllers.controller_base import ControllerBase


class Go2(RobotBase):
    """
    This class provides robot-specific data and available controllers for the Go2 robot based on the desired task.
    """

    def __init__(self, task, logger, device="cuda:0"):
        """
        Initialize the Go2 robot.

        Args:
            task (str): The task to be performed by the robot.
            logger (logging.Logger): The logger to be used for logging.
        """
        super().__init__(task=task, logger=logger, device=device)
        self.joint_mapper = JointMappingInterface("go2")

        # Keep backward compatibility with existing mapping arrays
        self.joints_unitree2isaac = self.joint_mapper.mujoco_to_isaac_mapping
        self.joints_isaac2unitree = self.joint_mapper.isaac_to_mujoco_mapping

        self.stand_down_joint_pos = [
            0.0473455,
            1.22187,
            -2.44375,
            -0.0473455,
            1.22187,
            -2.44375,
            0.0473455,
            1.22187,
            -2.44375,
            -0.0473455,
            1.22187,
            -2.44375,
        ]
        self.stand_up_joint_pos = [
            0.1,
            0.8,
            -1.5,
            -0.1,
            0.8,
            -1.5,
            0.1,
            1.0,
            -1.5,
            -0.1,
            1.0,
            -1.5,
        ]

    @property
    def name(self):
        """
        Returns the name of the robot.

        Returns:
            str: The name of the robot.
        """
        return "UnitreeGo2"

    @property
    def ee_names(self):
        """
        Returns the names of the feet of the robot.

        Returns:
            List[str]: The names of the feet of the robot.
        """
        return ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    @property
    def actuated_joint_names(self):
        """
        Returns the names of the joints of the robot.

        Returns:
            List[str]: The names of the joints of the robot.
        """
        return [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]

    @property
    def non_actuated_joint_names(self):
        """
        Returns the names of the non-actuated joints of the robot.
        """
        return []

    def isaaclab_joint_names(self): 
        """
        Returns the names of the joints as in IsaacLab order.
        """
        return [
            "FL_hip_joint",
            "FR_hip_joint",
            "RL_hip_joint",
            "RR_hip_joint",
            "FL_thigh_joint",
            "FR_thigh_joint",
            "RL_thigh_joint",
            "RR_thigh_joint",
            "FL_calf_joint",
            "FR_calf_joint",
            "RL_calf_joint",
            "RR_calf_joint",
        ]


    @property
    def base_link(self):
        """Return the name of the base link."""
        return "base_link"

    @property
    def floating_base(self):
        """Return if the robot has a floating base."""
        return True
    

    def init_low_cmd(self, cmd_msg):
        """Initialize low-level command message with Go2-specific defaults."""
        # Set header and flags
        cmd_msg.head[0] = 0xFE
        cmd_msg.head[1] = 0xEF
        cmd_msg.level_flag = 0xFF
        cmd_msg.gpio = 0

        # Initialize all motor commands
        for i in range(len(cmd_msg.motor_cmd)):
            cmd_msg.motor_cmd[i].mode = 0x01  # PMSM mode
            cmd_msg.motor_cmd[i].q = 0.0
            cmd_msg.motor_cmd[i].kp = 0.0
            cmd_msg.motor_cmd[i].dq = 0.0
            cmd_msg.motor_cmd[i].kd = 0.0
            cmd_msg.motor_cmd[i].tau = 0.0

    def motor_command_attributes(self) -> List[str]:
        """Get the list of motor command attributes supported by Go2."""
        return ["q", "kp", "dq", "kd", "tau"]

    def update_motor_command(self, cmd_msg, motor_idx: int, motor_data: Dict):
        """Update a specific motor command with Go2-specific logic."""
        # Update standard attributes
        for attr in ["q", "kp", "dq", "kd", "tau"]:
            if attr in motor_data:
                setattr(cmd_msg.motor_cmd[motor_idx], attr, motor_data[attr])

    def update_command_modes(self, cmd_msg, motor_commands: Dict):
        """Update Go2-specific command mode settings."""
        # Go2 doesn't have mode_pr or mode_machine settings
        return cmd_msg


    def get_mode_initialization_state(self, combined_state: Dict) -> bool:
        """Go2 mode initialization is always complete."""
        if combined_state.get("robot/joint_pos", None) is not None:
            return True
        else:
            return False
    
    @property
    def damping_gain(self):
        """Returns the gain used for the damping mode

        Returns:
            float: Kd value used for damping gain
        """
        
        return 2.0

    @property
    def default_joint_positions(self):
        """
        Returns the default joint positions of the robot.

        Returns:
            List[float]: The default joint positions of the robot.
        """
        return [0.0] * self.num_joints

    @property
    def effort_limit(self):
        """
        Returns the effort limit of the robot.

        Returns:
            float: The effort limit of the robot.
        """
        return 23.5

    @property
    def xml_path(self):
        """
        Returns the path to the XML file of the robot.

        Returns:
            str: The path to the XML file of the robot.
        """
        return "/home/atari/workspace/DOOM/src/robots/go2/go2.xml"
    
    @property
    def low_cmd_msg(self):
        """Return the low command message class."""
        return unitree_go_msg_dds__LowCmd_

    @property
    def low_cmd_msg_type(self):
        """Return the low command message type."""
        return Go2LowCmd_

    @property
    def available_controllers(self) -> "Dict[str, Dict[str, Type[ControllerBase]]]":
        """
        Returns a dictionary of available controllers for the Go2 robot based on the desired task.

        The dictionary contains the following keys:

        - "STAND": a dictionary of available stand controllers.
        - "LOCOMOTION": a dictionary of available locomotion controllers.

        The values of the dictionary are dictionaries containing the following keys:

        - "STAY_DOWN": the Go2StayDownController class.
        - "STAND_UP": the Go2StandUpController class.
        - "STAND_DOWN": the Go2StandDownController class.
        - "RL-CONTACT": the RLQuadrupedLocomotionContactController class.
        - "RL-VELOCITY": the RLQuadrupedLocomotionVelocityController class.

        The available controllers are determined by the task.

        Returns:
            Dict[str, Dict[str, Type[ControllerBase]]]: A dictionary of available controllers.
        """
        controllers = {}
        if "contact" in self.task:
            controllers = {
                "STAND": {
                    "STAY_DOWN": Go2StayDownController,
                    "STAND_UP": Go2StandUpController,
                    "STAND_DOWN": Go2StandDownController,
                    "STANCE": Go2StanceController,
                },
                "LOCOMOTION": {
                    "RL-CONTACT": RLQuadrupedLocomotionContactController,
                },
            }
        elif "velocity" in self.task:
            controllers = {
                "STAND": {
                    "STAY_DOWN": Go2StayDownController,
                    "STAND_UP": Go2StandUpController,
                    "STAND_DOWN": Go2StandDownController,
                },
                "LOCOMOTION": {
                    "RL-VELOCITY": RLQuadrupedLocomotionVelocityController,
                },
            }
        else:
            controllers = {}
        return controllers

    @property
    def subscribers(self) -> Dict[str, ROS2StateSubscriber | DDSStateSubscriber]:
        """
        Returns a dictionary of state subscribers for the Go2 robot.

        The dictionary contains the following keys:

        - "low_state": a DDSStateSubscriber for the low-level state of the robot.
        - "sports_mode_state": a DDSStateSubscriber for the sports mode state of the robot in simulation.
        - "vicon_robot_state": a ROS2StateSubscriber for the Vicon state of the robot in the real world.

        The subscribers are initialized with the corresponding handler functions and topics.

        Returns:
            Dict[str, ROS2StateSubscriber | DDSStateSubscriber]: A dictionary of state subscribers.
        """
        _subscribers = {}

        dds_low_state_sub = DDSStateSubscriber(
            topic="rt/lowstate",
            msg_type=Go2LowState_,
            handler_func=go2_low_state_handler,
            handler_args={"device": self.device} if self.device else {},
            logger=self.logger,
        )
        _subscribers["low_state"] = dds_low_state_sub

        if "sim" in self.task:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

            dds_sportsmode_state_sub = DDSStateSubscriber(
                topic="rt/sportmodestate",
                msg_type=SportModeState_,
                handler_func=sport_mode_state_handler,
                handler_args={"device": self.device} if self.device else {},
                logger=self.logger,
            )
            _subscribers["sports_mode_state"] = dds_sportsmode_state_sub
        else:
            from vicon_receiver.msg import Position

            ros2_vicon_sub = ROS2StateSubscriber(
                topic="/vicon/Go2/Go2",
                node_name="vicon_robot_state",
                msg_type=Position,
                handler_func=vicon_handler,
                handler_args={"device": self.device} if self.device else {},
                logger=self.logger,
            )
            _subscribers["vicon_robot_state"] = ros2_vicon_sub

        return _subscribers

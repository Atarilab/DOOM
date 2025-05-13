from typing import Dict, TYPE_CHECKING

from robots.robot_base import RobotBase
from utils.mj_wrapper import MjRobotWrapper
from controllers.stand_controller import (
    Go2StayDownController,
    Go2StandUpController,
    Go2StandDownController,
)
from controllers.rl_contact_controller import RLLocomotionContactController
from controllers.rl_controller import RLLocomotionVelocityController

from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as Go2LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as Go2LowCmd_

from state_manager.msg_handlers import go2_low_state_handler, sport_mode_state_handler, go2_vicon_handler
from state_manager.state_manager import DDSStateSubscriber, ROS2StateSubscriber

if TYPE_CHECKING:
    from controllers.controller_base import ControllerBase

class Go2(RobotBase):
    """
    This class provides robot-specific data and available controllers for the Go2 robot based on the desired task.
    """
    def __init__(self, task, logger):
        """
        Initialize the Go2 robot.

        Args:
            task (str): The task to be performed by the robot.
            logger (logging.Logger): The logger to be used for logging.
        """
        super().__init__(task=task, logger=logger)
        self.mj_model = MjRobotWrapper(self.xml_path, self.feet_names)
        self.low_cmd_msg = unitree_go_msg_dds__LowCmd_
        self.low_cmd_msg_type = Go2LowCmd_
        
        self.stand_down_joint_pos = [ 0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, ]
        self.stand_up_joint_pos = [ 0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5, ]

    @property
    def name(self):
        """
        Returns the name of the robot.

        Returns:
            str: The name of the robot.
        """
        return "Go2"
    
    @property
    def feet_names(self):
        """
        Returns the names of the feet of the robot.

        Returns:
            List[str]: The names of the feet of the robot.
        """
        return ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    @property
    def joint_names(self):
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
    def num_joints(self):
        """
        Returns the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        return len(self.joint_names)

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
        - "RL-CONTACT": the RLLocomotionContactController class.
        - "RL-VELOCITY": the RLLocomotionVelocityController class.

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
                },
                "LOCOMOTION": {
                    "RL-CONTACT": RLLocomotionContactController,
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
                    "RL-VELOCITY": RLLocomotionVelocityController,
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
        - "vicon_state": a ROS2StateSubscriber for the Vicon state of the robot in the real world.

        The subscribers are initialized with the corresponding handler functions and topics.

        Returns:
            Dict[str, ROS2StateSubscriber | DDSStateSubscriber]: A dictionary of state subscribers.
        """
        _subscribers = {}

        dds_low_state_sub = DDSStateSubscriber(
            topic="rt/lowstate",
            msg_type=Go2LowState_,
            handler_func=go2_low_state_handler,
            logger=self.logger,
        )
        _subscribers["low_state"] = dds_low_state_sub

        if "sim" in self.task:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
            dds_sportsmode_state_sub = DDSStateSubscriber(
                topic="rt/sportmodestate",
                msg_type=SportModeState_,
                handler_func=sport_mode_state_handler,
                logger=self.logger,
            )
            _subscribers["sports_mode_state"] = dds_sportsmode_state_sub
        else:
            from vicon_receiver.msg import Position
            ros2_vicon_sub = ROS2StateSubscriber(
                topic="/vicon/Go2/Go2",
                node_name="vicon_state",
                msg_type=Position,
                handler_func=go2_vicon_handler,
                logger=self.logger,
            )
            _subscribers["vicon_state"] = ros2_vicon_sub

        return _subscribers
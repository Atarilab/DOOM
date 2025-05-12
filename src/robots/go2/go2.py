from typing import Dict, TYPE_CHECKING

from robots.robot_base import RobotBase
from utils.mj_wrapper import MjQuadRobotWrapper
from controllers.stand_controller import (
    StayDownController,
    StandUpController,
    StandDownController,
)
from controllers.rl_contact_controller import RLLocomotionContactController
from controllers.rl_controller import RLLocomotionVelocityController

from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_

from state_manager.msg_handlers import low_state_handler, sport_mode_state_handler, vicon_handler
from state_manager.state_manager import DDSStateSubscriber, ROS2StateSubscriber

if TYPE_CHECKING:
    from controllers.controller_base import ControllerBase

class Go2(RobotBase):
    """
    This class provides robot-specific data and available controllers for the Go2 robot based on the desired task.
    """
    def __init__(self, task, logger):
        super().__init__(task=task, logger=logger)
        
        self.mj_model_wrapper = MjQuadRobotWrapper(self.xml_path)
        self.low_cmd_msg = unitree_go_msg_dds__LowCmd_
        self.low_cmd_msg_type = LowCmd_
        
        self.stand_down_joint_pos = [ 0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, ]
        self.stand_up_joint_pos = [ 0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5, ]

    @property
    def name(self):
        return "Go2"

    @property
    def joint_names(self):
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
        return len(self.joint_names)

    @property
    def default_joint_positions(self):
        return [0.0] * self.num_joints

    @property
    def effort_limit(self):
        return 23.5

    @property
    def xml_path(self):
        return "/home/atari/workspace/DOOM/src/robots/go2/go2.xml"
    
    @property
    def available_controllers(self) -> "Dict[str, Dict[str, Type[ControllerBase]]]":
        controllers = {}
        if "contact" in self.task:
            controllers = {
                "STAND": {
                    "STAY_DOWN": StayDownController,
                    "STAND_UP": StandUpController,
                    "STAND_DOWN": StandDownController,
                },
                "LOCOMOTION": {
                    "RL-CONTACT": RLLocomotionContactController,
                },
            }
        elif "velocity" in self.task:
            controllers = {
                "STAND": {
                    "STAY_DOWN": StayDownController,
                    "STAND_UP": StandUpController,
                    "STAND_DOWN": StandDownController,
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
        _subscribers = {}

        dds_low_state_sub = DDSStateSubscriber(
            topic="rt/lowstate",
            msg_type=LowState_,
            handler_func=low_state_handler,
            logger=self.logger,
        )
        _subscribers["low_state"] = dds_low_state_sub
        
        # ros2_low_state_sub = ROS2StateSubscriber(
        #     topic="/lowstate",
        #     node_name="low_state",
        #     msg_type=LowState,
        #     handler_func=low_state_handler
        # )
        # state_manager.add_subscriber("low_state", ros2_low_state_sub)

        if "sim" in self.task:
            #     ros2_sportsmode_state_sub = ROS2StateSubscriber(
            #         topic="/sportmodestate",
            #         node_name="sportmodestate",
            #         msg_type=SportModeState,
            #         handler_func=sport_mode_state_handler
            #     )
            #     state_manager.add_subscriber("sports_mode_state", ros2_sportsmode_state_sub)
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
                handler_func=vicon_handler,
                logger=self.logger,
            )
            _subscribers["vicon_state"] = ros2_vicon_sub

        return _subscribers
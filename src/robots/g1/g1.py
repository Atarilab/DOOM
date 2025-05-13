from typing import Dict, TYPE_CHECKING

from robots.robot_base import RobotBase
from utils.mj_wrapper import MjQuadRobotWrapper
from controllers.stand_controller import (
    G1StayUpController,
    G1StandUpController,
)

from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as G1LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as G1LowCmd_

from state_manager.msg_handlers import g1_low_state_handler
from state_manager.state_manager import DDSStateSubscriber, ROS2StateSubscriber

if TYPE_CHECKING:
    from controllers.controller_base import ControllerBase

class G1(RobotBase):
    """
    This class provides robot-specific data and available controllers for the G1 robot based on the desired task.
    """
    def __init__(self, task, logger):
        super().__init__(task=task, logger=logger)

        self.low_cmd_msg = unitree_hg_msg_dds__LowCmd_
        self.low_cmd_msg_type = G1LowCmd_
        
        # self.stand_down_joint_pos = [ 0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, ]
        # self.stand_up_joint_pos = [ 0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5, ]

    @property
    def name(self):
        return "G1"

    # TODO: Fetch from mj_model
    @property
    def joint_names(self):
        return [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
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
        return "/home/atari/workspace/DOOM/src/robots/g1/g1.xml"
    
    @property
    def available_controllers(self) -> "Dict[str, Dict[str, Type[ControllerBase]]]":
        controllers = {}
        if "contact" in self.task:
            controllers = {
                "STAND": {
                    "STAND_UP": G1StandUpController,
                    "STAY_UP": G1StayUpController,
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
            msg_type=G1LowState_,
            handler_func=g1_low_state_handler,
            logger=self.logger,
        )
        _subscribers["low_state"] = dds_low_state_sub

        # if "sim" in self.task:
        #     from unitree_sdk2py.idl.unitree_hg.msg.dds_ import SportModeState_
        #     dds_sportsmode_state_sub = DDSStateSubscriber(
        #         topic="rt/sportmodestate",
        #         msg_type=SportModeState_,
        #         handler_func=g1_sport_mode_state_handler,
        #         logger=self.logger, 
        #     )
        #     _subscribers["sports_mode_state"] = dds_sportsmode_state_sub
        # else:
        #     from vicon_receiver.msg import Position
        #     ros2_vicon_sub = ROS2StateSubscriber(
        #         topic="/vicon/G1/G1",
        #         node_name="vicon_state",
        #         msg_type=Position,
        #         handler_func=g1_vicon_handler,
        #         logger=self.logger,
        #     )
        #     _subscribers["vicon_state"] = ros2_vicon_sub

        return _subscribers
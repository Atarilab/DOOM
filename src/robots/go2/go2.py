from robots.robot_base import RobotBase
from utils.mj_wrapper import MjQuadRobotWrapper

class Go2(RobotBase):
    def __init__(self):
        super().__init__()
        self.mj_model_wrapper = MjQuadRobotWrapper(self.xml_path)
        
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
        
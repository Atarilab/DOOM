import time
from typing import TYPE_CHECKING, Dict, List, Type

from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as G1LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as G1LowState_

from controllers.rl_contact_bimanual_controller import RLHumanoidBimanualContactController
from controllers.rl_velocity_locomotion_controller import (
    RLHumanoidLocomotionVelocityController,
    RLHumanoidUnitreeLocomotionVelocityController,
)
from controllers.stand_controller import G1DefaultHandsController, G1LateralHandsController, G1LocoStandUpController, G1ManiStandUpController, G1ManipulationInitHandsController, G1StayUpController, G1UpperDefaultPosController, G1UpperExtendLateralController, G1UpperHomePosController, G1LowerStandUpController
from robots.robot_base import RobotBase
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from state_manager.msg_handlers import g1_low_state_handler, g1_upper_low_state_handler, g1_lower_low_state_handler, vicon_handler
from state_manager.state_manager import DDSStateSubscriber, ROS2StateSubscriber
from utils.helpers import create_joint_mapping

if TYPE_CHECKING:
    from controllers.controller_base import ControllerBase


class G1(RobotBase):
    """
    This class provides robot-specific data and available controllers for the G1 robot based on the desired task.
    """

    def __init__(self, task, logger):
        """
        Initialize the G1 robot.

        Args:
            task (str): The task to be performed by the robot.
            logger (logging.Logger): The logger to be used for logging.
        """
        super().__init__(task=task, logger=logger)

        # self.joints_unitree2isaac = create_joint_mapping(self.isaaclab_joint_names(), self.joint_names)
        # self.joints_isaac2unitree = create_joint_mapping(self.joint_names, self.isaaclab_joint_names())
        
        
        
        # Log available controllers
        self.logger.info(f"Available controllers for {self.name}: {self.available_controllers}")
        self.logger.info(f"Available subscribers for {self.name}: {self.subscribers}")

    @property
    def name(self):
        """
        Returns the name of the robot.

        Returns:
            str: The name of the robot.
        """
        return "UnitreeG1"
    

    @property
    def ee_names(self):
        """
        Returns the names of the feet of the robot.

        Returns:
            List[str]: The names of the feet of the robot.
        """
        return ["left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"]
    
    

    # TODO: Fetch from mj_model
    @property
    def actuated_joint_names(self):
        """
        Returns the names of the joints of the robot.

        Returns:
            List[str]: The names of the joints of the robot.
        """
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
    def base_link(self):
        """Return the name of the base link."""
        return "pelvis"
    
    @property
    def floating_base(self):
        """Return if the robot has a floating base."""
        return True
    
    @property
    def damping_gain(self):
        """Returns the gain used for the damping mode

        Returns:
            float: Kd value used for damping gain
        """
        
        return 8.0

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
        raise NotImplementedError

    @property
    def xml_path(self):
        """
        Returns the path to the XML file of the robot.

        Returns:
            str: The path to the XML file of the robot.
        """
        return "/home/atari/workspace/DOOM/src/robots/g1/g1_29dof.xml"
    
    @property
    def low_cmd_msg(self):
        """Return the low command message class."""
        return unitree_hg_msg_dds__LowCmd_

    @property
    def low_cmd_msg_type(self):
        """Return the low command message type."""
        return G1LowCmd_

    @property
    def available_controllers(self) -> "Dict[str, Dict[str, Type[ControllerBase]]]":
        """
        Returns a dictionary of available controllers for the G1 robot based on the desired task.

        Returns:
            Dict[str, Dict[str, Type[ControllerBase]]]: A dictionary of available controllers for the G1 robot based on the desired task.
        """
        self.logger.info(f"Available task for {self.name}: {self.task}")
        controllers = {
                "STAND": {
                    "STAND_UP": G1LocoStandUpController,
                    "DEFAULT_HANDS": G1DefaultHandsController,
                    "LATERAL_HANDS": G1LateralHandsController,
                    "MANIPULATION_INIT_HANDS": G1ManipulationInitHandsController,
                    # "UPPER_EXTEND_LATERAL": G1UpperExtendLateralController,
                    # "UPPER_HOME_POS": G1UpperHomePosController,
                }
            }
        
        if "manicont" in self.task:
            controllers["STAND"]["STAND_UP"] = G1ManiStandUpController
            controllers["BIMANUAL"] = {
                    # "RL-VELOCITY": RLHumanoidLocomotionVelocityController,
                    "RL-CONTACT": RLHumanoidBimanualContactController,
                }
        elif "velocity" in self.task:
            controllers["LOCOMOTION"] = {
                    "RL-VELOCITY": RLHumanoidLocomotionVelocityController,
                }
        elif "unitree" in self.task:
            controllers["LOCOMOTION"] = {
                    "RL-UNITREE": RLHumanoidUnitreeLocomotionVelocityController,
                }
        else:
            raise ValueError(f"Invalid task: {self.task}")

        self.logger.info(f"Available controllers for {self.name}: {controllers}")
        return controllers

    @property
    def subscribers(self) -> Dict[str, ROS2StateSubscriber | DDSStateSubscriber]:
        """
        Returns a dictionary of subscribers for the G1 robot.

        The dictionary contains the following keys:

        - "low_state": a DDSStateSubscriber for the low-level state of the robot.
        - "sports_mode_state": a DDSStateSubscriber for the sports mode state of the robot in simulation.
        - "vicon_state": a ROS2StateSubscriber for the Vicon state of the robot in the real world.

        The subscribers are initialized with the corresponding handler functions and topics.

        Returns:
            Dict[str, ROS2StateSubscriber | DDSStateSubscriber]: A dictionary of subscribers for the G1 robot.
        """
        _subscribers = {}

        dds_low_state_sub = DDSStateSubscriber(
            topic="rt/lowstate",
            msg_type=G1LowState_,
            handler_func=g1_low_state_handler,
            logger=self.logger,
        )
        _subscribers["low_state"] = dds_low_state_sub

        if "sim" in self.task:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

            from state_manager.msg_handlers import object_state_handler, sport_mode_state_handler

            dds_sportsmode_state_sub = DDSStateSubscriber(
                topic="rt/sportmodestate",
                msg_type=SportModeState_,
                handler_func=sport_mode_state_handler,
                logger=self.logger,
            )
            _subscribers["sports_mode_state"] = dds_sportsmode_state_sub

            if "manicont" in self.task:
                dds_object_state_sub = DDSStateSubscriber(
                    topic="rt/objectstate",
                    msg_type=SportModeState_,
                    handler_func=object_state_handler,
                    logger=self.logger,
                )
                _subscribers["object_state"] = dds_object_state_sub

        else:
            from vicon_receiver.msg import Position

            ros2_vicon_sub = ROS2StateSubscriber(
                topic="/vicon/G1/G1",
                node_name="vicon_state",
                msg_type=Position,
                handler_func=vicon_handler,
                logger=self.logger,
            )
            _subscribers["vicon_state"] = ros2_vicon_sub

        return _subscribers
    

    def init_low_cmd(self, cmd_msg):
        """Initialize low-level command message with G1-specific defaults."""
        self.mode_machine_ = 0
        self.update_mode_machine_ = False
        self.msc = None
        
        if "real" in self.task:
            self._init_motion_switcher()
        
        # Set mode machine and mode_pr
        cmd_msg.mode_machine = self.mode_machine_
        cmd_msg.mode_pr = self.MotorMode.PR

        # Initialize all motor commands
        for i in range(len(cmd_msg.motor_cmd)):
            cmd_msg.motor_cmd[i].mode = 1  # Enable motor
            cmd_msg.motor_cmd[i].q = 0.0
            cmd_msg.motor_cmd[i].kp = 0.0
            cmd_msg.motor_cmd[i].dq = 0.0
            cmd_msg.motor_cmd[i].kd = 0.0
            cmd_msg.motor_cmd[i].tau = 0.0
                    

    def motor_command_attributes(self) -> List[str]:
        """Get the list of motor command attributes supported by G1."""
        return ["q", "kp", "dq", "kd", "tau", "mode"]

    def update_motor_command(self, cmd_msg, motor_idx: int, motor_data: Dict):
        """Update a specific motor command with G1-specific logic."""
        # Update standard attributes
        for attr in ["q", "kp", "dq", "kd", "tau"]:
            if attr in motor_data:
                setattr(cmd_msg.motor_cmd[motor_idx], attr, motor_data[attr])
        
        cmd_msg.motor_cmd[motor_idx].mode = 1

    def update_command_modes(self, cmd_msg, motor_commands: Dict):
        """Update G1-specific command mode settings."""
        if "mode_pr" in motor_commands:
            cmd_msg.mode_pr = motor_commands["mode_pr"]
        else:
            cmd_msg.mode_pr = self.MotorMode.PR
        if "mode_machine" in motor_commands:
            cmd_msg.mode_machine = motor_commands["mode_machine"]
        else:
            cmd_msg.mode_machine = self.mode_machine_
            
            
        return cmd_msg

    def get_mode_initialization_state(self, combined_state: Dict) -> bool:
        """Check if G1 mode initialization is complete."""
        if combined_state.get("mode_machine") is not None:
            self.mode_machine_ = combined_state["mode_machine"]
            self.update_mode_machine_ = True
        return self.update_mode_machine_

    def _init_motion_switcher(self):
        """Initialize motion switcher client for G1 robot."""
        try:
            self.msc = MotionSwitcherClient()
            self.msc.SetTimeout(5.0)
            self.msc.Init()

            # Check and release any existing mode
            status, result = self.msc.CheckMode()
            while result['name']:
                self.logger.info(f"Releasing existing mode: {result['name']}")
                self.msc.ReleaseMode()
                status, result = self.msc.CheckMode()
                time.sleep(1)
                
            self.logger.info("G1 motion switcher initialized successfully")
        except Exception as e:
            raise RuntimeError(
                "Unable to read from robot. Please ensure the robot is powered on, "
                "or restart it to initialize correctly."
            )

    class MotorMode:
        PR = 0  # Series Control for Pitch/Roll Joints
        AB = 1  # Parallel Control for A/B Joints
        
class G1Upper(G1):
    """G1 robot with upper body only."""
    
    def __init__(self, task, logger):
        super().__init__(task, logger)
        self.actuated_joint_indices = [self.mj_model.joint_names[joint_name] - 1 for joint_name in self.actuated_joint_names]
        self.non_actuated_joint_indices = [self.mj_model.joint_names[joint_name] - 1 for joint_name in self.non_actuated_joint_names]
        
    @property
    def floating_base(self):
        """Return if the robot has a floating base."""
        return False

    # TODO: Fetch from mj_model
    @property
    def actuated_joint_names(self):
        """
        Returns the names of the joints of the robot.

        Returns:
            List[str]: The names of the joints of the robot.
        """
        return [

            # "waist_yaw_joint",
            # "waist_roll_joint",
            # "waist_pitch_joint",
            
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
    def non_actuated_joint_names(self):
        """
        Returns the names of the non-actuated joints of the robot.
        """
        return [
            # "waist_yaw_joint",
            # "left_hip_pitch_joint",
            # "left_hip_roll_joint",
            # "left_hip_yaw_joint",
            # "left_knee_joint",
            # "left_ankle_pitch_joint",
            # "left_ankle_roll_joint",
            
            # "right_hip_pitch_joint",
            # "right_hip_roll_joint",
            # "right_hip_yaw_joint",
            # "right_knee_joint",
            # "right_ankle_pitch_joint",
            # "right_ankle_roll_joint",
            
            # "waist_yaw_joint",
            # "waist_roll_joint",
            # "waist_pitch_joint",
        ]
    
    @property
    def ee_names(self):
        """
        Returns the names of the end effectors of the robot.
        """
        return ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    
        
    # @property
    # def num_joints(self):
    #     """
    #     Returns the number of joints of the robot.
    #     """
    #     return len(self.joint_names)

    @property
    def isaaclab_joint_names(self):
        """
        Returns the names of the joints as in IsaacLab order.
        """
        return [

            # "waist_yaw_joint",
            # "waist_roll_joint",
            # "waist_pitch_joint",
            
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
    def xml_path(self) -> str:
        """
        Returns the path to the XML file of the robot.

        Returns:
            str: The path to the XML file of the robot.
        """
        return "/home/atari/workspace/DOOM/src/robots/g1/g1_29dof_lock_waist_with_hand_rev_1_0.xml"
    
    @property
    def available_controllers(self) -> "Dict[str, Dict[str, Type[ControllerBase]]]":
        """
        Returns a dictionary of available controllers for the G1 Upper robot based on the desired task.

        Returns:
            Dict[str, Dict[str, Type[ControllerBase]]]: A dictionary of available controllers for the G1 Upper robot based on the desired task.
        """
        self.logger.info(f"Available task for {self.name}: {self.task}")
        controllers = {
                "STAND": {
                    "UPPER_DEFAULT_POS": G1UpperDefaultPosController,
                    "UPPER_EXTEND_LATERAL": G1UpperExtendLateralController,
                    "UPPER_HOME_POS": G1UpperHomePosController,
                }
            }
        
        if "manicont" in self.task:
            controllers["BIMANUAL"] = {
                    # "RL-VELOCITY": RLHumanoidLocomotionVelocityController,
                    "RL-CONTACT": RLHumanoidBimanualContactController,
                }


        self.logger.info(f"Available controllers for {self.name}: {controllers}")
        return controllers
    
    @property
    def subscribers(self) -> Dict[str, ROS2StateSubscriber | DDSStateSubscriber]:
        """
        Returns a dictionary of subscribers for the G1 Upper robot.

        The dictionary contains the following keys:

        - "low_state": a DDSStateSubscriber for the low-level state of the robot.
        - "sports_mode_state": a DDSStateSubscriber for the sports mode state of the robot in simulation.
        - "vicon_state": a ROS2StateSubscriber for the Vicon state of the robot in the real world.

        The subscribers are initialized with the corresponding handler functions and topics.

        Returns:
            Dict[str, ROS2StateSubscriber | DDSStateSubscriber]: A dictionary of subscribers for the G1 Upper robot.
        """
        _subscribers = {}

        dds_low_state_sub = DDSStateSubscriber(
            topic="rt/lowstate",
            msg_type=G1LowState_,
            handler_func=g1_upper_low_state_handler,
            logger=self.logger,
        )
        _subscribers["low_state"] = dds_low_state_sub

        if "sim" in self.task:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

            from state_manager.msg_handlers import object_state_handler, sport_mode_state_handler

            dds_sportsmode_state_sub = DDSStateSubscriber(
                topic="rt/sportmodestate",
                msg_type=SportModeState_,
                handler_func=sport_mode_state_handler,
                logger=self.logger,
            )
            _subscribers["sports_mode_state"] = dds_sportsmode_state_sub

            if "manicont" in self.task:
                dds_object_state_sub = DDSStateSubscriber(
                    topic="rt/objectstate",
                    msg_type=SportModeState_,
                    handler_func=object_state_handler,
                    logger=self.logger,
                )
                _subscribers["object_state"] = dds_object_state_sub

        else:
            from vicon_receiver.msg import Position

            ros2_vicon_sub = ROS2StateSubscriber(
                topic="/vicon/G1/G1",
                node_name="vicon_state",
                msg_type=Position,
                handler_func=vicon_handler,
                logger=self.logger,
            )
            _subscribers["vicon_state"] = ros2_vicon_sub

        return _subscribers
    

class G1Lower(G1):
    """G1 robot with upper body only."""
    
    def __init__(self, task, logger):
        super().__init__(task, logger)
        
    
    # TODO: Fetch from mj_model
    @property
    def actuated_joint_names(self):
        """
        Returns the names of the joints of the robot.

        Returns:
            List[str]: The names of the joints of the robot.
        """
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
        ]
        
    @property
    def non_actuated_joint_names(self):
        """
        Returns the names of the non-actuated joints of the robot.
        """
        return [
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
    def ee_names(self):
        """
        Returns the names of the end effectors of the robot.
        """
        return ["left_ankle_roll_link", "right_ankle_roll_link"]
    
        
    # @property
    # def num_joints(self):
    #     """
    #     Returns the number of joints of the robot.
    #     """
    #     return len(self.joint_names)

    @property
    def isaaclab_joint_names(self):
        """
        Returns the names of the joints as in IsaacLab order.
        """
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
         
        ]
    
    @property
    def xml_path(self) -> str:
        """
        Returns the path to the XML file of the robot.

        Returns:
            str: The path to the XML file of the robot.
        """
        return "/home/atari/workspace/DOOM/src/robots/g1/g1_29dof_lock_waist_rev_1_0.xml"
    
    @property
    def available_controllers(self) -> "Dict[str, Dict[str, Type[ControllerBase]]]":
        """
        Returns a dictionary of available controllers for the G1 Upper robot based on the desired task.

        Returns:
            Dict[str, Dict[str, Type[ControllerBase]]]: A dictionary of available controllers for the G1 Upper robot based on the desired task.
        """
        self.logger.info(f"Available task for {self.name}: {self.task}")
        controllers = {
                "STAND": {
                    "LOWER_DEFAULT_POS": G1LowerStandUpController,
                }
            }
        
        if "velocity" in self.task:
            controllers["LOCOMOTION"] = {
                    # "RL-VELOCITY": RLHumanoidLocomotionVelocityController,
                    "RL-VELOCITY": RLHumanoidLocomotionVelocityController,
                }


        self.logger.info(f"Available controllers for {self.name}: {controllers}")
        return controllers
    
    @property
    def subscribers(self) -> Dict[str, ROS2StateSubscriber | DDSStateSubscriber]:
        """
        Returns a dictionary of subscribers for the G1 Upper robot.

        The dictionary contains the following keys:

        - "low_state": a DDSStateSubscriber for the low-level state of the robot.
        - "sports_mode_state": a DDSStateSubscriber for the sports mode state of the robot in simulation.
        - "vicon_state": a ROS2StateSubscriber for the Vicon state of the robot in the real world.

        The subscribers are initialized with the corresponding handler functions and topics.

        Returns:
            Dict[str, ROS2StateSubscriber | DDSStateSubscriber]: A dictionary of subscribers for the G1 Upper robot.
        """
        _subscribers = {}

        dds_low_state_sub = DDSStateSubscriber(
            topic="rt/lowstate",
            msg_type=G1LowState_,
            handler_func=g1_lower_low_state_handler,
            logger=self.logger,
        )
        _subscribers["low_state"] = dds_low_state_sub

        if "sim" in self.task:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

            from state_manager.msg_handlers import sport_mode_state_handler

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
                topic="/vicon/G1/G1",
                node_name="vicon_state",
                msg_type=Position,
                handler_func=vicon_handler,
                logger=self.logger,
            )
            _subscribers["vicon_state"] = ros2_vicon_sub

        return _subscribers
# robot_interfaces/sim_robot_interface.py
import os 
import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from utils.unitree_sdk2py_bridge import UnitreeSdk2Bridge
from robot_interfaces.robot_interface_base import RobotInterfaceBase

class SimRobotInterface(RobotInterfaceBase):
    def __init__(self, config):

        self.robot_name = config["ROBOT"]
        self.robot_scene = os.getcwd() + "/robots/" + self.robot_name + "/scene.xml"

        # Initialize Mujoco model and data
        self.mj_model = mujoco.MjModel.from_xml_path(self.robot_scene)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # Set the timestep for the simulation
        self.mj_model.opt.timestep = config['SIMULATION_DT']
        
        # Initialize the viewer based on configuration
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False)
        
        # Running flag for thread control
        self.running = True
        
        self.dim_motor_sensor_ = 3 * self.mj_model.nu
        self.locker = threading.Lock()

        # Other configuration variables
        self.simulate_dt = config['SIMULATION_DT']
        self.viewer_dt = config['VIEWER_DT']
        self.domain_id = config['DOMAIN_ID']
        self.interface = config['INTERFACE']
        self.joystick_type = config['JOYSTICK_TYPE']
        self.use_joystick = config.get('USE_JOYSTICK', False)
        self.print_scene_info = config.get('PRINT_SCENE_INFORMATION', False)

        # Start threads
        self.viewer_thread = Thread(target=self._physics_viewer_thread)
        self.sim_thread = Thread(target=self._simulation_thread)

        self.viewer_thread.start()
        self.sim_thread.start()
        

    def _simulation_thread(self):
        ChannelFactoryInitialize(self.domain_id, self.interface)
        unitree = UnitreeSdk2Bridge(self.mj_model, self.mj_data)

        if self.use_joystick:
            unitree.SetupJoystick(device_id=0, js_type=self.joystick_type)
        if self.print_scene_info:
            unitree.PrintSceneInformation()

        while self.viewer.is_running():
            step_start = time.perf_counter()

            self.locker.acquire()
            
            mujoco.mj_step(self.mj_model, self.mj_data)

            self.locker.release()

            time_until_next_step = self.mj_model.opt.timestep - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def _physics_viewer_thread(self):
        while self.viewer.is_running():
            self.locker.acquire()
            # Find the robot body (might need adjustment based on your exact model)
            robot_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
            if robot_body_id >= 0:
                robot_pos = self.mj_data.xpos[robot_body_id]
                
                # Set camera lookat point
                self.viewer.cam.lookat[:] = robot_pos
                
                # Optional: soft zoom and angle
                self.viewer.cam.distance = 3.0
                self.viewer.cam.elevation = -20

            self.viewer.sync()
            self.locker.release()
            time.sleep(self.viewer_dt)


    def send_command(self, command):
        self.sim.data.ctrl[:] = command['command']
        self.sim.step()

    def receive_state(self):
        return {"observation": self.sim.data.qpos[:]}

    
    def stop(self, logger):
        """Gracefully stop the simulation and viewer threads."""
        self.running = False
        if self.viewer.is_running():
            self.viewer.close()
            
        self.viewer_thread.join()
        self.sim_thread.join()
        logger.info("Simulation exited successfully.")
# robot_interfaces/sim_robot_interface.py
import os
import threading
from threading import Thread
import time

import mujoco
import mujoco.viewer
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from utils.unitree_sdk2py_bridge import UnitreeSdk2Bridge


class SimRobotInterface():
    def __init__(self, config):

        self.robot_name = config["ROBOT"]
        self.robot_scene = os.path.join(os.getcwd(), "robots", self.robot_name, config["SCENE"])

        # Initialize Mujoco model and data
        self.mj_model = mujoco.MjModel.from_xml_path(self.robot_scene)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Check for object body in the scene
        self.object_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "object")

        # Set the timestep for the simulation
        self.mj_model.opt.timestep = config["SIMULATION_DT"]

        # Base link
        self.base_link_name = config["BASE_LINK"]

        # Elastic band
        use_elastic_band = config.get("ELASTIC_BAND", False)
        self.elastic_band = ElasticBand() if use_elastic_band else None
        # Initialize the viewer based on configuration
        viewer_kwargs = {
            "show_left_ui": False,
            "show_right_ui": False,
        }
        if use_elastic_band:
            self.band_link = self.mj_model.body(self.base_link_name)
            viewer_kwargs["key_callback"] = self.elastic_band.MujuocoKeyCallback

        # Set up viewer
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data, **viewer_kwargs)

        self.dim_motor_sensor_ = 3 * self.mj_model.nu
        self.locker = threading.Lock()

        # Other configuration variables
        self.simulate_dt = config["SIMULATION_DT"]
        self.viewer_dt = config["VIEWER_DT"]
        self.domain_id = config["DOMAIN_ID"]
        self.interface = config["INTERFACE"]

        self.print_scene_info = config.get("PRINT_SCENE_INFORMATION", False)

        # Start threads
        self.viewer_thread = Thread(target=self._physics_viewer_thread)
        self.sim_thread = Thread(target=self._simulation_thread)

        self.viewer_thread.start()
        self.sim_thread.start()

    def _simulation_thread(self):
        ChannelFactoryInitialize(self.domain_id, self.interface)
        unitree = UnitreeSdk2Bridge(self.mj_model, self.mj_data, robot=self.robot_name, object=self.object_id)

        if self.print_scene_info:
            unitree.PrintSceneInformation()

        while self.viewer.is_running():
            step_start = time.perf_counter()

            self.locker.acquire()

            if self.elastic_band:
                if self.elastic_band.enable:
                    force = self.elastic_band.Advance(self.mj_data.qpos[:3], self.mj_data.qvel[:3])
                    self.mj_data.xfrc_applied[self.band_link.id][:3] = force

            mujoco.mj_step(self.mj_model, self.mj_data)

            self.locker.release()

            time_until_next_step = self.mj_model.opt.timestep - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def _physics_viewer_thread(self):
        # Enable rendering of world frame axes
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
        while self.viewer.is_running():
            self.locker.acquire()
            # Find the robot body (might need adjustment based on your exact model)
            robot_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.base_link_name)
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

    def stop(self, logger):
        """Gracefully stop the simulation and viewer threads."""
        if self.viewer.is_running():
            self.viewer.close()

        self.viewer_thread.join()
        self.sim_thread.join()
        logger.info("Simulation exited successfully.")


class ElasticBand:

    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 2.5])
        self.length = 0
        self.enable = True

    def Advance(self, x, dx):
        """
        Args:
          δx: desired position - current position
          dx: current velocity
        """
        δx = self.point - x
        distance = np.linalg.norm(δx)
        direction = δx / distance
        v = np.dot(dx, direction)
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable

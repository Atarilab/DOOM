# tests/test_robot_interfaces.py
import unittest
from robot_interfaces.sim_robot_interface import SimRobotInterface

class TestSimRobotInterface(unittest.TestCase):
    def setUp(self):
        self.sim_interface = SimRobotInterface("path/to/mujoco/model.xml")

    def test_receive_state(self):
        state = self.sim_interface.receive_state()
        self.assertIsInstance(state, dict)
        self.assertIn("observation", state)

    def test_send_command(self):
        command = {"command": [0.0] * 6}
        self.sim_interface.send_command(command)
        # Check simulation steps correctly

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mello conrtroller for SE(3) teleoperation devices."""

import ast  # For parsing string data from Mello
import threading
import time
from collections.abc import Callable

import numpy as np
import serial  # For serial communication with Mello
from scipy.spatial.transform import Rotation  # For handling rotations

from ..device_base import DeviceBase  # Interface for device base class


class Se3Mello(DeviceBase):
    """A mello controller for SE(3) commands as delta poses.

    This class implements a mello controller to provide commands to a robotic arm with a Mello device.
    It uses the mello arm  to control the robot's end-effector in SE(3) space, allowing for
    teleoperation of the robot's pose in 3D space using motorized input devices.

    The command comprises of two parts:
    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    """

    def __init__(
        self,
        port: str = "/dev/serial/by-id/usb-M5Stack_Technology_Co.__Ltd_M5Stack_UiFlow_2.0_4827e266dd480000-if00",
        baudrate: int = 115200,
    ):
        """Initialize the Mello Gello controller.

        Args:
            pos_sensitivity: Magnitude of input position command scaling. Defaults to 0.4.
            rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 0.8.
        """

        # store inputs
        self.port = port  # serial port for Mello
        self.baudrate = baudrate  # baudrate for serial communication

        # Serial communication
        self.serial = None  # serial communication interface for Mello
        self._setup_serial()  # setup the serial communication with Mello

        # State variables
        self.prev_joints = [0] * 6
        self.prev_gripper = 0
        self.latest_values = self.prev_joints
        self._close_gripper = False
        self.running = True

        # dictionary for additional callbacks
        self._additional_callbacks = dict()

        # run a thread for listening to a devite updates
        self._start_read_thread()

    def __del__(self):
        """Dectructor for the class."""
        self.cleanup()

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Mello Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tScaled version of UR5 robot arm controller\n"
        msg += "\tJoints are controlled via motors and controllers\n"
        msg += "\tMove the arm in any direction by rotating motors (up, down, left right, diagonal)\n"
        msg += "\tUse analog stick to open/close the gripper\n"
        return msg

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("Serial port closed")

    """
    Operations
    """

    def reset(self):
        """Reset the internals."""
        self._close_gripper = False
        self.prev_joints = [0] * 6
        self.prev_gripper = 0
        self.latest_values = self.prev_joints

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard."""
        self._additional_callbacks[key] = func

    def advance(self):
        """Provides the result from mello event state.

        Returns:
            A tuple containing the latest SE(3) delta pose and gripper state.
        """
        # rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        return self.prev_joints, self.prev_gripper

    def _setup_serial(self):
        """Set up serial connection to Mello device."""
        try:
            self.serial = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=1)  # 1 second timeout
            print(f"Successfully connected to {self.port}")
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            raise

    def _start_read_thread(self):
        """Start a thread to read data from the Mello device."""
        self.read_thread = threading.Thread(target=self._read_thread)
        self.read_thread.daemon = True
        self.read_thread.start()

    def _read_thread(self):
        """Continuously read and parse joint data from Mello."""
        while self.running:
            try:
                if self.serial.in_waiting:
                    line = self.serial.readline().decode("utf-8").strip()
                    try:
                        data = ast.literal_eval(line)
                        joint_positions = data.get("joint_positions:", [0] * 7)

                        # Extract and process joint data
                        joints_deg = joint_positions[:6]
                        joints_deg[2] *= -1  # Flip direction if needed
                        joints_rad = self._degrees_to_radians(joints_deg)

                        gripper_value = joint_positions[-1] if len(joint_positions) > 6 else self.prev_gripper

                        # Compute deltas if not zero
                        if not all(j == 0 for j in joints_rad):
                            self.prev_joints = joints_rad

                        # Gripper state
                        self.prev_gripper = gripper_value
                        self.latest_values = self.prev_joints
                        self._close_gripper = False if gripper_value >= 0 else True

                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing serial data: {e}")
            except Exception as e:
                print(f"Error reading serial data: {e}")

            time.sleep(0.01)

    def _degrees_to_radians(self, degrees):
        """Convert a list of angles from degrees to radians."""
        return [np.deg2rad(d) for d in degrees]

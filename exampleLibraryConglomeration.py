

"""hal folder is below:







        content:

        """This module contains QCar specific implementations of hal features"""
from pytransform3d import rotations as pr
import numpy as np
import time
import os

from pal.utilities.math import wrap_to_pi
from hal.utilities.estimation import EKF, KalmanFilter
from hal.utilities.control import PID, StanleyController


class QCarEKF:
    """ An EKF designed to estimate the 2D position and orientation of a QCar.

    Attributes:
        kf (KalmanFilter): Kalman filter for orientation estimation.
        ekf (EKF): Extended Kalman filter for pose estimation.
        L (float): Wheelbase of the vehicle.
        x_hat (ndarray): State estimate vector [x; y; theta].
    """

    def __init__(
            self,
            x_0,
            Q_kf=np.diagflat([0.0001, 0.001]),
            R_kf=np.diagflat([.001]),
            Q_ekf=np.diagflat([0.01, 0.01, 0.01]),
            R_ekf=np.diagflat([0.01, 0.01, 0.001])
        ):
        """Initialize QCarEKF with initial state and noise covariance matrices.

        Args:
            x_0 (ndarray): Initial state vector [x, y, theta].
            Q_kf (ndarray, optional): KF process noise covariance matrix.
            R_kf (ndarray, optional): KF measurement noise covariance matrix.
            Q_ekf (ndarray, optional): EKF process noise covariance matrix.
            R_ekf (ndarray, optional): EKF measurement noise covariance matrix.
        """

        x_0 = np.squeeze(x_0)
        self.kf = KalmanFilter(
            x_0=[x_0[2], 0],
            P0=np.eye(2),
            Q=Q_kf,
            R=R_kf,
            A=np.array([[0, -1], [0, 0]]),
            B=np.array([[1], [0]]),
            C=np.array([[1, 0]])
        )

        self.ekf = EKF(
            x_0=x_0,
            P0=np.eye(3),
            Q=Q_ekf,
            R=R_ekf,
            f=self.f,
            J_f=self.J_f,
            C=np.eye(3)
        )

        self.L = 0.2
        self.x_hat = self.ekf.x_hat

    def f(self, x, u, dt):
        """Motion model for the kinematic bicycle model.

        Args:
            x (ndarray): State vector [x, y, theta].
            u (ndarray): Control input vector [v, delta].
            dt (float): Time step in seconds.

        Returns:
            ndarray: Updated state vector after applying motion model.
        """

        return x + dt * u[0] * np.array([
            [np.cos(x[2,0])],
            [np.sin(x[2,0])],
            [np.tan(u[1]) / self.L]
        ])

    def J_f(self, x, u, dt):
        """Jacobian of the motion model for the kinematic bicycle model.

        Args:
            x (ndarray): State vector [x, y, theta].
            u (ndarray): Control input vector [v, delta].
            dt (float): Time step in seconds.

        Returns:
            ndarray: Jacobian matrix of the motion model.
        """

        return np.array([
            [1, 0, -dt*u[0]*np.sin(x[2,0])],
            [0, 1, dt*u[0]*np.cos(x[2,0])],
            [0, 0, 1]
        ])

    def update(self, u=None, dt=None, y_gps=None, y_imu=None):
        """Update the EKF state estimate using GPS and IMU measurements.

        Args:
            u (ndarray, optional): Control input vector [v, delta].
            dt (float, optional): Time step in seconds.
            y_gps (ndarray, optional): GPS measurement vector [x, y, th].
            y_imu (float, optional): IMU measurement of orientation.
        """

        if dt is not None:
            if y_imu is not None:
                self.kf.predict(y_imu, dt)
                self.kf.x_hat[0,0] = wrap_to_pi(self.kf.x_hat[0,0])
            if u is not None:
                self.ekf.predict(u, dt)
                self.ekf.x_hat[2,0] = wrap_to_pi(self.ekf.x_hat[2,0])

        if y_gps is not None:
            y_gps = np.squeeze(y_gps)

            y_kf = (
                wrap_to_pi(y_gps[2] - self.kf.x_hat[0,0])
                + self.kf.x_hat[0,0]
            )
            self.kf.correct(y_kf, dt)
            self.kf.x_hat[0,0] = wrap_to_pi(self.kf.x_hat[0,0])

            y_ekf = np.array([
                [y_gps[0]],
                [y_gps[1]],
                [self.kf.x_hat[0,0]]
            ])
            z_ekf = y_ekf - self.ekf.C @ self.ekf.x_hat
            z_ekf[2] = wrap_to_pi(z_ekf[2])
            y_ekf = z_ekf + self.ekf.C @ self.ekf.x_hat
            self.ekf.correct(y_ekf, dt)
            self.ekf.x_hat[2,0] = wrap_to_pi(self.ekf.x_hat[2,0])

        else:
            y_ekf = (
                wrap_to_pi(self.kf.x_hat[0,0] - self.ekf.x_hat[2,0])
                + self.ekf.x_hat[2,0]
            )
            self.ekf.correct([None, None, y_ekf], dt)
            self.ekf.x_hat[2,0] = wrap_to_pi(self.ekf.x_hat[2,0])

        self.x_hat = self.ekf.x_hat

class QCarDriveController:
    """Implements a drive controller for a QCar that handles speed and steering

    Attributes:
        speedController (PID): PI controller for speed control.
        steeringController (StanleyController): Nonlinear Stanley controller
            for steering control.
    """

    def __init__(self, waypoints, cyclic):
        """Initialize QCarDriveController

        Args:
            waypoints (list): List of waypoints for the controller to follow.
            cyclic (bool): Indicates if the waypoint path is cyclic or not.
        """

        self.speedController = PID(
            Kp=0.1,
            Ki=1,
            Kd=0,
            uLimits=(-0.3, 0.3)
        )

        self.steeringController = StanleyController(
            waypoints=waypoints,
            k=1,
            cyclic=cyclic
        )
        self.steeringController.maxSteeringAngle = np.pi/6

    def reset(self):
        """Resets the internal state of the speed and steering controllers."""

        self.speedController.reset()
        self.steeringController.wpi = 0
        self.steeringController.pathComplete = False

    def updatePath(self, waypoints, cyclic):
        """Updates the waypoint path for the steering controller.

        Args:
            waypoints (list): List of new waypoints to follow
            cyclic (bool): Indicates if the updated path is cyclic or not.
        """

        self.steeringController.updatePath(waypoints, cyclic)

    def update(self, p, th, v, v_ref, dt):
        """Updates the drive controller with the current state of the QCar.

        Args:
            p (ndarray): Position vector [x, y].
            th (float): Orientation angle in radians.
            v (float): Current speed of the QCar.
            v_ref (float): Reference speed for the QCar.
            dt (float): Time step in seconds.

        Returns:
            float: Speed control input.
            float: Steering control input.
        """

        if not self.steeringController.pathComplete:
            delta = self.steeringController.update(p, th, v)
        else:
            delta = 0
            v_ref = 0

        u = self.speedController.update(v_ref, v, dt)

        return u, delta


        

        



        products:


            import numpy as np
from hal.utilities.path_planning import RoadMap

class SDCSRoadMap(RoadMap):
    """A RoadMap implementation for Quanser's Self-Driving Car studio (SDCS)"""

    def __init__(self, leftHandTraffic=False, useSmallMap=False):
        """Initialize a new SDCSRoadMap instance.

        Args:
            leftHandTraffic (bool): If true, assumes cars drive on the left.
                Defaults to False.
            useSmallMap (bool): If true, will use the smaller map variant.
                Defaults to False.
        """
        super().__init__()

        # useful constants
        scale = 0.002035
        xOffset = 1134
        yOffset = 2363

        innerLaneRadius = 305.5 * scale
        outerLaneRadius = 438 * scale
        trafficCircleRadius = 333 * scale
        oneWayStreetRadius = 350 * scale
        kinkStreetRadius = 375 * scale

        pi = np.pi
        halfPi = pi/2

        def scale_then_add_nodes(nodePoses):
            for pose in nodePoses:
                pose[0] = scale * (pose[0] - xOffset)
                pose[1] = scale * (yOffset - pose[1])
                self.add_node(pose)

        if leftHandTraffic:
            nodePoses = [
                [1134, 2427, halfPi],
                [1134, 2323, halfPi],
                [1266, 2323, -halfPi],
                [1688, 2896, pi],
                [1688, 2763, 0],
                [2242, 2323, -halfPi],
                [2109, 2323, halfPi],
                [1741, 1822, 0],
                [1634, 1955, pi],
                [766, 1822, 0],
                [766, 1955, pi],
                [504, 2589, 138*pi/180],
            ]
            if not useSmallMap:
                nodePoses += [
                    [1134, 1428, halfPi],
                    [1266, 1454, -halfPi],
                    [2242, 1454, -halfPi],
                    [2109, 1200, halfPi],
                    [1854.5, 814.5, 170.6*pi/180],
                    [1580, 540, 99.4*pi/180],
                    [1440, 856, 42*pi/180],
                    [1523, 958, -138*pi/180],
                    [1400, 153, 0],
                    [1134, 286, pi],
                    [159, 905, halfPi],
                    [291, 905, -halfPi],
                ]

            edgeConfigs = [
                [0, 1, 0],
                [1, 7, outerLaneRadius],
                [1, 10, innerLaneRadius],
                [2, 4, innerLaneRadius],
                [2, 11, innerLaneRadius],
                [3, 0, outerLaneRadius],
                [3, 11, outerLaneRadius],
                [4, 6, innerLaneRadius],
                [5, 3, outerLaneRadius],
                [6, 8, innerLaneRadius],
                [7, 5, outerLaneRadius],
                [8, 2, innerLaneRadius],
                [8, 10, 0],
                [9, 2, outerLaneRadius],
                [9, 7, 0],
                [10, 1, innerLaneRadius],
                [11, 9, oneWayStreetRadius],
            ]
            if not useSmallMap:
                edgeConfigs += [
                    [1, 12, 0],
                    [6, 15, 0],
                    [7, 15, innerLaneRadius],
                    [8, 12, outerLaneRadius],
                    [9, 12, innerLaneRadius],
                    [10, 22, outerLaneRadius],
                    [11, 22, outerLaneRadius],
                    [12, 18, outerLaneRadius],
                    [13, 2, 0],
                    [13, 7, innerLaneRadius],
                    [13, 10, outerLaneRadius],
                    [14, 5, 0],
                    [14, 8, outerLaneRadius],
                    [15, 16, innerLaneRadius],
                    [16, 17, trafficCircleRadius],
                    [16, 19, innerLaneRadius],
                    [17, 14, trafficCircleRadius],
                    [17, 16, trafficCircleRadius],
                    [17, 21, innerLaneRadius],
                    [18, 17, innerLaneRadius],
                    [19, 13, innerLaneRadius],
                    [20, 14, trafficCircleRadius],
                    [20, 16, trafficCircleRadius],
                    [21, 23, innerLaneRadius],
                    [22, 20, outerLaneRadius],
                    [23, 9, innerLaneRadius],
                ]
        else: # Right-side Traffic
            nodePoses = [
                [1134, 2299, -halfPi],
                [1266, 2323, halfPi],
                [1688, 2896, 0],
                [1688, 2763, pi],
                [2242, 2323, halfPi],
                [2109, 2323, -halfPi],
                [1632, 1822, pi],
                [1741, 1955, 0],
                [766, 1822, pi],
                [766, 1955, 0],
                [504, 2589, -42*pi/180],
            ]
            if not useSmallMap:
                nodePoses += [
                    [1134, 1300, -halfPi],
                    [1134, 1454, -halfPi],
                    [1266, 1454, halfPi],
                    [2242, 905, halfPi],
                    [2109, 1454,-halfPi],
                    [1580, 540, -80.6*pi/180],
                    [1854.4, 814.5, -9.4*pi/180],
                    [1440, 856, -138*pi/180],
                    [1523, 958, 42*pi/180],
                    [1134, 153, pi],
                    [1134, 286, 0],
                    [159, 905, -halfPi],
                    [291, 905, halfPi],
                ]

            edgeConfigs = [
                [0, 2, outerLaneRadius],
                [1, 7, innerLaneRadius],
                [1, 8, outerLaneRadius],
                [2, 4, outerLaneRadius],
                [3, 1, innerLaneRadius],
                [4, 6, outerLaneRadius],
                [5, 3, innerLaneRadius],
                [6, 0, outerLaneRadius],
                [6, 8, 0],
                [7, 5, innerLaneRadius],
                [8, 10, oneWayStreetRadius],
                [9, 0, innerLaneRadius],
                [9, 7, 0],
                [10, 1, innerLaneRadius],
                [10, 2, innerLaneRadius],
            ]
            if not useSmallMap:
                edgeConfigs += [
                    [1, 13, 0],
                    [4, 14, 0],
                    [6, 13, innerLaneRadius],
                    [7, 14, outerLaneRadius],
                    [8, 23, innerLaneRadius],
                    [9, 13, outerLaneRadius],
                    [11, 12, 0],
                    [12, 0, 0],
                    [12, 7, outerLaneRadius],
                    [12, 8, innerLaneRadius],
                    [13, 19, innerLaneRadius],
                    [14, 16, trafficCircleRadius],
                    [14, 20, trafficCircleRadius],
                    [15, 5, outerLaneRadius],
                    [15, 6, innerLaneRadius],
                    [16, 17, trafficCircleRadius],
                    [16, 18, innerLaneRadius],
                    [17, 15, innerLaneRadius],
                    [17, 16, trafficCircleRadius],
                    [17, 20, trafficCircleRadius],
                    [18, 11, kinkStreetRadius],
                    [19, 17, innerLaneRadius],
                    [20, 22, outerLaneRadius],
                    [21, 16, innerLaneRadius],
                    [22, 9, outerLaneRadius],
                    [22, 10, outerLaneRadius],
                    [23, 21, innerLaneRadius],
                ]

        scale_then_add_nodes(nodePoses)
        for edgeConfig in edgeConfigs:
            self.add_edge(*edgeConfig)




        










        """This module contains QCar specific implementations of the hal.utilities.geometry features"""

from pytransform3d import rotations as pr   

from hal.utilities.geometry import MobileRobotGeometry
import numpy as np

class QCarGeometry(MobileRobotGeometry):
    """QCarGeometry class for defining QCar-specific frames of reference.

    This class inherits from the MobileRobotGeometry class and adds frames
    specific to the QCar, such as the front and rear axles, CSI sensors,
    IMU, Realsense, and RPLidar.
    """

    def __init__(self):
        """Initialize the QCarGeometry class with QCar-specific frames."""

        super().__init__()

        self.defaultFrame = 'body'

        # Body reference frame center is located at,
        # X: halfway between front and rear axles
        # Y: halway between left and right wheels
        # Z: 0.0103 m above the ground (chassis bottom)
        # the following frames are all defined w.r.t this center
        self.add_frame(
            name='CG',
            p=[0.0248, -0.0074, 0.0606],
            R=np.eye(3)
        )
        self.add_frame(
            name='front_axle',
            p=[0.1300, 0, 0.0207],
            R=np.eye(3)
        )
        self.add_frame(
            name='rear_axle',
            p=[-0.1300, 0, 0.0207],
            R=np.eye(3)
        )
        self.add_frame(
            name='csi_front',
            p=[0.1930, 0, 0.0850],
            R=pr.active_matrix_from_extrinsic_euler_xyz(
                [-np.pi/2, 0, -np.pi/2]
            )
        )
        self.add_frame(
            name='csi_left',
            p=[0.0140, 0.0438, 0.0850],
            R=pr.active_matrix_from_extrinsic_euler_xyz([-np.pi/2, 0, 0])
        )
        self.add_frame(
            name='csi_rear',
            p=[-0.1650, 0, 0.0850],
            R=pr.active_matrix_from_extrinsic_euler_xyz([-np.pi/2, 0, np.pi/2])
        )
        self.add_frame(
            name='csi_right',
            p=[0.0140, -0.0674, 0.0850],
            R=pr.active_matrix_from_extrinsic_euler_xyz([-np.pi/2, 0, np.pi])
        )
        self.add_frame(
            name='imu',
            p=[0.1278, 0.0223, 0.0792],
            R=np.eye(3)
        )
        self.add_frame(
            name='realsense',
            p=[0.0822, 0.0003, 0.1479],
            R=pr.active_matrix_from_extrinsic_euler_xyz(
                [-np.pi/2, 0, -np.pi/2]
            )
        )
        self.add_frame(
            name='rplidar',
            p=[-0.0108, 0, 0.1696],
            R=np.eye(3)
        )

        self.defaultFrame = 'world'









    the pal folder is below:






        products:
    





    
"""qcar: A module for simplifying interactions with the QCar hardware platform.

This module provides a set of API classes and tools to facilitate working with
the QCar hardware platform. It is designed to make it easy to set up and read
data from various QCar sensors and components, as well as perform basic
input/output operations.
"""
import numpy as np
import platform
import os
import json
import time
from quanser.devices.exceptions import DeviceError
from quanser.hardware import HIL, HILError, PWMMode, MAX_STRING_LENGTH, Clock
from quanser.hardware.enumerations import BufferOverflowMode
try:
    from quanser.common import Timeout
except:
    from quanser.communications import Timeout

from pal.utilities.vision import Camera2D, Camera3D
from pal.utilities.lidar import Lidar
from pal.utilities.stream import BasicStream
from pal.utilities.math import Calculus
from pal.products.qcar_config import  QCar_check
from os.path import realpath, join, exists , dirname

IS_PHYSICAL_QCAR = ('nvidia' == os.getlogin()) \
    and ('aarch64' == platform.machine())
"""A boolean constant indicating if the current device is a physical QCar.

This constant is set to True if both the following conditions are met:
1. The current user's login name is 'nvidia'.
2. The underlying system's hardware architecture is 'aarch64'.

It's intended to be used for configuring execution as needed depending on if
the executing platform is a physical and virtual QCar.
"""

QCAR_CONFIG_PATH=join(dirname(realpath(__file__)),'qcar_config.json')
"""The absolute path of QCar configuration file.
QCar config file should always be saved in the same directory as qcar.py.
"""

if not exists(QCAR_CONFIG_PATH):
    try:
        QCar_check(IS_PHYSICAL_QCAR)
        QCAR_CONFIG=json.load(open(QCAR_CONFIG_PATH,'r'))
    except HILError as e:
        print('QCar configuration file loading unsuccessful')
        print(e.get_error_message())
else:
    QCAR_CONFIG=json.load(open(QCAR_CONFIG_PATH,'r'))
"""A Dictionary containing parameters specific to differnet QCar types.
When a class from qcar module is called, the config file is  loaded.
If it does not exist and IS_PHYSICAL_QCAR is true, the config will is 
automatically created.
"""

class QCar():
    """Class for performing basic QCarIO"""
    # Car constants

    def __init__(
            self,
            readMode=0,
            frequency=500,
            pwmLimit=0.3,
            steeringBias=0
        ):
        """ Configure and initialize the QCar.

        readMode:
            0 = immediate I/O,
            1 = task based I/O

        id: board identifier id number for virtual use only
        frequency: sampling frequency (used when readMode = 1)
        pwmLimit: maximum (and absolute minimum) command for writing to motors
        steeringBias: steering bias to add to steering command internally
        """

        self.card = HIL()
        self.hardware = IS_PHYSICAL_QCAR

        if self.hardware:
            boardIdentifier = "0"
            self.carType = QCAR_CONFIG['cartype']             #read from config file
        else:
            boardIdentifierVirtualQCar = "0@tcpip://localhost:18960?nagle='off'"
            self.carType = 0
            # boardIdentifierVirtualQCar2 = "0@tcpip://localhost:18960?nagle='off'"

        if self.hardware:
            try:
                self.card.open(QCAR_CONFIG["carname"], boardIdentifier)
            except HILError as e:
                print(e.get_error_message())
        else:
            try:
                self.card.open('qcar', boardIdentifierVirtualQCar)
            except HILError as e:
                print(e.get_error_message())

        # if self.carType == 0:
        #     print('No QCar found!')
        #     return
        self.readMode = readMode
        self.io_task_running = False
        self.pwmLimit = pwmLimit
        self.steeringBias = steeringBias
        self.frequency = frequency

        self._configure_parameters()


        if self.card.is_valid():
            self._set_options()

            if self.hardware and self.readMode == 1:
                self._create_io_task()
            print('QCar configured successfully.')

        # Read buffers (external)
        self.motorCurrent = np.zeros(1, dtype=np.float64)
        self.batteryVoltage = np.zeros(1, dtype=np.float64)
        self.motorEncoder = np.zeros(1, dtype=np.int32)
        self.motorTach = np.zeros(1, dtype=np.float64)
        self.accelerometer = np.zeros(3, dtype=np.float64)
        self.gyroscope = np.zeros(3, dtype=np.float64)

        self.START_TIME = time.time()

    def _elapsed_time(self):
        return time.time() - self.START_TIME

    def _configure_parameters(self):
        # Common Constants
        self.ENCODER_COUNTS_PER_REV = 720.0 # counts per revolution
        self.WHEEL_TRACK = 0.172 # left to right wheel distance in m

        ### QCar type-specific parameters

        self.WHEEL_RADIUS = QCAR_CONFIG['WHEEL_RADIUS'] # front/rear wheel radius in m
        self.WHEEL_BASE = QCAR_CONFIG['WHEEL_BASE'] # front to rear wheel distance in m
        self.PIN_TO_SPUR_RATIO = QCAR_CONFIG['PIN_TO_SPUR_RATIO']
            # (diff_pinion*pinion) / (spur*diff_spur)
        self.CPS_TO_MPS = (1/(self.ENCODER_COUNTS_PER_REV*4) # motor-speed unit conversion
            * self.PIN_TO_SPUR_RATIO * 2*np.pi * self.WHEEL_RADIUS)

        # Write channels
        self.WRITE_PWM_CHANNELS = np.array(
            QCAR_CONFIG['WRITE_PWM_CHANNELS'], 
            dtype=np.int32)
        self.WRITE_OTHER_CHANNELS = np.array(
            QCAR_CONFIG['WRITE_OTHER_CHANNELS'],
            dtype=np.int32
            )
        self.WRITE_DIGITAL_CHANNELS = np.array(
            QCAR_CONFIG['WRITE_DIGITAL_CHANNELS'],
            dtype=np.int32
            )
        # write buffer channels:
        self.writePWMBuffer = np.zeros(
            QCAR_CONFIG['writePWMBuffer'], 
            dtype=np.float64)
        self.writeDigitalBuffer = np.zeros(
            QCAR_CONFIG['writeDigitalBuffer'], 
            dtype=np.int8)
        self.writeOtherBuffer = np.zeros(
            QCAR_CONFIG['writeOtherBuffer'], 
            dtype=np.float64)

        # Read channels
        self.READ_ANALOG_CHANNELS = np.array(
            QCAR_CONFIG['READ_ANALOG_CHANNELS'], 
            dtype=np.int32)
        self.READ_ENCODER_CHANNELS = np.array(
            QCAR_CONFIG['READ_ENCODER_CHANNELS'], 
            dtype=np.uint32)
        self.READ_OTHER_CHANNELS = np.array(
            QCAR_CONFIG['READ_OTHER_CHANNELS'],
            dtype=np.int32
            )

        # Read buffers (internal)
        self.readAnalogBuffer = np.zeros(
            QCAR_CONFIG['readAnalogBuffer'], 
            dtype=np.float64)
        self.readEncoderBuffer = np.zeros(
            QCAR_CONFIG['readEncoderBuffer'], 
            dtype=np.int32)
        self.readOtherBuffer = np.zeros(
            QCAR_CONFIG['readOtherBuffer'], 
            dtype=np.float64)



    def _set_options(self):
        if self.carType in [1,0]:
            # Set PWM mode (duty cycle)
            self.card.set_pwm_mode(
                np.array([0], dtype=np.uint32),
                1,
                np.array([PWMMode.DUTY_CYCLE], dtype=np.int32)
            )
            # Set PWM frequency
            self.card.set_pwm_frequency(
                np.array([0], dtype=np.uint32),
                1,
                np.array([60e6/4096], dtype=np.float64)
            )
            # Set Motor coast to 0
            self.card.write_digital(
                np.array([40], dtype=np.uint32),
                1,
                np.zeros(1, dtype=np.float64)
            )

        # Set board-specific options
        boardOptionsString = ("steer_bias=" + str(self.steeringBias)
            + ";motor_limit=" + str(self.pwmLimit) + ';')
        self.card.set_card_specific_options(
            boardOptionsString,
            MAX_STRING_LENGTH
        )

        # Set Encoder Properties
        self.card.set_encoder_quadrature_mode(
            self.READ_ENCODER_CHANNELS,
            len(self.READ_ENCODER_CHANNELS),
            np.array([4],
            dtype=np.uint32)
        )
        self.card.set_encoder_filter_frequency(
            self.READ_ENCODER_CHANNELS,
            len(self.READ_ENCODER_CHANNELS),
            np.array([60e6/1],
            dtype=np.uint32)
        )
        self.card.set_encoder_counts(
            self.READ_ENCODER_CHANNELS,
            len(self.READ_ENCODER_CHANNELS),
            np.zeros(1, dtype=np.int32)
        )

    def _create_io_task(self):
        # Define reading task for QCar 1 or QCar 2
        self.readTask = self.card.task_create_reader(
            int(self.frequency*2),
            self.READ_ANALOG_CHANNELS,
            len(self.READ_ANALOG_CHANNELS),
            self.READ_ENCODER_CHANNELS,
            len(self.READ_ENCODER_CHANNELS),
            None,
            0,
            self.READ_OTHER_CHANNELS,
            len(self.READ_OTHER_CHANNELS)
        )

        # Set buffer overflow mode depending on
        # whether its for hardware or virtual QCar
        if self.hardware:
            self.card.task_set_buffer_overflow_mode(
                self.readTask,
                BufferOverflowMode.OVERWRITE_ON_OVERFLOW
            )
        else:
            self.card.task_set_buffer_overflow_mode(
                self.readTask,
                BufferOverflowMode.WAIT_ON_OVERFLOW
            )

        # Start the reading task
        self.card.task_start(
            self.readTask,
            Clock.HARDWARE_CLOCK_0,
            self.frequency,
            2**32-1
        )
        self.io_task_running = True

    def terminate(self):
        # This function terminates the QCar card after setting
        # final values for throttle, steering and LEDs.
        # Also terminates the task reader.

        try:
            # write 0 PWM command, 0 steering, and turn off all LEDs
            if self.carType in [1,0]:
                self.write(0, 0, np.zeros(8, dtype=np.float64))
            elif self.carType == 2:
                self.write(0, 0, np.zeros(16, dtype=np.int8))
            # self.card.write(
            #     None,
            #     0,
            #     self.WRITE_PWM_CHANNELS,
            #     len(self.WRITE_PWM_CHANNELS),
            #     None,
            #     0,
            #     self.WRITE_OTHER_CHANNELS,
            #     len(self.WRITE_OTHER_CHANNELS),
            #     None,
            #     np.zeros(len(self.WRITE_PWM_CHANNELS), dtype=np.float64),
            #     None,
            #     np.zeros(len(self.WRITE_OTHER_CHANNELS), dtype=np.float64)
            # )

            # if using Task based I/O, stop the readTask.
            if self.readMode:
                self.card.task_stop(self.readTask)
            self.card.close()

        except HILError as h:
            print(h.get_error_message())

    def read_write_std(self, throttle, steering, LEDs=None):
        """ Read and write standard IO signals for the QCar

        Use this to write throttle, steering and LED commands, as well as
            update buffers for battery voltage, motor current,
            motor encoder counts, motor tach speed, and IMU data.

        throttle - this method will saturate based on the pwmLimit.

        steering - this method will saturate from -0.6 rad to 0.6 rad

        LEDs - a numpy string of 8 values

        Updates the following 6 buffers: motorCurrent, batteryVoltage,
            accelerometer, gyroscope, motorEncoder, motorTach

        """
        self.write(throttle, steering, LEDs)
        self.read()


    def read(self):
        if not (self.hardware or self.io_task_running) and self.readMode:
            self._create_io_task()

        try:
            # if using task based I/O, use the read task
            self.currentTimeStamp = self._elapsed_time()
            if self.readMode == 1:
                self.card.task_read(
                    self.readTask,
                    1,
                    self.readAnalogBuffer,
                    self.readEncoderBuffer,
                    None,
                    self.readOtherBuffer
                )
            else: # use immediate I/O
                self.card.read(
                    self.READ_ANALOG_CHANNELS,
                    len(self.READ_ANALOG_CHANNELS),
                    self.READ_ENCODER_CHANNELS,
                    len(self.READ_ENCODER_CHANNELS),
                    None,
                    0,
                    self.READ_OTHER_CHANNELS,
                    len(self.READ_OTHER_CHANNELS),
                    self.readAnalogBuffer,
                    self.readEncoderBuffer,
                    None,
                    self.readOtherBuffer
                )
        except HILError as h:
            print(h.get_error_message())
        finally:
            # update external read buffers
            # self.timeStep = self.currentTimeStamp - self.prevTimeStamp
            # self.prevTimeStamp = self.currentTimeStamp
            self.motorCurrent = self.readAnalogBuffer[0]
            self.batteryVoltage = self.readAnalogBuffer[1]
            self.gyroscope = self.readOtherBuffer[0:3]
            self.accelerometer = self.readOtherBuffer[3:6]
            self.motorEncoder = self.readEncoderBuffer
            self.motorTach = self.readOtherBuffer[-1] * self.CPS_TO_MPS # is actually estimated qcar speed
            # if self.carType == 1:
                
            # elif self.carType == 2:
            #     self.motorTach = self.tachDerivator.send(self.readEncoderBuffer, self.timeStep) * QCar.CPS_TO_MPS

    def write(self, throttle, steering, LEDs=None):
        if not (self.hardware or self.io_task_running) and self.readMode:
            self._create_io_task()

        if self.carType in [1,0]:
            self.writeOtherBuffer[0] = -np.clip(steering, -0.6, 0.6)
            self.writePWMBuffer = -np.clip(throttle, -self.pwmLimit, self.pwmLimit)
            if LEDs is not None:
                self.writeOtherBuffer[1:9] = LEDs
        elif self.carType == 2:
            self.writeOtherBuffer[0] = np.clip(steering, -0.6, 0.6)
            self.writeOtherBuffer[1] = np.clip(throttle, -self.pwmLimit, self.pwmLimit)
            if LEDs is not None:
                # indicators
                self.writeDigitalBuffer[0:4] = LEDs[0:4]
                # brake lights
                self.writeDigitalBuffer[4] = LEDs[4]
                self.writeDigitalBuffer[5] = LEDs[4]
                self.writeDigitalBuffer[6] = LEDs[4]
                self.writeDigitalBuffer[7] = LEDs[4]
                # reverse lights
                self.writeDigitalBuffer[8] = LEDs[5]
                self.writeDigitalBuffer[9] = LEDs[5]
                # headlamps
                self.writeDigitalBuffer[10] = LEDs[6]
                self.writeDigitalBuffer[11] = LEDs[6]
                self.writeDigitalBuffer[12] = LEDs[6]
                self.writeDigitalBuffer[13] = LEDs[7]
                self.writeDigitalBuffer[14] = LEDs[7]
                self.writeDigitalBuffer[15] = LEDs[7]

        try:
            if self.carType in [1,0]:
                self.card.write(
                    None,
                    0,
                    self.WRITE_PWM_CHANNELS,
                    len(self.WRITE_PWM_CHANNELS),
                    None,
                    0,
                    self.WRITE_OTHER_CHANNELS,
                    len(self.WRITE_OTHER_CHANNELS),
                    None,
                    self.writePWMBuffer,
                    None,
                    self.writeOtherBuffer
                )
            elif self.carType == 2:
                self.card.write(
                    None,
                    0,
                    None,
                    0,
                    self.WRITE_DIGITAL_CHANNELS,
                    len(self.WRITE_DIGITAL_CHANNELS),
                    self.WRITE_OTHER_CHANNELS,
                    len(self.WRITE_OTHER_CHANNELS),
                    None,
                    None,
                    self.writeDigitalBuffer,
                    self.writeOtherBuffer
                )
        except HILError as h:
            print(h.get_error_message())

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.terminate()

class QCarCameras:
    """Class for accessing the QCar's CSI cameras.

    Args:
        frameWidth (int, optional): Width of the camera frame.
            Defaults to 820.
        frameHeight (int, optional): Height of the camera frame.
            Defaults to 410.
        frameRate (int, optional): Frame rate of the camera.
            Defaults to 30.
        enableRight (bool, optional): Whether to enable the right camera.
            Defaults to False.
        enableBack (bool, optional): Whether to enable the back camera.
            Defaults to False.
        enableLeft (bool, optional): Whether to enable the left camera.
            Defaults to False.
        enableFront (bool, optional): Whether to enable the front camera.
            Defaults to False.

    Attributes:
        csi (list): A list of Camera2D objects representing the enabled cameras.
        csiRight (Camera2D): The Camera2D object representing the right camera.
        csiBack (Camera2D): The Camera2D object representing the back camera.
        csiLeft (Camera2D): The Camera2D object representing the left camera.
        csiFront (Camera2D): The Camera2D object representing the front camera.
    """

    def __init__(
            self,
            frameWidth=820,
            frameHeight=410,
            frameRate=30,
            enableRight=False,
            enableBack=False,
            enableLeft=False,
            enableFront=False,
        ):
        """ Initializes QCarCameras object. """
        if QCAR_CONFIG['cartype'] in [0,1]:
            enable = [enableRight, enableBack, enableLeft, enableFront]
        elif QCAR_CONFIG['cartype']==2 :
            enable = [enableRight, enableBack, enableFront, enableLeft]
        self.csi = []
        for i in range(4):
            if enable[i]:
                if IS_PHYSICAL_QCAR:
                    cameraId = str(i)
                else:
                    cameraId = str(i) + "@tcpip://localhost:" + str(18961+i)

                self.csi.append(
                    Camera2D(
                        cameraId=cameraId,
                        frameWidth=frameWidth,
                        frameHeight=frameHeight,
                        frameRate=frameRate
                    )
                )
            else:
                self.csi.append(None)

        self.csiRight = self.csi[QCAR_CONFIG['csiRight']]
        self.csiBack = self.csi[QCAR_CONFIG['csiBack']]
        self.csiLeft = self.csi[QCAR_CONFIG['csiLeft']]
        self.csiFront = self.csi[QCAR_CONFIG['csiFront']]

    def readAll(self):
        """Reads frames from all enabled cameras."""
        flags=[]
        for c in self.csi:
            if c is not None:
                flags.append(c.read())
        return flags

    def __enter__(self):
        """Used for with statement."""

        return self

    def __exit__(self, type, value, traceback):
        """Used for with statement. Terminates all enabled cameras."""

        for c in self.csi:
            if c is not None:
                c.terminate()

    def terminate(self):
        """Used for with statement. Terminates all enabled cameras."""

        for c in self.csi:
            if c is not None:
                c.terminate()

class QCarLidar(Lidar):
    """QCarLidar class represents the LIDAR sensor on the QCar.

    Inherits from Lidar class in pal.utilities.lidar

    Args:
        numMeasurements (int): The number of LIDAR measurements.
        rangingDistanceMode (int): The ranging distance mode.
        interpolationMode (int): The interpolation mode.
        interpolationMaxDistance (int): The maximum interpolation distance.
        interpolationMaxAngle (int): The maximum interpolation angle.
        enableFiltering (bool): enable filtering RPLidar Data.
            Defaults to True.
        angularResolution (float): Desired angular resolution of the lidar
            data after filtering (if enabled).
    """

    def __init__(
            self,
            numMeasurements=384,
            rangingDistanceMode=2,
            interpolationMode=0,
            interpolationMaxDistance=0,
            interpolationMaxAngle=0,
            enableFiltering=True,
            angularResolution=1*np.pi/180
        ):
        """Initializes a new instance of the QCarLidar class.

        Args:
            numMeasurements (int): The number of LIDAR measurements.
            rangingDistanceMode (int): The ranging distance mode.
            interpolationMode (int): The interpolation mode.
            interpolationMaxDistance (int): The maximum interpolation distance.
            interpolationMaxAngle (int): The maximum interpolation angle.
            enableFiltering (bool): enable filtering RPLidar Data.
                Defaults to True.
            angularResolution (float): Desired angular resolution of the lidar
                data after filtering (if enabled).
        """

        # if IS_PHYSICAL_QCAR:
        #     self.url = (
        #         "serial-cpu://localhost:2?baud='115200',"
        #         + "word='8',parity='none',stop='1',flow='none',dsr='on'"
        #     )
        # else:
        #     self.url = "tcpip://localhost:18966"

        try:
            # QCar 1
            if IS_PHYSICAL_QCAR:
                self.url = QCAR_CONFIG['lidarurl']
            else:
                self.url = "tcpip://localhost:18966"

            super().__init__(
                type='RPLidar',
                numMeasurements=numMeasurements,
                rangingDistanceMode=rangingDistanceMode,
                interpolationMode=interpolationMode,
                interpolationMaxDistance=interpolationMaxDistance,
                interpolationMaxAngle=interpolationMaxAngle
            )

        except DeviceError as e:
            print("Lidar Error")
            print(e.get_error_message())


        # print(Lidar().url)

        self.enableFiltering = enableFiltering
        self.angularResolution = angularResolution
        self._phi = np.linspace(
            0,
            2*np.pi,
            np.int_(np.round(2*np.pi/self.angularResolution))
            )

    def read(self):
        """
        Reads data from the LIDAR sensor and applies filtering if enabled.
        """
        new=super().read()
        if self.enableFiltering:
            self.angles, self.distances = self.filter_rplidar_data(
                self.angles, self.distances
            )
        return new

    def filter_rplidar_data(self, angles, distances):
        """ Filters RP LIDAR data

        Filters RP LIDAR data by deleting invalid reads, eliminating duplicate
        reads, interpolating distances to regularly spaced angles, and
        filling gaps with zeros.

        Args:
            angles (numpy.ndarray): The array of measured angles in radians.
            distances (numpy.ndarray): The array of measured distances.

        Returns:
            phiMeas (numpy.ndarray): An array of regularly spaced angles
                in radians.
            rFiltered (numpy.ndarray): The filtered array of distances.
        """

        phiRes = self.angularResolution
        # Delete invalid reads
        ids = (distances==0).squeeze()
        rMeas = np.delete(distances,ids)
        phiMeas = np.delete(angles,ids)
        if phiMeas.size == 0: return phiMeas,rMeas

        # Flip angle direction from CW to CCW and add 90 deg offset
        #phiMeas = wrap_to_2pi(2.5*np.pi-phiMeas)

        # Eliminate duplicate reads and sort
        phiMeas, ids = np.unique(phiMeas,return_index=True)
        rMeas = rMeas[ids]

        # Interpolate distances to regularly spaced angles
        rFiltered = np.interp(
            self._phi,
            phiMeas,
            rMeas,
            period=2*np.pi
        )

        # Find gaps where measurements were missed
        ids = np.diff(phiMeas) > 1.1*phiRes
        ids_lb = np.append(ids,False)
        ids_ub = np.append(False,ids)

        # Fill gaps with zeros
        lb = np.int_(np.ceil(phiMeas[ids_lb]/phiRes))
        ub = np.int_(np.floor(phiMeas[ids_ub]/phiRes))
        for i in range(lb.size):
            rFiltered[lb[i]:ub[i]] = 0

        phiMeasMin = np.int_(np.round(phiMeas[0]/phiRes))
        phiMeasMax = np.int_(np.round(phiMeas[-1]/phiRes))
        rFiltered[0:phiMeasMin] = 0
        rFiltered[phiMeasMax+1:] = 0

        return self._phi.astype('float32'), rFiltered.astype('float32')

class QCarRealSense(Camera3D):
    """
    A class for accessing 3D camera data from the RealSense camera on the QCar.

    Inherits from Camera3D class in pal.utilities.vision

    Args:
        mode (str): Mode to use for capturing data. Default is 'RGB&DEPTH'.
        frameWidthRGB (int): Width of the RGB frame. Default is 1920.
        frameHeightRGB (int): Height of the RGB frame. Default is 1080.
        frameRateRGB (int): Frame rate of the RGB camera. Default is 30.
        frameWidthDepth (int): Width of the depth frame. Default is 1280.
        frameHeightDepth (int): Height of the depth frame. Default is 720.
        frameRateDepth (int): Frame rate of the depth camera. Default is 15.
        frameWidthIR (int): Width of the infrared (IR) frame. Default is 1280.
        frameHeightIR (int): The height of the IR frame. Default is 720.
        frameRateIR (int): Frame rate of the IR camera. Default is 15.
        readMode (int): Mode to use for reading data from the camera.
            Default is 1.
        focalLengthRGB (numpy.ndarray): RGB camera Focal length in pixels.
            Default is np.array([[None], [None]], dtype=np.float64).
        principlePointRGB (numpy.ndarray): Principle point of the RGB camera
            in pixels. Default is np.array([[None], [None]], dtype=np.float64).
        skewRGB (float): Skew factor for the RGB camera. Default is None.
        positionRGB (numpy.ndarray): An array of shape (3, 1) that holds the
            position of the RGB camera in the car's frame of reference.
        orientationRGB (numpy.ndarray): An array of shape (3, 3) that holds the
            orientation of the RGB camera in the car's frame of reference.
        focalLengthDepth (numpy.ndarray): An array of shape (2, 1) that holds
            the focal length of the depth camera.
        principlePointDepth (numpy.ndarray): An array of shape (2, 1) that
            holds the principle point of the depth camera.
        skewDepth (float, optional): Skew of the depth camera
        positionDepth (numpy.ndarray, optional): An array of shape (3, 1) that
            holds the position of the depth camera
        orientationDepth (numpy.ndarray): An array of shape (3, 3) that holds
            the orientation of the Depth camera in the car's reference frame.
    """
    def __init__(
            self,
            mode='RGB&DEPTH',
            frameWidthRGB=1920,
            frameHeightRGB=1080,
            frameRateRGB=30,
            frameWidthDepth=1280,
            frameHeightDepth=720,
            frameRateDepth=15,
            frameWidthIR=1280,
            frameHeightIR=720,
            frameRateIR=15,
            readMode=1,
            focalLengthRGB=np.array([[None], [None]], dtype=np.float64),
            principlePointRGB=np.array([[None], [None]], dtype=np.float64),
            skewRGB=None,
            positionRGB=np.array([[None], [None], [None]], dtype=np.float64),
            orientationRGB=np.array(
                [[None, None, None], [None, None, None], [None, None, None]],
                dtype=np.float64),
            focalLengthDepth=np.array([[None], [None]], dtype=np.float64),
            principlePointDepth=np.array([[None], [None]], dtype=np.float64),
            skewDepth=None,
            positionDepth=np.array([[None], [None], [None]], dtype=np.float64),
            orientationDepth=np.array(
                [[None, None, None], [None, None, None], [None, None, None]],
                dtype=np.float64)
        ):

        if IS_PHYSICAL_QCAR:
            deviceId = '0'
        else:
            deviceId = "0@tcpip://localhost:18965"
            frameWidthRGB = 640
            frameHeightRGB = 480
            frameRateRGB = 30
            frameWidthDepth = 640
            frameHeightDepth = 480
            frameRateDepth = 15
            frameWidthIR = 640
            frameHeightIR = 480
            frameRateIR = 30

        super().__init__(
            mode,
            frameWidthRGB,
            frameHeightRGB,
            frameRateRGB,
            frameWidthDepth,
            frameHeightDepth,
            frameRateDepth,
            frameWidthIR,
            frameHeightIR,
            frameRateIR,
            deviceId,
            readMode,
            focalLengthRGB,
            principlePointRGB,
            skewRGB,
            positionRGB,
            orientationRGB,
            focalLengthDepth,
            principlePointDepth,
            skewDepth,
            positionDepth,
            orientationDepth
        )


class QCarGPS:
    """A class that reads GPS data from the GPS server.

    In order to use this class, the qcarLidarToGPS must already be running.
    You can launch a qcarLidarToGPS server using the QCarLidarToGPS class
    defined in pal.products.qcar.

    Attributes:
        position (numpy.ndarray): Holds the most recent position
            read from the GPS. Format is [x, y, z].
        orientation (numpy.ndarray): Holds the most recent orientation
            read from the GPS. Format is [roll, pitch, yaw].

    Methods:
        read(): Reads GPS data from the GPS server and updates the position
            and orientation attributes.
        terminate(): Terminates the GPS client.

    Example:
        .. code-block:: python

            # Launch the qcarLidarToGPS server (if not already running)
            gpsServer = QCarLidarToGPS(initialPose=[0, 0, 0])

            # Create an instance of the QCarGPS class
            gps = QCarGPS()

            # Read GPS data
            gps.read()

            print('position = ' + gps.position)
            print('orientation = ' + gps.orientation)

            # Terminate the GPS client
            gps.terminate()

    """

    def __init__(self, initialPose=[0, 0, 0], calibrate=False):
        """
        Initializes the QCarGPS class with the initial pose of the QCar.

        Args:
            initialPose (list, optional): Initial pose of the QCar
                as [x0, y0, th0] (default [0, 0, 0]).
        """
        
        self._need_calibrate = calibrate
        if IS_PHYSICAL_QCAR:
            self.__initLidarToGPS(initialPose)

        self._timeout = Timeout(seconds=0, nanoseconds=1)

        # Setup GPS client and connect to GPS server
        self.position = np.zeros((3))
        self.orientation = np.zeros((3))

        self._gps_data = np.zeros((6), dtype=np.float32)
        self._gps_client = BasicStream(
            uri="tcpip://localhost:18967",
            agent='C',
            receiveBuffer=np.zeros(6, dtype=np.float32),
            sendBufferSize=1,
            recvBufferSize=(self._gps_data.size * self._gps_data.itemsize),
            nonBlocking=True
        )
        t0 = time.time()
        while not self._gps_client.connected:
            if time.time()-t0 > 5:
                print("Couldn't Connect to GPS Server")
                return
            self._gps_client.checkConnection()

        # Setup Lidar data client and connect to Lidar data server
        self.scanTime = 0
        self.angles = np.zeros(384)
        self.distances = np.zeros(384)

        self._lidar_data = np.zeros(384*2 + 1, dtype=np.float64)
        self._lidar_client = BasicStream(
            uri="tcpip://localhost:18968",
            agent='C',
            receiveBuffer=np.zeros(384*2 + 1, dtype=np.float64),
            sendBufferSize=1,
            recvBufferSize=8*(384*2 + 1),
            nonBlocking=True
        )
        t0 = time.time()
        while not self._lidar_client.connected:
            if time.time()-t0 > 5:
                print("Couldn't Connect to Lidar Server")
                return
            self._lidar_client.checkConnection()

        self.enableFiltering = True
        self.angularResolution = 1*np.pi/180
        self._phi = np.linspace(
            0,
            2*np.pi,
            np.int_(np.round(2*np.pi/self.angularResolution))
        )


    def __initLidarToGPS(self, initialPose):
        self.__initialPose = initialPose

        self.__stopLidarToGPS()

        if self._need_calibrate:
            self.__calibrate()
            # wait period to complete calibration completely
            time.sleep(16)
        if os.path.exists(os.path.join(os.getcwd(),'angles_new.mat')):
            self.__emulateGPS()
            time.sleep(4)
            print('GPS Server started.')
        else:
            print('Calibration files not found, please set the argument \'calibration\' to True.')
            exit(1)

    def __stopLidarToGPS(self):
        # Quietly stop qcarLidarToGPS if it is already running:
        # the -q flag kills the executable
        # the -Q flag kills quietly (no errors thrown if its not running)
        os.system(
            'quarc_run -t tcpip://localhost:17000 -q -Q '
            + QCAR_CONFIG['lidarToGps']
        )

    def __calibrate(self):
        """Calibrates the QCar at its initial position and heading."""

        print('Calibrating QCar at position ', self.__initialPose[0:2],
            ' (m) and heading ', self.__initialPose[2], ' (rad).')

        # setup the path to the qcarCaptureScan file
        captureScanfile = os.path.normpath(os.path.join(
            os.path.dirname(__file__),
            '../../../resources/applications/QCarScanMatching/'
                + QCAR_CONFIG['captureScan']
        ))

        os.system(
            'quarc_run -t tcpip://localhost:17000 '
            + captureScanfile + ' -d ' + os.getcwd()
        )

    def __emulateGPS(self):
        """Starts the GPS emulation using the qcarLidarToGPS executable."""

        # setup the path to the qcarLidarToGPS file
        lidarToGPSfile = os.path.normpath(os.path.join(
            os.path.dirname(__file__),
            '../../../resources/applications/QCarScanMatching/'
                + QCAR_CONFIG['lidarToGps']
        ))
        os.system(
            'quarc_run -r -t tcpip://localhost:17000 '
            + lidarToGPSfile + ' -d ' + os.getcwd()
            + ' -pose_0 ' + str(self.__initialPose[0])
            + ',' + str(self.__initialPose[1])
            + ',' + str(self.__initialPose[2])
        )

    def readGPS(self):
        """Reads GPS data from the server and updates the position
            and orientation attributes.

        Returns:
            bool: True if new GPS data was received.
        """
        recvFlag, bytesReceived = self._gps_client.receive(
            iterations=1,
            timeout=self._timeout)

        if recvFlag:
            self.position = self._gps_client.receiveBuffer[0:3]
            self.orientation = self._gps_client.receiveBuffer[3:6]

        return recvFlag

    def readLidar(self):
        """Reads GPS data from the server and updates the position
            and orientation attributes.

        Returns:
            bool: True if new GPS data was received.
        """
        recvFlag, bytesReceived = self._lidar_client.receive(
            iterations=1,
            timeout=self._timeout)

        if recvFlag:
            self.scanTime = self._lidar_client.receiveBuffer[0]
            self.distances = self._lidar_client.receiveBuffer[1:385]
            self.angles = self._lidar_client.receiveBuffer[385:769]

            self.angles, self.distances = self.filter_rplidar_data(
                self.angles,
                self.distances
            )

        return recvFlag


    def filter_rplidar_data(self, angles, distances):
        """ Filters RP LIDAR data

        Filters RP LIDAR data by deleting invalid reads, eliminating duplicate
        reads, interpolating distances to regularly spaced angles, and
        filling gaps with zeros.

        Args:
            angles (numpy.ndarray): The array of measured angles in radians.
            distances (numpy.ndarray): The array of measured distances.

        Returns:
            phiMeas (numpy.ndarray): An array of regularly spaced angles
                in radians.
            rFiltered (numpy.ndarray): The filtered array of distances.
        """

        phiRes = self.angularResolution
        # Delete invalid reads
        ids = (distances==0)
        phiMeas = np.delete(angles,ids)
        rMeas = np.delete(distances,ids)
        if phiMeas.size == 0: return phiMeas,rMeas

        # Flip angle direction from CW to CCW and add 90 deg offset
        #phiMeas = wrap_to_2pi(2.5*np.pi-phiMeas)

        # Eliminate duplicate reads and sort
        phiMeas, ids = np.unique(phiMeas,return_index=True)
        rMeas = rMeas[ids]

        # Interpolate distances to regularly spaced angles
        rFiltered = np.interp(
            self._phi,
            phiMeas,
            rMeas,
            period=2*np.pi
        )

        # Find gaps where measurements were missed
        ids = np.diff(phiMeas) > 1.1*phiRes
        ids_lb = np.append(ids,False)
        ids_ub = np.append(False,ids)

        # Fill gaps with zeros
        lb = np.int_(np.ceil(phiMeas[ids_lb]/phiRes))
        ub = np.int_(np.floor(phiMeas[ids_ub]/phiRes))
        for i in range(lb.size):
            rFiltered[lb[i]:ub[i]] = 0

        phiMeasMin = np.int_(np.round(phiMeas[0]/phiRes))
        phiMeasMax = np.int_(np.round(phiMeas[-1]/phiRes))
        rFiltered[0:phiMeasMin] = 0
        rFiltered[phiMeasMax+1:] = 0

        return self._phi, rFiltered


    def terminate(self):
        """ Terminates the GPS client. """
        self._gps_client.terminate()
        self._lidar_client.terminate()
        if IS_PHYSICAL_QCAR:
            self.__stopLidarToGPS()

    def __enter__(self):
        """ Used for with statement. """
        return self

    def __exit__(self, type, value, traceback):
        """ Used for with statement. Terminates the GPS client. """
        self.terminate()
    






import numpy as np
from quanser.hardware import HIL, HILError
import json
from os.path import split,abspath,join,realpath,dirname
from os import getlogin
import platform

class QCar_check():
    def __init__(self,IS_PHYSICAL):
        self.card = HIL()
        # initialize with QCar 1 parameters
        self.dict={'cartype':1,
                   'carname':'qcar',
                   'lidarurl':"serial-cpu://localhost:2?baud='115200',word='8',parity='none',stop='1',flow='none',dsr='on'",
                   'WHEEL_RADIUS': 0.066/2,
                   'WHEEL_BASE':0.256,
                   "PIN_TO_SPUR_RATIO":(13.0*19.0) / (70.0*37.0), ### same for both
                   "WRITE_PWM_CHANNELS": [0], #np.array([0], dtype=np.int32),
                   'WRITE_OTHER_CHANNELS':[1000, 11008, 11009, 11010, 11011, 11000, 11001, 11002, 11003],
                                           #np.array([1000, 11008, 11009, 11010, 11011, 11000, 11001, 11002, 11003], dtype=np.int32),
                   'WRITE_DIGITAL_CHANNELS':[-1], # None
                   'writePWMBuffer': 1, #np.zeros(1, dtype=np.float64),
                   'writeDigitalBuffer': 1, # None
                   'writeOtherBuffer': 9, #np.zeros(9, dtype=np.float64),
                   'READ_ANALOG_CHANNELS':[5, 6], #np.array([5, 6], dtype=np.int32),
                   'READ_ENCODER_CHANNELS':[0], #np.array([0], dtype=np.uint32), ### same for both
                   'READ_OTHER_CHANNELS':[3000, 3001, 3002, 4000, 4001, 4002, 14000], 
                                        #np.array([3000, 3001, 3002, 4000, 4001, 4002, 14000],dtype=np.int32), ### same for both
                   'readAnalogBuffer': 2, #np.zeros(2, dtype=np.float64), ### same for both
                   'readEncoderBuffer': 1, #np.zeros(1, dtype=np.int32), ### same for both
                   'readOtherBuffer': 7, #np.zeros(7, dtype=np.float64), ### same for both
                   'csiRight': 0,
                   'csiBack':1,
                   'csiLeft':2,
                   'csiFront':3,
                   'lidarToGps':'qcarLidarToGPS.rt-linux_nvidia',
                   'captureScan':'qcarCaptureScan.rt-linux_nvidia'
                   }
        if IS_PHYSICAL:
            self.check_car_type()
        else:
            self.dict['cartype']=0
        self.create_config()

    def check_car_type(self):
        try:
            self.card.open("qcar", "0")
            if self.card.is_valid():
                pass
        except HILError as e:            
            if str(e) == '-986':
                self.dict['cartype']=0
                pass
            else:
                print(e.get_error_message())

        if self.dict['cartype']==0:
            try:
                self.card.open("qcar2", "0")
                if self.card.is_valid():
                    self.dict['cartype']=2
                    self.dict['carname']='qcar2'
                    self.dict['lidarurl']="serial-cpu://localhost:1?baud='256000',word='8',parity='none',stop='1',flow='none',dsr='on'"
                    self.dict['WHEEL_RADIUS'] = 0.066/2
                    self.dict['WHEEL_BASE'] = 0.256
                    self.dict['PIN_TO_SPUR_RATIO'] = (13.0*19.0) / (70.0*37.0)  ### same for both
                    self.dict['WRITE_PWM_CHANNELS'] = [-1] # None
                    self.dict['WRITE_OTHER_CHANNELS'] = [1000, 11000] #np.array([1000, 11000],dtype=np.int32)
                    self.dict['WRITE_DIGITAL_CHANNELS'] = [17, 18, 25, 26, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24]
                                                        #np.array([17, 18, 25, 26, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24],dtype=np.int32)
                    self.dict['writePWMBuffer'] = 1 # None
                    self.dict['writeDigitalBuffer'] = 16 #np.zeros(16, dtype=np.int8)
                    self.dict['writeOtherBuffer'] = 2 #np.zeros(2, dtype=np.float64)
                    self.dict['READ_ANALOG_CHANNELS'] = [4, 2] #np.array([4, 2], dtype=np.int32)
                    self.dict['READ_ENCODER_CHANNELS'] = [0] #np.array([0], dtype=np.uint32) ### same for both
                    self.dict['READ_OTHER_CHANNELS'] = [3000, 3001, 3002, 4000, 4001, 4002, 14000]
                                                     #np.array([3000, 3001, 3002, 4000, 4001, 4002, 14000],dtype=np.int32) ### same for both
                    self.dict['readAnalogBuffer'] = 2 #np.zeros(2, dtype=np.float64) ### same for both
                    self.dict['readEncoderBuffer'] = 1 #np.zeros(1, dtype=np.int32) ### same for both
                    self.dict['readOtherBuffer'] = 7 #np.zeros(7, dtype=np.float64) ### same for both
                    self.dict['csiRight'] = 0
                    self.dict['csiBack'] = 1
                    self.dict['csiLeft'] = 3
                    self.dict['csiFront'] = 2
                    self.dict['lidarToGps']='qcar2LidarToGPS.rt-linux_qcar2'
                    self.dict['captureScan']='qcar2CaptureScan.rt-linux_qcar2'
            except HILError as e:            
                if str(e) == '-986':
                    self.dict['cartype']=0
                    pass
                else:
                    print(e.get_error_message())

    def create_config(self):
        name='qcar_config.json'
        qcar_env=dirname(realpath(__file__))
        save_path=join(qcar_env,name)
        with open(save_path,'w') as outfile:
            json.dump(self.dict,outfile)
        self.card.close()

if __name__ == '__main__':
    IS_PHYSICAL_QCAR = ('nvidia' == getlogin()) \
    and ('aarch64' == platform.machine())
    test=QCar_check(IS_PHYSICAL_QCAR)
    qcar_env=dirname(realpath(__file__))
    read_path=join(qcar_env,'qcar_config.json')
    with open(read_path, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        for i in json_object:
            print (i,type(json_object[i]))






# Traffic Light Client
"""
This script provides a user-friendly library to interact with traffic lights. 
It establishes a connection to a Raspberry Pi Zero W running a micro-controller that controls the traffic light LEDs. 
The library offers functionalities to control the traffic lights manually (setting them to red, yellow, or green) or switch them to an automatic timed mode. 
It also allows for turning off the lights and starting or stopping a stream.

"""

import urllib.request, sys
from urllib.error import HTTPError, URLError
from socket import timeout
import time

class TrafficLight():
    
    def __init__(self, ip):
        """ 
        Connection to the lights
        
        Args:
            ip (string): 192.168.2.xxx when connected to Quanser_UVS network
        """
        self.url = "http://" + ip + ":5000/"

    def status(self):
        """
        Check and return the status of LEDs in use

        Args:
            None
        
        Returns (str):
            0 -> No lit LEDs, 1 -> Red LED lit, 2 -> Yellow LED lit, 3 -> Green LED lit
        """ 
        request = 'status'
        response = self._sendreq(self.url + request)
        return response
    
    def shutdown(self):
        """
        Shutdown the Traffic Light
        
        Args:
            None
        """
        request = 'shutdown'
        response = self._sendreq(self.url + request)
        print('Shutting Down Traffic Light ' + self.url)
        return response
    
    def auto(self):
        """
        Set the Traffic lights to automatic mode

        Args:
            None

        Automatic mode: The traffic lights cycle through the following sequence:
        Red light illuminated for 30 seconds.
        Green light illuminated for 30 seconds.
        Yellow light illuminated for 3 seconds.
        The cycle then repeats.
        """
        response = self.timed(0,0,0) # the r/y/g flags are set to 0s to trigger automatic mode
        return response
        
    def red(self):
        """
        Set Traffic Light to Red 
        
        Args:
            None

        Two LEDs cannot be turned on at the same time
        """
        request = 'immediate/red'
        response = self._sendreq(self.url + request)
        return response
     
    def yellow(self):
        """
        Set Traffic Light to Yellow 
        
        Args:
            None

        Two LEDs cannot be turned on at the same time
        """
        request = 'immediate/yellow'
        response = self._sendreq(self.url + request)
        return response
        
    def green(self):
        """
        Set Traffic Light to Green 
        
        Args:
            None

        Two LEDs cannot be turned on at the same time
        """
        request = 'immediate/green'
        response = self._sendreq(self.url + request)
        return response
     
    def color(self, color):
        """
        Function to set one of the traffic light LEDs on.
        Can only have one color ON at a time. 
        
        Args:
            color(int): 0-off; 1-red; 2-yellow; 3-green
        """
        if int(color) == 0:
            response = self.off()
        elif int(color) == 1:
            response = self.red()
        elif int(color) == 2:
            response = self.yellow()
        elif int(color) == 3:
            response = self.green()
        else:
            response = self.off()
        return response
 
    def timed(self, red = 30, yellow = 3, green = 30):
        """
        Set custom timed cycle for the Traffic Lights LEDs 
        
        Args:
            red (int): x seconds where red will be on
            yellow (int): x seconds where yellow will be on
            green (int): x seconds where green will be on

        Two LEDs cannot be turned on at the same time. 
        Default mode, The timed lights cycle set red -> 30s, yellow -> 3s & green -> 30s 
        """
        self.off()
        time.sleep(0.25)
        request = "timed/" + str(red) + "/" + str(yellow) + "/" + str(green)
        response = self._sendreq(self.url + request)
        return response
    
    def off(self):
        """
        Turn the LEDs off without shutting down
        
        Args:
            None
        """
        request = 'immediate/off'
        response = self._sendreq(self.url + request)
        return response
    
    def start_stream(self):
        """
        Connect to Matlab/Simulink using QUARC Streaming API
        
        Args:
            None

        Closes the serial port in python so it can be accessed 
        by Matlab/Simulink to connect/communicate with the lights
        """
        self.off()
        request = 'start_stream'
        response = self._sendreq(self.url + request)
        return response
    
    def stop_stream(self):
        """
        Reconnects back to Python to use TrafficLight functions
        
        Args:
            None

        Reopens the serial port in python so all the methods 
        in TrafficLight can resume communicating with the lights 
        """
        request = 'close_stream'
        response = self._sendreq(self.url + request)
        return response
    
    def isStreaming(self):
        """
        Check if the Streaming connection is Open
        
        Args:
            None
        
        Return:
            (string): Returns the status of streaming connection

        Check if the streaming connection is already open. 
        If you want to use the methods in TrafficLight, 
        the stream connection needs to be closed first.
        """
        request = 'check_stream'
        streaming = self._sendreq(self.url + request)
        if(streaming == '1' or streaming == '0'):
            return 'Streaming connection is open' if streaming == '1' else 'Streaming connection is NOT open'
        else:
            return streaming
        
    #Send the formatted request
    def _sendreq(self, url):
        #Format the HTTP get request with a timeout 
        # of 1s to account for async tasks that will not return
        response = "Call complete!"
        try:
            if url.find("stream") == -1:
                timeoutURL=1
            else:
                timeoutURL=4
            response = urllib.request.urlopen(url, timeout=timeoutURL).read().decode('utf-8')
        #If the URL is not correct
        except (HTTPError, URLError) as error:
            if(self.isStreaming() == '1'):
                response = "Streaming is open, Close the stream to use TrafficLight() functions"
            else:
                response = "Error endpoint not found at " + url
        #If the request was not expected to return the call it complete, otherwise flag a timeout
        except timeout:
            if url.find("timed") == -1:
                if(self.isStreaming() == '1'):
                    response = "Streaming is open, Close the stream to use TrafficLight() functions"
                else:
                    response = "Call timed out"
            else:
                response = "Async call complete"

        return response







import os


__imagesDirPath = os.path.normpath(os.path.join(
            os.path.dirname(__file__), '../../../resources/images/'))

SDCS_CITYSCAPE = os.path.normpath(
    os.path.join(__imagesDirPath, 'sdcs_cityscape.png'))

SDCS_CITYSCAPE_SMALL = os.path.normpath(
    os.path.join(__imagesDirPath, 'sdcs_cityscape_small.png'))








import os


__rtModelDirPath = os.environ['RTMODELS_DIR']

# QCar RT Models
QCAR = os.path.normpath(
    os.path.join(__rtModelDirPath, 'qcar/QCar_Workspace'))

QCAR_STUDIO = os.path.normpath(
    os.path.join(__rtModelDirPath, 'qcar/QCar_Workspace_Studio'))

QBOT_PLATFORM = os.path.normpath(
    os.path.join(__rtModelDirPath, 'QBotPlatform/QBotPlatform_Workspace'))

QBOT_PLATFORM_DRIVER = os.path.normpath(
    os.path.join(__rtModelDirPath, 'QBotPlatform/qbot_platform_driver_virtual'))















"""gamepad: A module for simplifying interactions with gamepads (controllers).

This module provides classes and utilities to facilitate working with common
types of gamepads, such as the Logitech F710. It is designed to make it easy
to interface with these devices, read their states, and manage their
connections.
"""
from quanser.devices import GameController
import numpy as np
import platform


class LogitechF710:
    """Class for interacting with the Logitech Gamepad F710.

    This class opens a GameController device and establishes a connection
    to it. It provides methods for reading gamepad states and terminating
    the connection.

    Attributes:
        system (str): The current operating system name.
        mode (int): The gamepad mapping mode, depending on the OS.
        flag (bool): A flag used for trigger updates.
        leftJoystickX (float): Left joystick right/left value.
        leftJoystickY (float): Left joystick up/down value.
        rightJoystickX (float): Right joystick right/left value.
        rightJoystickY (float): Right joystick up/down value.
        trigger (float): Trigger value.
        buttonA (int): Button A state.
        buttonB (int): Button B state.
        buttonX (int): Button X state.
        buttonY (int): Button Y state.
        buttonLeft (int): Left button state.
        buttonRight (int): Right button state.
        up (int): Up arrow state.
        right (int): Right arrow state.
        left (int): Left arrow state.
        down (int): Down arrow state.
    """

    system = platform.system()

    if system == 'Windows':
        mode = 0
    elif system == 'Linux':
        mode = 1
    else:
        mode = -1

    flag = False

    # Continuous axis
    leftJoystickX = 0
    leftJoystickY = 0
    rightJoystickX = 0
    rightJoystickY = 0
    trigger = 0

    # Buttons
    buttonA = 0
    buttonB = 0
    buttonX = 0
    buttonY = 0
    buttonLeft = 0
    buttonRight = 0

    # Arrow keys
    up = 0
    right = 0
    left = 0
    down = 0

    def __init__(self, deviceID=1):
        """Initialize and open a connection to a LogitechF710 GameController.
        """
        self.gameController = GameController()
        self.gameController.open(deviceID)

    def read(self):
        """Update the gamepad states by polling the GameController.

        The updated states are:
        Continuous:
            leftJoystickX: Left Joystick (up/down) (-1 to 1)
            leftJoystickY: Left Joystick (right/left) (-1 to 1)
            rightJoystickX: Right Joystick (up/down) (-1 to 1)
            rightJoystickY: Right Joystick (right/left) (-1 to 1)
            trigger: Left and right triggers
                (0.5 -> 0 for right trigger, 0.5 -> 1 for left trigger)

        Buttons:
            buttonA, buttonB, buttonX, buttonY, buttonLeft, buttonRight
            up, right, down, left
        """
        data, new = self.gameController.poll()

        # Update the lateral and longitudinal axis
        self.leftJoystickX = -1 * data.x
        self.leftJoystickY = -1 * data.y
        self.rightJoystickX = -1 * data.rx
        self.rightJoystickY = -1 * data.ry

        # Trigger mapping for a Windows-based system
        if self.mode == 0:
            if data.z == 0 and not self.flag:
                self.trigger = 0
            else:
                self.trigger = 0.5 + 0.5 * data.z
                self.flag = True

        # Trigger mapping for a Linux-based system
        if self.mode == 1:
            if data.rz == 0 and not self.flag:
                self.trigger = 0
            else:
                self.trigger = 0.5 + 0.5 * data.rz
                self.flag = True

        # Update the buttons
        self.buttonA = int(data.buttons & (1 << 0))
        self.buttonB = int((data.buttons & (1 << 1)) / 2)
        self.buttonX = int((data.buttons & (1 << 2)) / 4)
        self.buttonY = int((data.buttons & (1 << 3)) / 8)
        self.buttonLeft = int((data.buttons & (1 << 4)) / 16)
        self.buttonRight = int((data.buttons & (1 << 5)) / 32)

        # Update the arrow keys
        val = 180 * data.point_of_views[0] / np.pi
        self.up = 0
        self.right = 0
        self.left = 0
        self.down = 0
        if val >= 310 or (val >= 0 and val < 50):
            self.up = 1
        if val >= 40 and val < 140:
            self.right = 1
        if val >= 130 and val < 230:
            self.down = 1
        if val >= 220 and val < 320:
            self.left = 1

        return new

    def terminate(self):
        """Terminate the GameController connection."""
        self.gameController.close()

















        """lidar: A module for simplifying interactions with common LiDAR devices.

This module provides a Lidar class to facilitate working with common LiDAR
devices, such as RPLidar and Leishen M10. It is designed to make it easy to
interface with these devices, read their measurements, and manage their
connections.
"""
import numpy as np
from quanser.devices import (
    RangingMeasurements,
    RangingMeasurementMode,
    DeviceError,
    RangingDistance
)

class Lidar():
    """A class for interacting with common LiDAR devices.

    This class provides an interface for working with LiDAR devices, such as
    RPLidar and Leishen MS10 or M10P. It simplifies the process of reading measurements
    and managing connections with these devices.

    Attributes:
        numMeasurements (int): The number of measurements per scan.
        distances (numpy.ndarray): An array containing distance measurements.
        angles (numpy.ndarray): An array containing the angle measurements.

    Example usage:

    .. code-block:: python

        from lidar import Lidar

        # Initialize a Lidar device (e.g. RPLidar)
        lidar_device = Lidar(type='RPLidar')

        # Read LiDAR measurements
        lidar_device.read()

        # Access measurement data
        print((lidar_device.distances, lidar_device.angles))

        # Terminate the LiDAR device connection
        lidar_device.terminate()

    """

    def __init__(
            self,
            type='RPLidar',
            numMeasurements=384,
            rangingDistanceMode=2,
            interpolationMode=0,
            interpolationMaxDistance=0,
            interpolationMaxAngle=0
        ):
        """Initialize a Lidar device with the specified configuration.

        Args:
            type (str, optional): The type of LiDAR device
                ('RPLidar' or 'LeishenMS10' or 'LeishenM10P'). Defaults to 'RPLidar'.
            numMeasurements (int, optional): The number of measurements
                per scan. Defaults to 384.
            rangingDistanceMode (int, optional): Ranging distance mode
                (0: Short, 1: Medium, 2: Long). Defaults to 2.
            interpolationMode (int, optional): Interpolation mode
                (0: Normal, 1: Interpolated). Defaults to 0.
            interpolationMaxDistance (float, optional): Maximum distance
                for interpolation. Defaults to 0.
            interpolationMaxAngle (float, optional): Maximum angle for
                interpolation. Defaults to 0.
        """

        self.numMeasurements = numMeasurements
        self.distances = np.zeros((numMeasurements,1), dtype=np.float32)
        self.angles = np.zeros((numMeasurements,1), dtype=np.float32)
        self._measurements = RangingMeasurements(numMeasurements)
        self._rangingDistanceMode = rangingDistanceMode
        self._interpolationMode = interpolationMode
        self._interpolationMaxDistance = interpolationMaxDistance
        self._interpolationMaxAngle = interpolationMaxAngle

        if type.lower() == 'rplidar':
            self.type = 'RPLidar'
            from quanser.devices import RPLIDAR as RPL
            self._lidar = RPL()
            if not hasattr(self, "url"):
                self.url = ("serial-cpu://localhost:2?baud='115200',"
                        "word='8',parity='none',stop='1',flow='none',dsr='on'")
            # Open the lidar device with ranging mode settings.
            self._lidar.open(self.url, self._rangingDistanceMode)

        elif type.lower() == 'leishenms10':
            self.type = 'LeishenMS10'
            from quanser.devices import LeishenMS10
            self._lidar = LeishenMS10()
            if not hasattr(self, "url"):
                self.url = ("serial-cpu://localhost:2?baud='460800',"
                        "word='8',parity='none',stop='1',flow='none'")
            self._lidar.open(self.url, samples_per_scan = self.numMeasurements)

        elif type.lower() == 'leishenm10p':
            self.type = 'LeishenM10P'
            from quanser.devices import LeishenM10P
            self._lidar = LeishenM10P()
            if not hasattr(self, "url"):
                self.url = ("serial://localhost:0?baud='512000',"
                        "word='8',parity='none',stop='1',flow='none',device='/dev/lidar'") #serial://localhost:0?device='/dev/lidar',baud='512000',word='8',parity='none',stop='1',flow='none'
            self._lidar.open(self.url, samples_per_scan = self.numMeasurements)

        else:
            # TODO: Assert error
            return

        try:
            # Ranging distance mode check
            if rangingDistanceMode == 2:
                self._rangingDistanceMode = RangingDistance.LONG
            elif rangingDistanceMode == 1:
                self._rangingDistanceMode = RangingDistance.MEDIUM
            elif rangingDistanceMode == 0:
                self._rangingDistanceMode = RangingDistance.SHORT
            else:
                print('Unsupported Ranging Distance Mode provided.'
                        'Configuring LiDAR in Long Range mode.')
                self._rangingDistanceMode = RangingDistance.LONG

            # Interpolation check (will be used in the read method)
            if interpolationMode == 0:
                self._interpolationMode = RangingMeasurementMode.NORMAL
            elif interpolationMode == 1:
                self._interpolationMode = RangingMeasurementMode.INTERPOLATED
                self._interpolationMaxAngle = interpolationMaxAngle
                self._interpolationMaxDistance = interpolationMaxDistance
            else:
                print('Unsupported Interpolation Mode provided.'
                        'Configuring LiDAR without interpolation.')
                self._interpolationMode = RangingMeasurementMode.NORMAL

        except DeviceError as de:
            if de.error_code == -34:
                pass
            else:
                print(de.get_error_message())

    def read(self):
        """Read a scan and store the measurements

        Read a scan from the LiDAR device and store the measurements in the
        'distances' and 'angles' attributes.
        """
        flag = False
        try:
            numValues = self._lidar.read(
                self._interpolationMode,
                self._interpolationMaxDistance,
                self._interpolationMaxAngle,
                self._measurements
            )
            if numValues > 0:
                self.distances = np.array(self._measurements.distance)
                self.angles = np.array(self._measurements.heading)
                flag = True
        except DeviceError as de:
            if de.error_code == -34:
                pass
            else:
                print(de.get_error_message())
        finally:
            return flag

    def terminate(self):
        """Terminate the LiDAR device connection correctly."""
        try:
            self._lidar.close()
            # print("lidar closed")
        except DeviceError as de:
            if de.error_code == -34:
                pass
            else:
                print(de.get_error_message())

    def __enter__(self):
        """Return self for use in a 'with' statement."""
        return self

    def __exit__(self, type, value, traceback):
        """
        Terminate the LiDAR device connection when exiting a 'with' statement.
        """
        self.terminate()


















        """math.py: A module providing handy functions for common mathematical tasks.

This module contains a collection of utility functions for completing various
mathematical operations, such as wrapping angles, filtering signals, and
performing numerical differentiation. These functions are designed to simplify
and streamline the process of working with mathematical operations in a variety
of applications, including data processing, analysis, and control systems.
"""
import numpy as np


TWO_PI = 2 * np.pi
"""A constant representing the value of two times pi, for convenience."""


def wrap_to_2pi(th: float) -> float:
    """Wrap an angle in radians to the interval [0, 2*pi).

    Args:
        th (float): The angle to be wrapped in radians.

    Returns:
        float: The wrapped angle in radians.
    """
    return np.mod(np.mod(th, TWO_PI) + TWO_PI, TWO_PI)


def wrap_to_pi(th: float) -> float:
    """Wrap an angle in radians to the interval [-pi, pi).

    Args:
        th (float): The angle to be wrapped in radians.

    Returns:
        float: The wrapped angle in radians.
    """
    th = th % TWO_PI
    th = (th + TWO_PI) % TWO_PI
    if th > np.pi:
        th -= TWO_PI
    return th


def angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute the angle in radians between two vectors.

    Args:
        v1 (numpy.ndarray): A 2-dimensional input vector.
        v2 (numpy.ndarray): A 2-dimensional input vector.

    Returns:
        float: The angle between the two input vectors in radians.
    """
    num = np.dot(v1, v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    try:
        th = np.arccos(num / den)
    except:
        th = 0
    return th


def signed_angle(v1: np.ndarray, v2: np.ndarray = None) -> float:
    """Find the signed angle between two vectors

    Compute the signed angle in radians between two vectors or the angle of
    a single vector with respect to the x-axis if v2 is None.

    Args:
        v1 (numpy.ndarray): A 2-dimensional input vector.
        v2 (numpy.ndarray, optional): A 2-dimensional input vector.
            Defaults to None.

    Returns:
        float: The signed angle between the two input vectors in radians, or
            the angle of v1 in radians with respect to the x-axis.

    Notes:
        - If v2 is None, then the function computes the angle of v1 with
            respect to the x-axis.
        - The function uses the `wrap_to_pi` function to ensure that the signed
            angle is in the range [-pi, pi).
    """
    if v2 is None:
        return np.arctan2(np.arctan2(v1[1], v1[0]))
    return wrap_to_pi(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))



def get_mag_and_angle(v):
    mag = np.linalg.norm(v)
    alpha = np.arctan2(v[1], v[0])
    return mag, alpha


def find_overlap(a: np.ndarray, b: np.ndarray, i: int, j: int) -> tuple:
    """Finds slices that correspond to overlapping cells of two 2D numpy arrays

    Args:
        a (numpy.ndarray): First input array of shape (m, n).
        b (numpy.ndarray): Second input array of shape (p, q).
        i (int): Row index in a's index-space where b[0,0] is located.
        j (int): Column index in a's index-space where b[0,0] is located.

    Returns:
        Tuple of two slices: aSlice and bSlice.
        aSlice (tuple): Slice of a corresponding to the overlapping cells.
        bSlice (tuple): Slice of b corresponding to the overlapping cells.

    Notes:
        - The function assumes that the first element of b is located at
            position (i, j) in a's index-space.
        - The function clips the indices of a and b to ensure that they are
            within bounds.
    """

    ma,na = a.shape
    mb,nb = b.shape

    ia = np.int_(np.array([np.clip(i,0,ma),np.clip(i+mb,0,ma)]))
    ja = np.int_(np.array([np.clip(j,0,na),np.clip(j+nb,0,na)]))
    ib = np.int_(ia - i)
    jb = np.int_(ja - j)

    aSlice = (slice(ia[0],ia[1]),slice(ja[0],ja[1]))
    bSlice = (slice(ib[0],ib[1]),slice(jb[0],jb[1]))

    return aSlice, bSlice


def ddt_filter(u, state, A, Ts):
    # d/dt with filtering:
    # y = As*u/(s+A)
    #
    # Z-domain with Tustin transform:
    # y = (2AZ - 2A)/((2+AT)z + (AT-2))
    #
    # Divide through by z to get z^-1 terms the convert to time domain
    # y_k1(AT-2) + y_k(AT+2) = 2Au_k - 2Au_k1
    # y_k = 1/(AT+2) * ( 2Au_k - 2Au_k1 - y_k1(AT-2) )
    #
    # y - output
    # u - input
    # state - previous state returned by this function -- initialize to np.array([0,0], dtype=np.float64)
    # Ts - sample time in seconds
    # A - filter bandwidth in rad/s

    y = 1/(A*Ts+2)*(2*A*u - 2*A*state[0] - state[1]*(A*Ts - 2));

    state[0] = u;
    state[1] = y;

    return y, state

def lp_filter(u, state, A, Ts):
    # y = A*u/(s+A)
    #
    # y - output
    # u - input
    # state - previous state returned by this function -- initialize to np.array([0,0], dtype=np.float64)
    # Ts - sample time in seconds
    # A - filter bandwidth in rad/s

    y = (u*Ts*A + state[0]*Ts*A - state[1]*(Ts*A - 2) ) / (2 + Ts*A)

    state[0] = u
    state[1] = y

    return y, state


class SignalGenerator:
    """Class object consisting of common signal generators"""

    def sine(self, amplitude, angularFrequency, phase=0, mean=0):
        """
        This function outputs a sinusoid wave based on the provided timestamp.

        For example:

        .. code-block:: python

            generatorSine = Signal_Generator().sine(2, pi/2)
            initialOutput = next(generatorSine)
            while True:
                timestamp = your_timing_function()
                output = generatorSine.send(timestamp)
        """

        output = amplitude*np.sin(phase) + mean
        while True:
            timestamp = yield output
            output = amplitude*np.sin(
                angularFrequency*timestamp + phase) + mean

    def cosine(self, amplitude, angularFrequency, phase=0, mean=0):
        """Outputs a cosinusoid wave based on the provided timestamp.

        For example:

        .. code-block:: python

            generatorCosine = Signal_Generator().cosine(2, pi/2)
            initialOutput = next(generatorCosine)
            while True:
                timestamp = your_timing_function()
                output = generatorCosine.send(timestamp)
        """

        output = amplitude*np.sin(phase + np.pi/2) + mean
        while True:
            timestamp = yield output
            output = amplitude*np.sin(
                angularFrequency*timestamp + phase + np.pi/2) + mean

    def PWM(self, frequency, width, phase=0):
        """This function outputs a PWM wave based on the provided timestamp.

        For example:

        .. code-block:: python

            generatorPWM = Signal_Generator().PWM(2, 0.5)
            initialOutput = next(generatorPWM)
            while True:
                timestamp = your_timing_function()
                output = generatorPWM.send(timestamp)
        """

        period = 1/frequency
        if phase%1 >= width:
            output = 0
        else:
            output = 1
        while True:
            timestamp = yield output
            marker = ( ( (timestamp % period) / period ) + phase ) % 1
            if marker > width:
                output = 0
            else:
                output = 1

    def square(self, amplitude, period):
        """Outputs a square wave based on the provided timestamp."""
        val = 0
        while True:
            timestamp = yield val
            if timestamp % period < period/2.0:
                val = amplitude
            else:
                val = -amplitude

class Calculus:
    """Class object consisting of basic derivative and integration functions"""

    def differentiator(self, dt, x0=0):
        """Finite-difference-based numerical derivative.

        Provide the sample time (s), and use the .send(value) method
        to differentiate.

        For example:

        .. code-block:: python

            diff_1 = Calculus().differentiator(0.01)
            while True:
                value = some_random_function()
                value_derivative = diff_1.send(value)

        Multiple differentiators can be defined for different signals.
        Do not use the same handle to differentiate different value signals.
        """
        derivative = 0
        while True:
            x = yield derivative
            derivative = (x - x0)/dt
            x0 = x

    def differentiator_variable(self, dt, x0=0):
        """Finite-difference-based numerical derivative.

        Provide the sample time (s), and use the .send(value) method
        to differentiate.

        For example:

        .. code-block:: python

            diff_1 = Calculus().differentiator_variable(0.01)
            while True:
                value = some_random_function()
                time_step = some_time_taken
                value_derivative = diff_1.send((value, time_step))

        Multiple differentiators can be defined for different signals.
        Do not use the same handle to differentiate different value signals.
        """
        derivative = 0
        while True:
            x, dt = yield derivative
            derivative = (x - x0)/dt
            x0 = x

    def integrator(self, dt, integrand=0, saturation=None):
        """Iterative numerical integrator.

        Provide the sample time (s), and use the .send(value) method
        to integrate.

        For example:

        .. code-block:: python

            intg_1 = Calculus().integrator(0.01)
            while True:
                value = some_random_function()
                value_integral = intg_1.send(value)

        Multiple integrators can be defined for different signals.
        Do not use the same handle to integrate different value signals.
        """
        self.saturation = saturation
        while True:
            x = yield integrand
            if self.saturation is None:
                integrand = integrand + x * dt
            else:
                temp = integrand + x * dt
                if temp < saturation and temp > -saturation:
                    integrand = temp

    def integrator_variable(self, dt, integrand=0):
        """Iterative numerical integrator.

        Provide the sample time (s), and use the .send(value) method
        to integrate.

        For example:

        .. code-block:: python

            intg_1 = Calculus().integrator_variable(0.01)
            while True:
                value = some_random_function()
                time_step = some_time_taken
                value_integral = intg_1.send((value, time_step)))

        Multiple integrators can be defined for different signals.
        Do not use the same handle to integrate different value signals.
        """
        while True:
            x, dt = yield integrand
            integrand = integrand + x * dt

class Filter:
    """Class object consisting of different filter functions"""

    def low_pass_first_order(self, wn, dt, x0=0):
        """Standard first order low pass filter.

        Provide the filter frequency (rad/s), sample time (s), and
        use the .send(value) method to filter.

        For example:

        .. code-block:: python

            myFilter = filter().low_pass_first_order(20, 0.01)
            valueFiltered = next(myFilter)
            while True:
                value = some_random_function()
                valueFiltered = myFilter.send(value)

        Multiple filters can be defined for different signals.
        Do not use the same handle to filter different signals.
        """
        output = 0
        myIntegrator = Calculus().integrator(dt, integrand=x0)
        next(myIntegrator)
        while True:
            x = yield output
            output = myIntegrator.send(wn * (x - output))

    def low_pass_first_order_variable(self, wn, dt, x0=0):
        """Standard first order low pass filter.

        Provide the filter frequency (rad/s), sample time (s), and
        use the .send(value) method to filter.

        For example:

        .. code-block:: python

            myFilter = filter().low_pass_first_order(20, 0.01)
            valueFiltered = next(myFilter)
            while True:
                value = some_random_function()
                valueFiltered = myFilter.send(value)

        Multiple filters can be defined for different signals.
        Do not use the same handle to filter different signals.
        """
        output = x0
        myIntegrator = Calculus().integrator_variable(dt, integrand=x0)
        next(myIntegrator)
        while True:
            x, dt = yield output
            output = myIntegrator.send((wn * (x - output), dt))

    def low_pass_second_order(self, wn, dt, zeta=1, x0=0):
        """Standard second order low pass filter.

        Provide the filter frequency (rad/s), sample time (s), and
        use the .send(value) method to filter.

        For example:

        .. code-block:: python

            myFilter = filter().low_pass_second_order(20, 0.01)
            valueFiltered = next(myFilter)
            while True:
                value = some_random_function()
                valueFiltered = myFilter.send(value)

        Multiple filters can be defined for different signals.
        Do not use the same handle to filter different signals.
        """
        output = x0
        temp = 0
        myFIrstIntegrator = Calculus().integrator(dt, integrand=0)
        mySecondIntegrator = Calculus().integrator(dt, integrand=x0)
        next(myFIrstIntegrator)
        next(mySecondIntegrator)
        while True:
            x = yield output
            temp = myFIrstIntegrator.send(wn * ( x - output - 2*zeta*temp ) )
            output = mySecondIntegrator.send(wn * temp)

    def complimentary_filter(self, kp, ki, dt, x0=0):
        """Complementary filter based rate and correction signal
            using a combination of low and high pass on them respectively.
        """
        output = 0
        temp = 0
        integratorRate = Calculus().integrator(dt, integrand=0)
        integratorTuner = Calculus().integrator(dt, integrand=x0)
        next(integratorRate)
        next(integratorTuner)
        while True:
            rate, correction = yield output
            temp = integratorRate.send(rate)
            error = output - correction
            output = temp - (kp*error) - (ki*integratorTuner.send(error))

    def moving_average(self, samples, x0=0):
        """Standard moving average filter.

        Provide the number of samples to average, and use the .send(value)
        method to filter.

        For example:

        .. code-block:: python

            myFilter = filter().moving_average(20)
            valueFiltered = next(myFilter)
            while True:
                value = some_random_function()
                valueFiltered = myFilter.send(value)

        Multiple filters can be defined for different signals.
        Do not use the same handle to filter different signals.
        """
        window = x0*np.ones(samples)
        average = x0
        while True:
            newValue = yield average
            window = np.append(newValue, window[0:samples-1])
            average = window.mean()

        










from pal.utilities.stream import BasicStream
from threading import Thread
try:
    from quanser.common import Timeout
except:
    from quanser.communications import Timeout
import cv2
import numpy as np
import sys
from pal.utilities.scope import Scope, MultiScope, XYScope

class Probe():
    '''Class object to send data to a remote Observer.
    Includes support for Displays (for video data), Plots (standard polar
    plot as an image) and Scope (standard time series plotter).'''

    def __init__(self,
                 ip = 'localhost'):

        self.remoteHostIP = ip
        self.agents = dict()
        self.agentType = []
        # agentType =>  0 Video Display
        #               1 Polar Plot
        #               2 Scope
        self.numDisplays = 0
        self.numPlots = 0
        self.numScopes = 0
        self.connected = False

    def add_display(self,
            imageSize = [480,640,3],
            scaling = True,
            scalingFactor = 2,
            name = 'display'
        ):

        self.numDisplays += 1
        _display = RemoteDisplay(ip = self.remoteHostIP,
            id = self.numDisplays,
            imageSize = imageSize,
            scaling = scaling,
            scalingFactor = scalingFactor
            )

        if name == 'display':
            name = 'display_'+str(self.numDisplays)
        # agent type => 0
        self.agents[name] = (_display, 0)

        return True

    def add_plot(self,
            numMeasurements = 1680,
            scaling = True,
            scalingFactor = 2,
            name = 'plot'
        ):
        self.numPlots += 1
        _plot = RemotePlot(ip = self.remoteHostIP,
                           numMeasurements=numMeasurements,
                           id=self.numPlots,
                           scaling = scaling,
                           scalingFactor = scalingFactor)

        if name == 'plot':
            name = 'plot_'+str(self.numDisplays)
        # agent type => 1
        self.agents[name] = (_plot, 1)

        return True

    def add_scope(self,
            numSignals = 1,
            name = 'scope'
        ):
        self.numScopes += 1

        _scope = RemoteScope(numSignals=numSignals, id=self.numScopes, ip=self.remoteHostIP)

        if name == 'scope':
            name = 'scope_'+str(self.numDisplays)
        # agent type => 2
        self.agents[name] = (_scope, 2)

        return True

    def check_connection(self):
        '''Attempts to connect every unconnected probe in the agentList.
        Returns True if every probe is successfully connected.'''
        self.connected = True

        for key in self.agents:
            if not self.agents[key][0].connected:
                self.agents[key][0].check_connection()
            self.connected = self.connected and self.agents[key][0].connected
        return self.connected

    def send(self, name,
             imageData=None,
             lidarData=None,
             scopeData=None):
        '''Ensure that at least one of imageData, lidarData, scopeData is provided,
        and that the type of data provided matches the expected name, otherwise an
        error message is printed and the method returns False.

        imageData => numpy array conforming to imageSize used in display definition \n
        lidarData => (ranges, angles) tuple, where ranges and angles are numpy arrays conforming to numMeasurements used in plot definition \n
        scopeData => (time, data) tuple, with data being a numpy array conforming to numSignals used in scope definition and time is the timestamp \n
        '''

        flag = False
        agentType = self.agents[name][1]
        if agentType == 0:
            if imageData is None:
                print("Image data not provided for a display agent.")
            else:
                flag = self.agents[name][0].send(imageData)
        elif agentType == 1:
            if lidarData is None:
                print("Lidar data not provided for a plot agent.")
            else:
                flag = self.agents[name][0].send(distances=lidarData[0], angles=lidarData[1])
        elif agentType == 2:
            if scopeData is None:
                print("Scope data not provided for a scope agent")
            else:
                flag = self.agents[name][0].send(scopeData[0], data=scopeData[1])

        return flag

    def terminate(self):
        for key in self.agents:
            self.agents[key][0].terminate()

class ObserverAgent():
    def __init__(self, uriAddress, id, bufferSize, buffer, agentType, properties):

        self.server = BasicStream(uriAddress, agent='S',
                                  recvBufferSize=bufferSize,
                                  receiveBuffer=buffer,
                                  nonBlocking=False)
        self.id = id
        self.buffer = buffer
        self.bufferSize = bufferSize
        self.agentType = agentType
        # agentType =>  0 Video Display
        #               1 Polar Plot
        #               2 Scope
        self.properties = properties
        self.connected = self.server.connected
        self.timeout = Timeout(seconds=0, nanoseconds=1000000)
        self.counter = 0

        self.name = properties['name']
        if self.agentType == 0:
            self.imageSize = self.properties['imageSize']
            self.scalingFactor = self.properties['scalingFactor']
            # return True
        elif self.agentType == 1:
            self.numMeasurements = self.properties['numMeasurements']
            self.frameSize = self.properties['frameSize']
            self.pixelsPerMeter = self.properties['pixelsPerMeter']
            self.distances = np.zeros((self.numMeasurements,1), dtype=np.float32)
            self.angles = np.zeros((self.numMeasurements,1), dtype=np.float32)
            # return True
        elif self.agentType == 2:
            self.numSignals=self.properties['numSignals']
            self.signalNames=self.properties['signalNames']
            self.timeWindow = 10
            self.refreshWindow = self.timeWindow
            self.timer_bias = 0
            self.scope = Scope(
                title=self.name,
                timeWindow=self.timeWindow,
                xLabel='Time (s)',
                yLabel='Data'
            )
            if self.signalNames is None:
                for i in range(self.numSignals):
                    self.scope.attachSignal(name='signal_'+str(i+1))
            else:
                for i in range(self.numSignals):
                    self.scope.attachSignal(name=self.signalNames[i])
            # return True
        # else:
            # return False

    def receive(self):
        recvFlag = False
        exitCondition = False
        if self.server.connected:
            recvFlag, bytesReceived = self.server.receive(
                iterations=2, timeout=self.timeout)
            # add some sort of compression
            if recvFlag:
                self.counter = 1
            if not recvFlag:
                self.counter += 1
            if self.counter >= 1000:
                exitCondition = True
        return recvFlag, exitCondition

    def process(self):
        if self.agentType == 0:
            cv2.imshow(self.name, self.server.receiveBuffer)
            return True
        elif self.agentType == 1:
            self.distances = self.server.receiveBuffer[:,0]
            self.angles = self.server.receiveBuffer[:,1]
            image_lidar = np.zeros((self.frameSize, self.frameSize), dtype=np.uint8)
            offset = np.int16(self.frameSize/2)
            x = np.int16(np.clip(offset - self.pixelsPerMeter * self.distances * np.cos(self.angles), 0, self.frameSize-1))
            y = np.int16(np.clip(offset - self.pixelsPerMeter * self.distances * np.sin(self.angles), 0, self.frameSize-1))
            image_lidar[x, y] = np.uint8(255)
            cv2.imshow(self.name, image_lidar)
            return True
        elif self.agentType == 2:
            self.data = self.server.receiveBuffer[1:]
            self.scope.sample(self.server.receiveBuffer[0], list(self.data))
            return True
        else:
            return False

    def check_connection(self):
        '''Checks if the sendCamera object is connected to its server. returns True or False.'''

        # First check if the server was connected.
        self.server.checkConnection(timeout=self.timeout)
        self.connected = self.server.connected

    def terminate(self):
        '''Terminates the connection.'''
        self.server.terminate()

class Observer():
    '''Class object to send data to a remote Observer.
    Includes support for Displays (for video data), Plots (standard polar
    plot as an image) and Scope (standard time series plotter).'''

    def __init__(self):

        self.agentList = []
        self.numDisplays = 0
        self.numPlots = 0
        self.numScopes = 0
        self.agentThreads = []

    def add_display(self, imageSize = [480,640,3], scalingFactor = 2, name = None):

        if scalingFactor < 1:
            scalingFactor == 1
        scaling = scalingFactor > 1

        if (scaling and
            (imageSize[0] % scalingFactor != 0 or
            imageSize[1] % scalingFactor != 0)):
            sys.exit('Select a scaling factor that is a factor of both width and height of image')

        if scaling:
            imageSize[0] = int(imageSize[0]/scalingFactor)
            imageSize[1] = int(imageSize[1]/scalingFactor)

        self.counter = 0
        bufferSize = np.prod(imageSize)
        self.numDisplays += 1
        port = 18800+self.numDisplays
        uriAddress  = 'tcpip://localhost:' + str(port)

        if name == None:
            name = 'Display '+str(self.numDisplays)

        properties = dict()
        properties['imageSize'] = [imageSize[0],imageSize[1],imageSize[2]]
        properties['scalingFactor'] = scalingFactor
        properties['name'] = name

        display = ObserverAgent(uriAddress=uriAddress,
                                id=self.numDisplays,
                                bufferSize=bufferSize,
                                buffer=np.zeros((imageSize[0], imageSize[1], imageSize[2]), dtype=np.uint8),
                                agentType=0,
                                properties=properties)

        self.agentList.append(display)

        return True

    def add_plot(self, numMeasurements, frameSize = 400, pixelsPerMeter = 40, scalingFactor = 4, name = None):

        if scalingFactor < 1:
            scalingFactor == 1
        scaling = scalingFactor > 1

        self.numPlots += 1
        port = 18600+self.numPlots
        uriAddress  = 'tcpip://localhost:'+str(port)
        if name == None:
            name = 'Plot '+str(self.numPlots)

        properties = dict()
        properties['frameSize'] = frameSize
        properties['pixelsPerMeter'] = pixelsPerMeter
        properties['numMeasurements'] = int(numMeasurements/scalingFactor)
        properties['name'] = name

        plot = ObserverAgent(uriAddress=uriAddress,
                                id=self.numPlots,
                                bufferSize=int(numMeasurements * 2 * 4 / scalingFactor),
                                buffer=np.zeros((int(numMeasurements / scalingFactor),2), dtype=np.float32),
                                agentType=1,
                                properties=properties)

        self.agentList.append(plot)

        return True

    def add_scope(self, numSignals = 1, name = None, signalNames=None):

        self.numScopes += 1

        if name == None:
            name = 'Scope '+str(self.numScopes)

        properties = dict()
        properties['numSignals'] = numSignals
        properties['name'] = name
        properties['signalNames'] = signalNames

        port = 18700+self.numScopes
        uriAddress  = 'tcpip://localhost:'+str(port)
        scope = ObserverAgent(uriAddress=uriAddress,
                                id=self.numPlots,
                                bufferSize=(numSignals + 1) * 8,
                                buffer=np.zeros((numSignals + 1, 1), dtype=np.float64),
                                agentType=2,
                                properties=properties)

        self.agentList.append(scope)

        return True

    def thread_function(self, index):
        agent = self.agentList[index]
        print('Launching thread ', index)
        while True:
            if not agent.connected:
                agent.check_connection()

            if agent.connected:
                flag, exit = agent.receive()
                if exit:
                    break
                if flag:
                    agent.process()
                    if agent.agentType == 0 or agent.agentType == 1:
                        cv2.waitKey(1)

    def launch(self):
        refreshFlag = False

        for index, agent in enumerate(self.agentList):
            if agent.agentType == 2:
                refreshFlag = True
                scopeIdx = index
            self.agentThreads.append(Thread(target=self.thread_function, args=[index]))
            self.agentThreads[-1].start()

        if refreshFlag:
            while self.agentThreads[scopeIdx].is_alive():
                MultiScope.refreshAll()

    def terminate(self):
        for probe in self.agentList:
            probe.terminate()

class RemoteDisplay: # works as a client
    '''Class object to send camera feed to a device. Works as a client.'''
    def __init__(
            self,
            ip = 'localhost',
            id = 0,
            imageSize = [480,640,3],
            scaling = True,
            scalingFactor = 2
        ):

        '''
        ip - IP address of the device receiving the data as a string, eg. '192.168.2.4' \n
        id - unique identifier for sending and receiving data. Needs to match the ID in the receiveCamera object. \n
        imageSize - list defining image size. eg. [height, width, # of channels] \n
        scaling - Boolean describing if image will be scaled when sending data. This will decrease its size by the factor in scalingFactor.Default is True \n
        scalingFactor - Factor by which the image will be decreased by. Default is 2. If original image is 480x640, sent image will be 240x320\n
         \n
        Consideration: values for id, imageSize, scaling and scalingFactor need to be the same on the SendCamera and ReceiveCamera objects.
        '''
        if scaling and scalingFactor < 1:
            scalingFactor == 1

        if (scaling and
            (imageSize[0] % scalingFactor != 0 or
            imageSize[1] % scalingFactor != 0)):
            sys.exit('Select a scaling factor that is a factor of both width and height of image')

        if scaling:
            imageSize[0] = int(imageSize[0]/scalingFactor)
            imageSize[1] = int(imageSize[1]/scalingFactor)
            self.newSize = (imageSize[1], imageSize[0])

        bufferSize = np.prod(imageSize)
        # if scaling:
        #     print('Remote Display will stream an image of dimension', self.newSize, 'for', bufferSize, 'bytes.')
        # else:
        #     print('Remote Display will stream an image of dimension', imageSize, 'for', bufferSize, 'bytes.')

        id = np.clip(id,0,80)
        port = 18800+id
        uriAddress  = 'tcpip://' + str(ip) + ':' + str(port)

        self.client = BasicStream(uriAddress, agent='C',
                                  sendBufferSize=bufferSize,
                                  nonBlocking=True)
        self.scaling = scaling
        self.scalingFactor = scalingFactor
        self.connected = self.client.connected

        self.timeout = Timeout(seconds=0, nanoseconds=10000000)

    def check_connection(self):
        '''Checks if the sendCamera object is connected to its server. returns True or False.'''
        # First check if the server was connected.
        self.client.checkConnection(timeout=self.timeout)
        self.connected = self.client.connected

    def send(self,image):
        '''Resizes the image if needed and sends the image added as the input.
         \n
        image - image to send, needs to match imageSize defined while initializing the object'''
        if self.client.connected:
            # add some sort of compression
            if self.scaling:
                image2 = cv2.resize(image, self.newSize)

                sent = self.client.send(image2)
            else:
                sent = self.client.send(image)
            if sent == -1:
                return False
            else:
                return True

    def terminate(self):
        '''Terminates the connection.'''
        self.client.terminate()

class RemotePlot: # works as a client

    def __init__(
            self,
            ip = 'localhost',
            id = 1,
            numMeasurements = 1680,
            scaling = True,
            scalingFactor = 4
        ):
        self.scalingFactor = scalingFactor
        self.scaling = scaling
        self.numMeasurements = numMeasurements
        if scaling:
            self.numMeasurements = int(numMeasurements/self.scalingFactor)
        bufferSize = self.numMeasurements * 2 * 4 # 4 bytes per float and 2 for distances + angles
        port = 18600+id
        uriAddress  = 'tcpip://' + ip + ':'+ str(port)
        self.client = BasicStream(uriAddress, agent='C',
                                  sendBufferSize=bufferSize,
                                  nonBlocking=True)
        self.connected = self.client.connected
        self.timeout = Timeout(seconds=0, nanoseconds=1)

    def check_connection(self):
        '''Checks if the sendCamera object is connected to its server. returns True or False.'''
        # First check if the server was connected.
        self.client.checkConnection(timeout=self.timeout)
        self.connected = self.client.connected

    def send(self, distances = None, angles = None):
        if distances is None or angles is None:
            return False

        if self.client.connected:
            if self.scaling:
                data = np.concatenate((np.reshape(distances[0:-1:self.scalingFactor], (-1, 1)), np.reshape(angles[0:-1:self.scalingFactor], (-1, 1))), axis=1)
            else:
                data = np.concatenate((np.reshape(distances, (-1, 1)), np.reshape(angles, (-1, 1))), axis=1)
            result = self.client.send(data)
            if result == -1:
                return False
            else:
                return True

    def terminate(self):
        '''Terminates the connection.'''
        self.client.terminate()

class RemoteScope():
    def __init__(
            self,
            numSignals = 1,
            id = 1,
            ip = 'localhost'
        ):

        self.numMeasurements = numSignals
        bufferSize = (self.numMeasurements+1) * 8 # 8 bytes per double
        port = 18700+id
        uriAddress  = 'tcpip://' + ip + ':'+ str(port)
        self.client = BasicStream(uriAddress, agent='C',
                                  sendBufferSize=bufferSize,
                                  nonBlocking=True)
        self.connected = self.client.connected
        self.timeout = Timeout(seconds=0, nanoseconds=1000000)

    def check_connection(self):
        '''Checks if the sendCamera object is connected to its server. returns True or False.'''
        # First check if the server was connected.
        self.client.checkConnection(timeout=self.timeout)
        self.connected = self.client.connected

    def send(self, time, data = None):
        if data is None:
            return False

        if self.client.connected:
            timestamp = np.array([time], dtype=np.float64)
            flag = self.client.send(np.concatenate((timestamp, data)))
            if flag == -1:
                return False
            else:
                return True

    def terminate(self):
        '''Terminates the connection.'''
        self.client.terminate()








"""
scope.py: A module providing classes for real-time plotting and visualization.

This module contains a collection of classes designed to simplify the process
of visualizing real-time data such as signals, images, and video streams.
"""
import sys
import numpy as np
from array import array
from collections import deque
from time import time
import pyqtgraph as pg

from pyqtgraph.Qt import QtWidgets

if 'PyQt6' in sys.modules:
    from PyQt6.QtCore import Qt
elif 'PyQt5' in sys.modules:
    from PyQt5.QtCore import Qt
else:
    raise ImportError("Scope requires either PyQt5 or PyQt6")


#region : Descriptor Classes
class _ScopeInfo():
    def __init__(self, **kwargs):
        self.title = kwargs.pop('title', None)
        self.rows = kwargs.pop('rows', 1)
        self.cols = kwargs.pop('cols', 1)

        self.maxSampleRate = kwargs.pop('maxSampleRate', 1024) # samples/second
        self.fps = kwargs.pop('fps', 30)

        self.axes = []

class _AxisInfo():
    def __init__(self, **kwargs):
        self._title = kwargs.pop('title', None)
        if self._title is None:
            self._title = kwargs.pop('_title', None)

        self._xLabel = kwargs.pop('xLabel', None)
        if self._xLabel is None:
            self._xLabel = kwargs.pop('_xLabel', None)

        self._yLabel = kwargs.pop('yLabel', None)
        if self._yLabel is None:
            self._yLabel = kwargs.pop('_yLabel', None)

        self._yLim = kwargs.pop('yLim', None)
        if self._yLim is None:
            self._yLim = kwargs.pop('_yLim', None)

        self.row = kwargs.pop('row', 0)
        self.col = kwargs.pop('col', 0)
        self.rowSpan = kwargs.pop('rowSpan', 1)
        self.colSpan = kwargs.pop('colSpan', 1)

        self.signals = []

class _XYAxisInfo(_AxisInfo):
    def __init__(self, **kwargs):
        self._xLim = kwargs.pop('xLim', None)
        if self._xLim is None:
            self._xLim = kwargs.pop('_xLim', None)

        super().__init__(**kwargs)

        self.images = []

class _TYAxisInfo(_AxisInfo):
    def __init__(self, **kwargs):
        self.timeWindow = kwargs.pop('timeWindow', 10)
        self.displayMode = kwargs.pop('displayMode', 0)
        super().__init__(**kwargs)

class _SignalInfo():
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name', None)
        self.color = kwargs.pop('color', None)
        self.lineStyle = kwargs.pop('lineStyle', None)
        self.width = kwargs.pop('width', None)
        self.scale = kwargs.pop('scale', 1)
        self.offset = kwargs.pop('offset', 0)

class _ImageInfo(): #XXX
    def __init__(self, **kwargs):
        self._scale = kwargs.pop('scale', (1,1))
        if self._scale is None:
            self._scale = kwargs.pop('_scale', None)

        self._offset = kwargs.pop('offset', (0,0))
        if self._offset is None:
            self._offset = kwargs.pop('_offset', None)

        self._rotation = kwargs.pop('rotation', 0)
        if self._rotation is None:
            self._rotation = kwargs.pop('_rotation', None)

        self._levels = kwargs.pop('levels', (0, 1))
        if self._levels is None:
            self._levels = kwargs.pop('_levels', None)
#endregion

#region : Primary Scope Classes
class MultiScope(_ScopeInfo):
    """MultiScope: A class for creating and a scope with multiple axes.

    This class inherits from _ScopeInfo and allows for the creation and
    management of a scope with multiple axes in a grid layout, supporting both
    time-y and x-y axes.

    Attributes:
        activeScopes (list): A list of active MultiScope instances.
        spf (float): Time to wait between frames.
        tpf (float): Time of the last frame.
    """

    activeScopes = []
    spf = None # time to wait between frames
    tpf = 0 # time of last frame

    def __init__(self, *args, **kwargs):
        """Initialize a MultiScope instance.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        graphicsLayoutWidget = kwargs.pop('graphicsLayoutWidget', None)

        if args and isinstance(args[0], _ScopeInfo):
            super().__init__(**vars(args[0]))
            axes = args[0].axes
        else:
            super().__init__(**kwargs)
            axes = []

        MultiScope.activeScopes.append(self)
        if MultiScope.spf is None:
            MultiScope.spf = 1 / self.fps
        else:
            MultiScope.spf = np.minimum(
                MultiScope.spf,
                1 / self.fps
            )

        # = Create buffers for short term storage of samples prior to plotting
        self._bufferSize = int(np.ceil(2*self.maxSampleRate/self.fps))

        # = Create window object
        if graphicsLayoutWidget is None:
            self.graphicsLayoutWidget = pg.GraphicsLayoutWidget(
                show=True,
                title=self.title
            )
        else:
            self.graphicsLayoutWidget = graphicsLayoutWidget

        # = Create empty grid of cells to be filled with axes later
        self.cells = []
        for i in range(self.rows):
            self.cells.append([None]*self.cols)

        for axis in axes:
            self._addAxis(axis)

    def _cells_available(self,row, col, rowSpan, colSpan):
        # = Check if requested cells outside of grid
        if (row < 0
            or col < 0
            or row+rowSpan > self.rows
            or col+colSpan > self.cols
            ):
            return False

        # = Check if any cells inside the grid are already occupied
        for i in range(rowSpan):
            for j in range(colSpan):
                if self.cells[row+i][col+j] is not None:
                    return False

        return True

    def _populate_cells(self, axis):
        # = Assumes cells_available has already been checked
        for i in range(axis.rowSpan):
            for j in range(axis.colSpan):
                self.cells[axis.row+i][axis.col+j] = axis

    def _addAxis(self, axisInfo):
        if isinstance(axisInfo, _TYAxisInfo):
            AxisType = TYAxis
        elif isinstance(axisInfo, _XYAxisInfo):
            AxisType = XYAxis
        else:
            raise TypeError('Invalid Axis type provided')

        if not self._cells_available(
                axisInfo.row,
                axisInfo.col,
                axisInfo.rowSpan,
                axisInfo.colSpan
            ):
            raise Exception("Scope cell range invalid or already occupied ")

        # = Create a new axis object and add it to the list of axes
        axis = AxisType(
            graphicsLayoutWidget=self.graphicsLayoutWidget,
            bufferSize=self._bufferSize,
            **vars(axisInfo)
        )

        for sig in axisInfo.signals:
            axis.attachSignal(sig)

        if isinstance(axisInfo, _XYAxisInfo):
            for img in axisInfo.images:
                axis.attachImage(img)

        self.axes.append(axis)


    # Really addYtAxis, but addAxis reads nicer...
    def addAxis(self, *args, **kwargs):
        """Add a time-y axis to the MultiScope instance.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        if args:
            if isinstance(args[0], _TYAxisInfo):
                self._addAxis(args[0])
            else:
                raise TypeError(
                    'Must use keyword arguments, unless '
                    + 'providing a _TYAxisInfo object.'
                )
        else:
            self._addAxis(_TYAxisInfo(**kwargs))

    def addXYAxis(self, *args, **kwargs):
        """Add an x-y axis to the MultiScope instance.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        if args:
            if isinstance(args[0], _XYAxisInfo):
                self._addAxis(args[0])
            else:
                raise TypeError(
                    'Must use keyword arguments, unless '
                    + 'providing a _XYAxisInfo object.'
                )
        else:
            self._addAxis(_XYAxisInfo(**kwargs))


    def refresh(self):
        """Refresh the MultiScope instance, updating contained axes.
        """
        MultiScope.refreshAll()

    def refreshAll():
        """Refresh all active MultiScope instances, updating all scope plots.
        """
        if time()-MultiScope.tpf >= MultiScope.spf:
            MultiScope.tpf = time()
            flush = True
        else:
            flush = False

        for scope in MultiScope.activeScopes:
            for axis in scope.axes:
                axis.refresh(flush)

        if flush:
            QtWidgets.QApplication.processEvents()


class Axis(_AxisInfo):
    """Axis: A base class for creating and managing scope plot axes

    This class inherits from _AxisInfo and is used to create and manage
    scope plot axes. It is not meant to be used directly but serves as
    the base class for specialized axis classes (e.g., TYAxis, XYAxis).

    Attributes:
        xLabel (str): The label for the x-axis.
        yLabel (str): The label for the y-axis.
        xLim (tuple): The limits of the x-axis in the form (min, max).
        yLim (tuple): The limits of the y-axis in the form (min, max).
    """

    def __init__(self, graphicsLayoutWidget, bufferSize, **kwargs):
        """Initialize an Axis instance.

        Args:
            graphicsLayoutWidget (GraphicsLayoutWidget): The parent widget.
            bufferSize (int): Buffer size for short term storage of samples.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        # = Create and Configure Plot
        self.plot = graphicsLayoutWidget.addPlot(
            self.row,
            self.col,
            self.rowSpan,
            self.colSpan,
            labels={
                'bottom': (self._xLabel,),
                'left': (self._yLabel,)
            }
        )
        self.plot.showGrid(x=True, y=True)
        self.plot.addLegend()
        self.yLim = self._yLim

        # = Create buffers for short term storage of samples prior to plotting
        self._bufferSize = bufferSize
        self._tBuffer = array('f', range(bufferSize))
        self._dataBuffer = []
        self._iBuffer = -1
        self._sampleQueue = deque()

        self.refresh = self._initial_refresh

    def attachSignal(self, *args, **kwargs):
        """Attach a signal to the Axis instance.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        di = len(self.signals)
        s = Signal(self.plot, di=di, *args, **kwargs)
        self.signals.append(s)

    def sample(self, t, data):
        """Sample data for the Axis instance.

        Args:
            t (float): The time value.
            data (list): The data samples for each attached signal.
        """
        data = [data] if np.isscalar(data) else data
        self._sampleQueue.appendleft([t, data])

    def clean(self):
        self._sampleQueue.clear

    def clear(self):
        """Clear the signals in the Axis instance."""
        for s in self.signals:
            s.clear()

    def _initial_refresh(self, flush):
        if len(self._sampleQueue) < 2:
            return
        self.refresh = self._post_initial_refresh
        self.refresh(flush)

    def _post_initial_refresh(self, flush):
        pass

    @property
    def xLabel(self):
        """str: The label for the x-axis of the Axis instance."""
        return self._xLabel
    @xLabel.setter
    def xLabel(self, newXLabel):
        self._xLabel = newXLabel
        self.plot.setLabel('bottom', newXLabel)

    @property
    def yLabel(self):
        """str: The label for the y-axis of the Axis instance."""
        return self._yLabel
    @yLabel.setter
    def yLabel(self, newYLabel):
        self._yLabel = newYLabel
        self.plot.setLabel('left', newYLabel)

    @property
    def xLim(self):
        """tuple: The limits of the x-axis in the form (min, max).
            Set to None to enable auto-ranging.
        """
        return self._xLim
    @xLim.setter
    def xLim(self, newXLimits):
        if newXLimits is None:
            self.plot.enableAutoRange(axis='x')
            return
        self._xLim = newXLimits
        self.plot.setXRange(newXLimits[0],newXLimits[1])

    @property
    def yLim(self):
        """tuple: The limits of the y-axis in the form (min, max).
            Set to None to enable auto-ranging.
        """
        return self._yLim
    @yLim.setter
    def yLim(self, newYLimits):
        if newYLimits is None:
            self.plot.enableAutoRange(axis='y')
            return
        self._yLim = newYLimits
        self.plot.setYRange(newYLimits[0],newYLimits[1])


class XYAxis(Axis, _XYAxisInfo):
    """XYAxis: A class for creating and managing an x-y axis in a MultiScope.

    This class inherits from Axis and _XYAxisInfo and is used to create
    and manage an x-y plot axis in a MultiScope.
    """

    def __init__(self, **kwargs):
        """Initialize an XYAxis instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.xLim = self._xLim

    def attachSignal(self, *args, **kwargs):
        """Attach a 2D signal to the axis.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        # - x data from 0:self._bufferSize+1
        # - y data from self._bufferSize:self._bufferSize*2+1
        self._dataBuffer.append(array('f', range(2*self._bufferSize)))
        super().attachSignal(*args, **kwargs)

    def attachImage(self, *args, **kwargs):
        """Attach an image to the axis.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.plot.showGrid(x=False, y=False)
        img = Image(self.plot, *args, **kwargs)
        self.images.append(img)

    def _post_initial_refresh(self, flush):
        # = empty out sample queue
        while True: # XXX break if max iterations passed
            try:
                if self._iBuffer >= self._bufferSize-1:
                    flush = True
                    break
                sample = self._sampleQueue.pop()
                self._iBuffer += 1
                self._tBuffer[self._iBuffer] = sample[0]
                for i in range(len(self.signals)):
                    self._dataBuffer[i][self._iBuffer] = sample[1][i][0]
                    self._dataBuffer[i][self._iBuffer+self._bufferSize] = \
                        sample[1][i][1]
            except IndexError:
                break

        if flush and self._iBuffer > 0:
            # = Plot chunk for using new data points
            for i, signal in enumerate(self.signals):
                signal.add_chunk(
                    self._dataBuffer[i][0:self._iBuffer+1],
                    self._dataBuffer[i]\
                        [self._bufferSize:self._bufferSize+self._iBuffer+1]
                )
                self._dataBuffer[i][0] = \
                    self._dataBuffer[i][self._iBuffer]
                self._dataBuffer[i][self._bufferSize] = \
                    self._dataBuffer[i][self._bufferSize+self._iBuffer]

            self._tBuffer[0] = self._tBuffer[self._iBuffer]
            self._iBuffer = 0


class TYAxis(Axis, _TYAxisInfo):
    """TYAxis: A class for creating and managing a time-y axis in a MultiScope.

    This class inherits from Axis and _TYAxisInfo and is used to create
    and manage a time-y plot axis in a MultiScope.
    """

    def __init__(self, **kwargs):
        """Initialize a TYAxis instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

    def attachSignal(self, *args, **kwargs):
        """Attach a signal to the TYAxis instance.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        self._dataBuffer.append(array('f', range(self._bufferSize)))
        super().attachSignal(*args, **kwargs)

    def _post_initial_refresh(self, flush):
        # = empty out sample queue
        while True: # XXX break if max iterations passed
            try:
                if self._iBuffer >= self._bufferSize-1:
                    flush = True
                    break
                sample = self._sampleQueue.pop()
                self._iBuffer += 1
                self._tBuffer[self._iBuffer] = sample[0]
                for i in range(len(self.signals)):
                    self._dataBuffer[i][self._iBuffer] = sample[1][i]
            except IndexError:
                break

        if flush and self._iBuffer > 0:
            # = Plot chunk for using new data points
            self.plot.setXRange(
                np.max([0, self._tBuffer[self._iBuffer]-self.timeWindow]),
                np.max([self._tBuffer[self._iBuffer], self.timeWindow]),
                padding=0
            )
            for i, signal in enumerate(self.signals):
                signal.add_chunk(
                    self._tBuffer[0:self._iBuffer+1],
                    self._dataBuffer[i][0:self._iBuffer+1]
                )
                self._dataBuffer[i][0] = \
                    self._dataBuffer[i][self._iBuffer]

            self._tBuffer[0] = self._tBuffer[self._iBuffer]
            self._iBuffer = 0


class Signal(_SignalInfo):
    maxChunks = 1024

    defaultStyles = [
        _SignalInfo(color=[196,78,82], width=1.5, lineStyle='-'),
        _SignalInfo(color=[85,85,85], width=1.5, lineStyle='-.'),
        _SignalInfo(color=[85,168,104], width=1.5, lineStyle=':'),
        _SignalInfo(color=[76,114,176], width=1.5, lineStyle='--'),
        _SignalInfo(color=[248,217,86], width=1.5, lineStyle='-.'),
        _SignalInfo(color=[129,114,179], width=1.5, lineStyle='-..'),
        _SignalInfo(color=[221,132,82], width=1.5, lineStyle='-'),
    ]

    def __init__(self, plot, *args, **kwargs):
        di = kwargs.pop('di', 0)

        if args and isinstance(args[0], _SignalInfo):
            super().__init__(**vars(args[0]))
        else:
            super().__init__(**kwargs)
        self.plot = plot
        self._pDIs = deque()

        # = Configure Cosmetic Properties
        if self.color is None:
            self.color = Signal.defaultStyles[di].color
        if self.width is None:
            self.width = Signal.defaultStyles[di].width
        if self.lineStyle is None:
            self.lineStyle = Signal.defaultStyles[di].lineStyle

        if 'PyQt6' in sys.modules:
            if self.lineStyle == ':':
                lineStyle = Qt.PenStyle.DotLine
            elif self.lineStyle == '--':
                lineStyle = Qt.PenStyle.DashLine
            elif self.lineStyle == '-.':
                lineStyle = Qt.PenStyle.DashDotLine
            elif self.lineStyle == '-..':
                lineStyle = Qt.PenStyle.DashDotDotLine
            else:
                lineStyle = Qt.PenStyle.SolidLine
        else:
            if self.lineStyle == ':':
                lineStyle = Qt.DotLine
            elif self.lineStyle == '--':
                lineStyle = Qt.DashLine
            elif self.lineStyle == '-.':
                lineStyle = Qt.DashDotLine
            elif self.lineStyle == '-..':
                lineStyle = Qt.DashDotDotLine
            else:
                lineStyle = Qt.SolidLine

        self.pen = pg.mkPen(
            color=pg.mkColor(self.color),
            width=self.width,
            style=lineStyle
        )

        self._pDI4Legend = pg.PlotDataItem(name=self.name, pen=self.pen)
        self.plot.addItem(self._pDI4Legend)

    def add_chunk(self, x, y):
        pdi = pg.PlotDataItem(
            x,
            y,
            skipFiniteCheck=True,
            connect='all',
            pen=self.pen
        )
        self.plot.addItem(pdi)
        self._pDIs.append(pdi)

        while len(self._pDIs) > Signal.maxChunks:
            pdi = self._pDIs.popleft()
            self.plot.removeItem(pdi)

    def clear(self):
        # print(len(self._pDIs))
        while len(self._pDIs) > 0:
            pdi = self._pDIs.popleft()
            self.plot.removeItem(pdi)


class Image(_ImageInfo):
    """Image: A class for containing and managing images to be plotted

    This class inherits from _ImageInfo and provides additional functionality
    for displaying images within a scope, including scaling, offset, and
    rotation.
    """

    def __init__(self, plot, *args, **kwargs):
        """
        Initialize an Image instance.

        Args:
            plot (pg.PlotItem): The plot to which the image will be added.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        if args and isinstance(args[0], _ImageInfo):
            super().__init__(**vars(args[0]))
        else:
            super().__init__(**kwargs)

        self.imageItem = pg.ImageItem(
            levels=self._levels,
            axisOrder='row-major'
        )
        plot.addItem(self.imageItem)

        self._tr = pg.QtGui.QTransform()
        self.scale = self._scale
        self.offset = self._offset
        self.rotation = self._rotation

    def setImage(self, image):
        """Set the image data for the Image instance.

        Args:
            image (np.ndarray): The image data to be displayed.
        """
        self.imageItem.setImage(
            image=image,
            levels=self._levels,
        )

    @property
    def scale(self):
        """Get the scale of the image.

        Returns:
            tuple: The scale factor of the image (x_scale, y_scale).
        """
        return self._scale
    @scale.setter
    def scale(self, newScale):
        """Set the scale of the image.

        Args:
            newScale (tuple): The new scale factor for the image
                (x_scale, y_scale).
        """
        self._scale = newScale
        self._tr.scale(*self._scale)
        self.imageItem.setTransform(self._tr)

    @property
    def offset(self):
        """Get the offset of the image.

        Returns:
            tuple: The offset of the image (x_offset, y_offset).
        """
        return self._offset
    @offset.setter
    def offset(self, newOffset):
        """Set the offset of the image.

        Args:
            newOffset (tuple): The new offset for the image
                (x_offset, y_offset).
        """
        self._offset = newOffset
        self._tr.translate(*self._offset)
        self.imageItem.setTransform(self._tr)

    @property
    def rotation(self):
        """Get the rotation of the image.

        Returns:
            float: The rotation angle of the image in degrees.
        """
        return self._rotation
    @rotation.setter
    def rotation(self, newRotation):
        """Set the rotation of the image.

        Args:
            newRotation (float): The new rotation angle for the image in
                degrees.
        """
        self._rotation = newRotation
        self._tr.rotate(self._rotation)
        self.imageItem.setTransform(self._tr)

    @property
    def levels(self):
        """Get the color levels of the image.

        Returns:
            tuple: The color levels of the image (min_level, max_level).
        """
        return self._levels
    @levels.setter
    def levels(self, newLevels):
        """Set the color levels of the image.

        Args:
            newLevels (tuple): The new color levels for the image
                (min_level, max_level).
        """
        self._levels = newLevels
        self.imageItem.setLevels(self._levels)

#endregion


class Scope():
    """
    Scope: A class for real-time plotting of 1D signals.

    This class provides an interface for real-time visualization of 1D signals,
    such as sensor readings, control signals, or any time-varying data.
    """

    def __init__(self, title=None, **kwargs):
        """Initialize a Scope instance.

        Args:
            title (str, optional): The title of the scope window.
            **kwargs: Additional keyword arguments for creating the axis.
        """
        self._ms = MultiScope(
            title=title,
            graphicsLayoutWidget=kwargs.pop('graphicsLayoutWidget', None),
        )
        self._ms.addAxis(**kwargs)

    def attachSignal(self, **kwargs):
        """Attach a signal to the Scope instance for plotting.

        Args:
            **kwargs: Additional keyword arguments for attaching the signal.
        """
        self._ms.axes[0].attachSignal(**kwargs)

    def sample(self, t, data):
        """Provide a new data sample for the attached signals.

        Args:
            t (float): The timestamp of the data sample.
            data (float or np.ndarray): The new data sample to be plotted.
        """
        self._ms.axes[0].sample(t, data)

    def clean(self):
        self._ms.axes[0].clean()

    def clear(self):
        self._ms.axes[0].clear()

    def refresh(self):
        """Refresh the Scope instance to update the plot with the latest data.
        """
        self._ms.refresh()

    def refreshAll():
        """Refresh all the active Scope instances.
        """
        MultiScope.refreshAll()

class XYScope():
    """XYScope: A class for real-time display of 2D signals and images.

    This class provides an interface for real-time visualization of 2D signals
    and images, updating them in real-time.

    Attributes:
        images (list): A list of attached images for the XYScope instance.
    """

    def __init__(self, title=None, **kwargs):
        """Initialize an XYScope instance.

        Args:
            title (str, optional): The title of the scope window.
            **kwargs: Additional keyword arguments for creating the axis.
        """
        self._ms = MultiScope(
            title=title,
            graphicsLayoutWidget=kwargs.pop('graphicsLayoutWidget', None),
        )
        self._ms.addXYAxis(**kwargs)

        self.images = []

    def attachSignal(self, **kwargs):
        """Attach a signal to the XYScope instance for plotting.

        Args:
            **kwargs: Additional keyword arguments for attaching the signal.
        """
        self._ms.axes[0].attachSignal(**kwargs)

    def attachImage(self, *args, **kwargs):
        """Attach an image to the XYScope instance for display.

        Args:
            *args: Additional arguments for attaching the image.
            **kwargs: Additional keyword arguments for attaching the image.
        """
        self._ms.axes[0].attachImage(*args, **kwargs)
        self.images.append(self._ms.axes[0].images[-1])

    def sample(self, t, data):
        """Provide a new data sample for the attached signals.

        Args:
            t (float): The timestamp of the data sample.
            data (list): The new data sample to be plotted.
        """
        self._ms.axes[0].sample(t, data)

    def clear(self):
        self._ms.axes[0].clear()

    def refresh(self):
        """Refresh the display of the XYScope to update the signals and images.
        """
        self._ms.refresh()

    def refreshAll():
        """Refresh all the active Scope instances.
        """
        MultiScope.refreshAll()
















        """
stream.py: A module providing utility classes for using Quanser's stream API.

This module contains a collection of classes designed to simplify the process
of using the low-level stream API for network communication. These classes
handle tasks such as connecting to remote systems, sending and receiving data
over a network connection, and managing the connection's state.
"""
from quanser.communications import Stream, StreamError, PollFlag
try:
    from quanser.common import Timeout
except:
    from quanser.communications import Timeout
import numpy as np
import pickle
import io

class BaseStream:
    """Base class for StreamServer and StreamClient."""
    sendBufferSize = 1024
    receiveBufferSize = 1024
    defaultTimeout = 100 * 1e-6 # 100 microseconds

    def __init__(self):
        self._stream = Stream()
        self._status = 0

        self.timeoutDuration = BaseStream.defaultTimeout

    @property
    def timeoutDuration(self):
        """Returns the timeout duration in seconds"""
        return self._timeout.seconds + self._timeout.nanoseconds * 1e-9

    @timeoutDuration.setter
    def timeoutDuration(self, timeout_duration_seconds):
        """Sets the timeout as an instance of Timeout.

        Args:
            timeout_duration_seconds (float): The desired timeout duration
                in seconds.
        """
        seconds = int(timeout_duration_seconds)
        nanoseconds = int((timeout_duration_seconds - seconds) * 1e9)
        self._timeout = Timeout(seconds=seconds, nanoseconds=nanoseconds)

    def terminate(self):
        """Terminate the stream by shutting down and closing the connection."""
        if self._status == 1:
            self._stream.shutdown()
            self._stream.close()
            self._status = 2

    def _check_poll_flags(self, flags):
        """Check if the specified are set for self._stream.

        Flag options come from quanser.communications.PollFlags
        Note: this function will fail if self._stream is already closed.

        Args:
            flags (PollFlag): The poll flags to check.

        Returns:
            bool: True if the specified flags are set, False otherwise.
        """
        pollResult = self._stream.poll(self._timeout, flags)
        return (pollResult & flags) == flags

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

class StreamServer(BaseStream):
    """Stream server class for accepting and managing client connections."""

    def __init__(self, uri, blocking=False):
        """Initialize the stream server.

        Args:
            uri (str): The URI to listen on.
            blocking (bool, optional): If True, the server operates in
                blocking mode. Defaults to False.
        """
        super().__init__()

        self._stream.listen(uri=uri, non_blocking=(not blocking))
        self._status = 1

    @property
    def status(self):
        """int: Get the status of the server (1: listening, 2: closed)."""
        return self._status

    def accept_clients(self):
        """Accept a client connection.

        Returns:
            StreamClient: A new StreamClient instance connected to the
                accepted client or None if no client was accepted.
        """
        if self._status < 2 and self._check_poll_flags(PollFlag.ACCEPT):
            client = self._stream.accept(
                BaseStream.sendBufferSize,
                BaseStream.receiveBufferSize
            )
            return StreamClient(client)
        return None

class StreamClient(BaseStream):
    """Stream client class for connecting to a stream server."""

    def __init__(self, uri, blocking=False):
        """Initialize the stream client.

        Args:
            uri (str): The URI to connect to.
            blocking (bool, optional): If True, the client operates in
                blocking mode. Defaults to False.
        """
        super().__init__()

        if isinstance(uri, Stream):
            # Not documented and not intended for end users. Used for
            # passing in a Stream object acquired by a server.
            self._status = 1
            self._stream = uri
        else:
            self._uri = uri
            self._blocking = blocking
            self._connect()

        self._buffer = bytearray(BaseStream.receiveBufferSize)


    def _connect(self):
            self._stream.connect(
                uri=self._uri,
                non_blocking=(not self._blocking),
                send_buffer_size=BaseStream.sendBufferSize,
                receive_buffer_size=BaseStream.receiveBufferSize
            )

    @property
    def status(self):
        """Returns the status of the client.

        Returns:
            0: waiting for connection
            1: connected
            2: closed
        """
        if self._status == 0:
            try:
                if self._check_poll_flags(PollFlag.CONNECT):
                    self._status = 1
            except StreamError as e:
                self._stream.close()
                self._connect()
        return self._status


    def receive(self):
        """Receive data from the stream.

        Returns:
            object: The received object or None if no data was received
                or the connection is closed.
        """
        if self.status == 1 and self._check_poll_flags(PollFlag.RECEIVE):
            bytesReceived = self._stream.receive(
                self._buffer,
                BaseStream.receiveBufferSize
            )
            if bytesReceived <= 0:
                self.terminate()
            elif bytesReceived > 0:
                header = self._buffer[0]
                payload = self._buffer[1:bytesReceived]

                if header == 1:
                    return payload.decode('utf-8')
                elif header == 2:
                    return np.load(io.BytesIO(payload))
                else:
                    return pickle.loads(payload)
        return None

    def send(self, obj):
        """Send data over the stream.

        Args:
            obj (object): The object to send.

        Returns:
            bool: True if the data was sent successfully, False otherwise.
        """
        if self.status == 1:
            try:
                if isinstance(obj, str):
                    byteArray = bytearray([1]) + obj.encode('utf-8')

                elif isinstance(obj, np.ndarray):
                    buffer = io.BytesIO()
                    np.save(buffer, obj)
                    byteArray = bytearray([2]) + buffer.getvalue()

                else:
                    byteArray = bytearray([0]) + pickle.dumps(obj)

                self._stream.send(
                    buffer=byteArray,
                    buffer_size=len(byteArray)
                )
                self._stream.flush()

                return True
            except StreamError as e:
                print(e.get_error_message()) # TODO: Check error code
                self.terminate()
        return False

class BasicStream:
    '''Class object consisting of basic stream server/client functionality'''
    def __init__(self, uri, agent='S', receiveBuffer=np.zeros(1, dtype=np.float64), sendBufferSize=2048, 
                 recvBufferSize=2048, nonBlocking=False, verbose=False, reshapeOrder = 'C'):
        '''
        This functions simplifies functionality of the quanser_stream module to provide a
        simple server or client. \n
         \n
        uri - IP server and port in one string, eg. 'tcpip://IP_ADDRESS:PORT' \n
        agent - 'S' or 'C' string representing server or client respectively \n
        receiveBuffer - numpy buffer that will be used to determine the shape and size of data received \n
        sendBufferSize - (optional) size of send buffer, default is 2048 \n
        recvBufferSize - (optional) size of recv buffer, default is 2048 \n
        nonBlocking - set to False for blocking, or True for non-blocking connections \n
         \n
        Stream Server as an example running at IP 192.168.2.4 which receives two doubles from the client: \n
        >>> myServer = BasicStream('tcpip://localhost:18925', 'S', receiveBuffer=np.zeros((2, 1), dtype=np.float64))
         \n
        Stream Client as an example running at IP 192.168.2.7 which receives a 480 x 640 color image from the server: \n
        >>> myClient = BasicStream('tcpip://192.168.2.4:18925', 'C', receiveBuffer=np.zeros((480, 640, 3), dtype=np.uint8))

        '''
        self.agent 			= agent
        self.sendBufferSize = sendBufferSize
        self.recvBufferSize = recvBufferSize
        self.uri 			= uri
        self.receiveBuffer  = receiveBuffer
        self.verbose        = verbose
        self.reshapeOrder   = reshapeOrder  
        # reshape order need to be specified as "F" when reading images streamed from a MATLAB server

        # If the agent is a Client, then Server isn't needed.
        # If the agent is a Server, a Client will also be needed. The server can start listening immediately.

        self.clientStream = Stream()
        if agent=='S':
            self.serverStream = Stream()

        # Set polling timeout to 10 milliseconds
        self.t_out = Timeout(seconds=0, nanoseconds=10000000)

        # connected flag initialized to False
        self.connected = False

        try:
            if agent == 'C':
                self.connected = self.clientStream.connect(uri, nonBlocking, self.sendBufferSize, self.recvBufferSize)
                if self.connected and self.verbose:
                    print('Connected to a Server successfully.')

            elif agent == 'S':
                if self.verbose:
                    print('Listening for incoming connections.')
                self.serverStream.listen(self.uri, nonBlocking)
            pass

        except StreamError as e:
            if self.agent == 'S' and self.verbose:
                print('Server initialization failed.')
            elif self.agent == 'C' and self.verbose:
                print('Client initialization failed.')
            print(e.get_error_message())

    def checkConnection(self, timeout=Timeout(seconds=0, nanoseconds=100)):
        '''When using non-blocking connections (nonBlocking set to True), the constructor method for this class does not block when
        listening (as a server) or connecting (as a client). In such cases, use the checkConnection method to attempt continuing to
        accept incoming connections (as a server) or connect to a server (as a client).  \n
         \n
        Stream Server as an example \n
        >>> while True:
        >>> 	if not myServer.connected:
        >>> 		myServer.checkConnection()
        >>>		if myServer.connected:
        >>> 		yourCodeGoesHere()
         \n
        Stream Client as an example \n
        >>> while True:
        >>> 	if not myClient.connected:
        >>> 		myClient.checkConnection()
        >>>		if myClient.connected:
        >>> 		yourCodeGoesHere()
         \n
        '''
        if self.agent == 'C' and not self.connected:
            try:
                pollResult = self.clientStream.poll(timeout, PollFlag.CONNECT)

                if (pollResult & PollFlag.CONNECT) == PollFlag.CONNECT:
                    self.connected = True
                    if self.verbose: print('Connected to a Server successfully.')

            except StreamError as e:
                if e.error_code == -33:
                    self.connected = self.clientStream.connect(self.uri, True, self.sendBufferSize, self.recvBufferSize)
                else:
                    if self.verbose: print('Client initialization failed.')
                    print(e.get_error_message())

        if self.agent == 'S' and not self.connected:
            try:
                pollResult = self.serverStream.poll(self.t_out, PollFlag.ACCEPT)
                if (pollResult & PollFlag.ACCEPT) == PollFlag.ACCEPT:
                    self.connected = True
                    if self.verbose: print('Found a Client successfully...')
                    self.clientStream = self.serverStream.accept(self.sendBufferSize, self.recvBufferSize)

            except StreamError as e:
                if self.verbose: print('Server initialization failed...')
                print(e.get_error_message())

    def terminate(self):
        '''Use this method to correctly shutdown and then close connections. This method automatically closes all streams involved (Server will shutdown server streams as well as client streams). \n
         \n
        Stream Server as an example \n
        >>> while True:
        >>> 	if not myServer.connected:
        >>> 		myServer.checkConnection()
        >>>		if myServer.connected:
        >>> 		yourCodeGoesHere()
        >>>			if breakCondition:
        >>>				break
        >>> myServer.terminate()
         \n
        Stream Client as an example	 \n
        >>> while True:
        >>> 	if not myClient.connected:
        >>> 		myClient.checkConnection()
        >>>		if myClient.connected:
        >>> 		yourCodeGoesHere()
        >>>			if breakCondition:
        >>>				break
        >>> myClient.terminate()

        '''

        if self.connected:
            self.clientStream.shutdown()
            self.clientStream.close()
            if self.verbose: print('Successfully terminated clients...')

        if self.agent == 'S':
            self.serverStream.shutdown()
            self.serverStream.close()
            if self.verbose: print('Successfully terminated servers...')

    def receive(self, iterations=1, timeout=Timeout(seconds=0, nanoseconds=10)):
        '''
        This functions populates the receiveBuffer with bytes if available. \n \n

        Accepts: \n
        iterations - (optional) number of times to poll for incoming data before terminating, default is 1 \n
         \n
        Returns: \n
        receiveFlag - flag indicating whether the number of bytes received matches the expectation. To check the actual number of bytes received, use the bytesReceived class object. \n
         \n
        Stream Server as an example \n
        >>> while True:
        >>> 	if not myServer.connected:
        >>> 		myServer.checkConnection()
        >>>		if myServer.connected:
        >>> 		flag = myServer.receive()
        >>>			if breakCondition or not flag:
        >>>				break
        >>> myServer.terminate()
         \n
        Stream Client as an example	 \n
        >>> while True:
        >>> 	if not myClient.connected:
        >>> 		myClient.checkConnection()
        >>>		if myClient.connected:
        >>> 		flag = myServer.receive()
        >>>			if breakCondition or not flag:
        >>>				break
        >>> myClient.terminate()

        '''

        self.t_out = timeout
        counter = 0
        dataShape = self.receiveBuffer.shape

        # Find number of bytes per array cell based on type
        numBytesBasedOnType = len(np.array([0], dtype=self.receiveBuffer.dtype).tobytes())

        # Calculate total dimensions
        dim = 1
        for i in range(len(dataShape)):
            dim = dim*dataShape[i]

        # Calculate total number of bytes needed and set up the bytearray to receive that
        totalNumBytes = dim*numBytesBasedOnType
        self.data = bytearray(totalNumBytes)
        self.bytesReceived = 0
        # print(totalNumBytes)
        # Poll to see if data is incoming, and if so, receive it. Poll a max of 'iteration' times
        try:
            while True:

                # See if data is available
                pollResult = self.clientStream.poll(self.t_out, PollFlag.RECEIVE)
                counter += 1
                if not (iterations == 'Inf'):
                    if counter > iterations:
                        break
                if not ((pollResult & PollFlag.RECEIVE) == PollFlag.RECEIVE):
                    continue # Data not available, skip receiving

                # Receive data
                self.bytesReceived = self.clientStream.receive_byte_array(self.data, totalNumBytes)

                # data received, so break this loop
                break

            #  convert byte array back into numpy array and reshape.
            self.receiveBuffer = np.reshape(np.frombuffer(self.data, dtype=self.receiveBuffer.dtype), dataShape, order = self.reshapeOrder)

        except StreamError as e:
            print(e.get_error_message())
        finally:
            receiveFlag = self.bytesReceived==1
            return receiveFlag, totalNumBytes*self.bytesReceived

    def send(self, buffer):
        """
        This functions sends the data in the numpy array buffer
        (server or client). \n \n

        INPUTS: \n
        buffer - numpy array of data to be sent \n

        OUTPUTS: \n
        bytesSent - number of bytes actually sent (-1 if send failed) \n
         \n
        Stream Server as an example \n
        >>> while True:
        >>> 	if not myServer.connected:
        >>> 		myServer.checkConnection()
        >>>		if myServer.connected:
        >>> 		sent = myServer.send()
        >>>			if breakCondition or sent == -1:
        >>>				break
        >>> myServer.terminate()
         \n
        Stream Client as an example	 \n
        >>> while True:
        >>> 	if not myClient.connected:
        >>> 		myClient.checkConnection()
        >>>		if myClient.connected:
        >>> 		sent = myServer.send()
        >>>			if breakCondition or sent == -1:
        >>>				break
        >>> myClient.terminate()

        """

        # Set up array to hold bytes to be sent
        byteArray = buffer.tobytes()
        self.bytesSent = 0

        # Send bytes and flush immediately after
        try:
            self.bytesSent = self.clientStream.send_byte_array(byteArray, len(byteArray))
            self.clientStream.flush()
        except StreamError as e:
            print(e.get_error_message())
            self.bytesSent = -1 # If an error occurs, set bytesSent to -1 for user to check
        finally:
            return self.bytesSent
        












        """vision.py: A module for simplifying interaction with various camera types.

This module provides a set of classes designed to make it easier to interface
with different types of cameras, such as standard 2D cameras, depth cameras,
or stereo cameras. The classes include functionality for connecting to cameras,
reading image data, and performing necessary conversions or transformations.
"""
import numpy as np

from quanser.multimedia import Video3D, Video3DStreamType, VideoCapture, \
    MediaError, ImageFormat, ImageDataType, VideoCapturePropertyCode, \
    VideoCaptureAttribute


class Camera3D():
    def __init__(
            self,
            mode='RGB, Depth',
            frameWidthRGB=1920,
            frameHeightRGB=1080,
            frameRateRGB=30.0,
            frameWidthDepth=1280,
            frameHeightDepth=720,
            frameRateDepth=15.0,
            frameWidthIR=1280,
            frameHeightIR=720,
            frameRateIR=15.0,
            deviceId='0',
            readMode=1,
            focalLengthRGB=np.array([[None], [None]], dtype=np.float64),
            principlePointRGB=np.array([[None], [None]], dtype=np.float64),
            skewRGB=None,
            positionRGB=np.array([[None], [None], [None]], dtype=np.float64),
            orientationRGB=np.array(
                [[None, None, None], [None, None, None], [None, None, None]],
                dtype=np.float64),
            focalLengthDepth=np.array([[None], [None]], dtype=np.float64),
            principlePointDepth=np.array([[None], [None]], dtype=np.float64),
            skewDepth=None,
            positionDepth=np.array([[None], [None], [None]], dtype=np.float64),
            orientationDepth=np.array(
                [[None, None, None], [None, None, None], [None, None, None]],
                dtype=np.float64)
        ):
        """This class configures RGB-D cameras (eg. Intel Realsense) for use.

        By default, mode is set to RGB&DEPTH, which reads both streams.
        Set it to RGB or DEPTH to get exclusive RGB or DEPTH streaming.
        If you specify focal lengths, principle points, skew as well as
        camera position & orientation in the world/inertial frame,
        camera instrinsics/extrinsic matrices can also be extracted
        using corresponding methods in this class.
        """

        self.mode = mode
        self.readMode = readMode
        self.streamIndex = 0

        self.imageBufferRGB = np.zeros(
            (frameHeightRGB, frameWidthRGB, 3),
            dtype=np.uint8
        )
        self.imageBufferDepthPX = np.zeros(
            (frameHeightDepth, frameWidthDepth, 1),
            dtype=np.uint8
        )
        self.imageBufferDepthM = np.zeros(
            (frameHeightDepth, frameWidthDepth, 1),
            dtype=np.float32
        )
        self.imageBufferIRLeft = np.zeros(
            (frameHeightIR, frameWidthIR, 1),
            dtype=np.uint8
        )
        self.imageBufferIRRight = np.zeros(
            (frameHeightIR, frameWidthIR, 1),
            dtype=np.uint8
        )

        self.frameWidthRGB = frameWidthRGB
        self.frameHeightRGB = frameHeightRGB
        self.frameWidthDepth = frameWidthDepth
        self.frameHeightDepth = frameHeightDepth
        self.frameWidthIR = frameWidthIR
        self.frameHeightIR = frameHeightIR

        self.focalLengthRGB = 2*focalLengthRGB
        self.focalLengthRGB[0, 0] = -self.focalLengthRGB[0, 0]
        self.principlePointRGB = principlePointRGB
        self.skewRGB = skewRGB
        self.positionRGB = positionRGB
        self.orientationRGB = orientationRGB

        self.focalLengthDepth = 2*focalLengthDepth
        self.focalLengthDepth[0, 0] = -self.focalLengthDepth[0, 0]
        self.principlePointDepth = principlePointDepth
        self.skewDepth = skewDepth
        self.positionDepth = positionDepth
        self.orientationDepth = orientationDepth

        try:
            self.video3d = Video3D(deviceId)
            self.streamOpened = False
            if 'rgb' in self.mode.lower():
                self.streamRGB = self.video3d.stream_open(
                    Video3DStreamType.COLOR,
                    self.streamIndex,
                    frameRateRGB,
                    frameWidthRGB,
                    frameHeightRGB,
                    ImageFormat.ROW_MAJOR_INTERLEAVED_BGR,
                    ImageDataType.UINT8
                )
                self.streamOpened = True
            if 'depth' in self.mode.lower():
                self.streamDepth = self.video3d.stream_open(
                    Video3DStreamType.DEPTH,
                    self.streamIndex,
                    frameRateDepth,
                    frameWidthDepth,
                    frameHeightDepth,
                    ImageFormat.ROW_MAJOR_GREYSCALE,
                    ImageDataType.UINT8
                )
                self.streamOpened = True
            if 'ir' in self.mode.lower():
                self.streamIRLeft = self.video3d.stream_open(
                    Video3DStreamType.INFRARED,
                    1,
                    frameRateIR,
                    frameWidthIR,
                    frameHeightIR,
                    ImageFormat.ROW_MAJOR_GREYSCALE,
                    ImageDataType.UINT8
                )
                self.streamIRRight = self.video3d.stream_open(
                    Video3DStreamType.INFRARED,
                    2,
                    frameRateIR,
                    frameWidthIR,
                    frameHeightIR,
                    ImageFormat.ROW_MAJOR_GREYSCALE,
                    ImageDataType.UINT8
                )
                self.streamOpened = True
            # else:
            #     self.streamRGB = self.video3d.stream_open(
            #         Video3DStreamType.COLOR,
            #         self.streamIndex,
            #         frameRateRGB,
            #         frameWidthRGB,
            #         frameHeightRGB,
            #         ImageFormat.ROW_MAJOR_INTERLEAVED_BGR,
            #         ImageDataType.UINT8
            #     )
            #     self.streamDepth = self.video3d.stream_open(
            #         Video3DStreamType.DEPTH,
            #         self.streamIndex,
            #         frameRateDepth,
            #         frameWidthDepth,
            #         frameHeightDepth,
            #         ImageFormat.ROW_MAJOR_GREYSCALE,
            #         ImageDataType.UINT8
            #     )
            #     self.streamOpened = True
            self.video3d.start_streaming()
        except MediaError as me:
            print(me.get_error_message())

    def terminate(self):
        """Terminates all started streams correctly."""

        try:
            self.video3d.stop_streaming()
            if self.streamOpened:
                if 'rgb' in self.mode.lower():
                    self.streamRGB.close()
                if 'depth' in self.mode.lower():
                    self.streamDepth.close()
                if 'ir' in self.mode.lower():
                    self.streamIRLeft.close()
                    self.streamIRRight.close()

            self.video3d.close()

        except MediaError as me:
            print(me.get_error_message())

    def read_RGB(self):
        """Reads an image from the RGB stream. It returns a timestamp
            for the frame just read. If no frame was available, it returns -1.
        """

        timestamp = -1
        try:
            frame = self.streamRGB.get_frame()
            while not frame:
                if not self.readMode:
                    break
                frame = self.streamRGB.get_frame()
            if not frame:
                pass
            else:
                frame.get_data(self.imageBufferRGB)
                timestamp = frame.get_timestamp()
                frame.release()
        except KeyboardInterrupt:
            pass
        except MediaError as me:
            print(me.get_error_message())
        finally:
            return timestamp

    def read_depth(self, dataMode='PX'):
        """Reads an image from the depth stream. Set dataMode to
            'PX' for pixels or 'M' for meters. Use the corresponding image
            buffer to get image data. If no frame was available, it returns -1.
        """
        timestamp = -1
        try:
            frame = self.streamDepth.get_frame()
            while not frame:
                if not self.readMode:
                    break
                frame = self.streamDepth.get_frame()
            if not frame:
                pass
            else:
                if dataMode == 'PX':
                    frame.get_data(self.imageBufferDepthPX)
                elif dataMode == 'M':
                    frame.get_meters(self.imageBufferDepthM)
                timestamp = frame.get_timestamp()
                frame.release()
        except KeyboardInterrupt:
            pass
        except MediaError as me:
            print(me.get_error_message())
        finally:
            return timestamp

    def read_IR(self, lens='LR'):
        """Reads an image from the left and right IR streams
            based on the lens parameter (L or R). Use the corresponding
            image buffer to get image data. If no frame was available,
            it returns -1.
        """
        timestamp = -1
        try:
            if 'l' in lens.lower():
                frame = self.streamIRLeft.get_frame()
                while not frame:
                    if not self.readMode:
                        break
                    frame = self.streamIRLeft.get_frame()
                if not frame:
                    pass
                else:
                    frame.get_data(self.imageBufferIRLeft)
                    timestamp = frame.get_timestamp()
                    frame.release()
            if 'r' in lens.lower():
                frame = self.streamIRRight.get_frame()
                while not frame:
                    if not self.readMode:
                        break
                    frame = self.streamIRRight.get_frame()
                if not frame:
                    pass
                else:
                    frame.get_data(self.imageBufferIRRight)
                    timestamp = frame.get_timestamp()
                    frame.release()
        except KeyboardInterrupt:
            pass
        except MediaError as me:
            print(me.get_error_message())
        finally:
            return timestamp

    def extrinsics_rgb(self):
        """Provides the Extrinsic Matrix for the RGB Camera"""
        # define rotation matrix from camera frame into the body frame
        transformFromCameraToBody = np.concatenate(
            (np.concatenate((self.orientationRGB, self.positionRGB), axis=1),
                [[0, 0, 0, 1]]),
            axis=0
        )

        return np.linalg.inv(transformFromCameraToBody)

    def intrinsics_rgb(self):
        """Provides the Intrinsic Matrix for the RGB Camera"""
        # construct the intrinsic matrix
        return np.array(
            [[self.focalLengthRGB[0,0], self.skewRGB,
                    self.principlePointRGB[0,0]],
             [0, self.focalLengthRGB[1,0], self.principlePointRGB[1,0]],
             [0, 0, 1]],
            dtype = np.float64
        )

    def extrinsics_depth(self):
        """Provides the Extrinsic Matrix for the Depth Camera"""
        # define rotation matrix from camera frame into the body frame
        transformFromCameraToBody = np.concatenate(
            (np.concatenate((self.orientationDepth, self.positionDepth),
                axis=1), [[0, 0, 0, 1]]),
            axis=0
        )

        return np.linalg.inv(transformFromCameraToBody)

    def intrinsics_depth(self):
        """Provides the Intrinsic Matrix for the Depth Camera"""
        # construct the intrinsic matrix
        return np.array(
            [[self.focalLengthDepth[0,0], self.skewDepth,
                self.principlePointDepth[0,0]],
             [0, self.focalLengthDepth[1,0], self.principlePointDepth[1,0]],
             [0, 0, 1]],
            dtype = np.float64
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.terminate()


class Camera2D():
    def __init__(
            self,
            cameraId="0",
            frameWidth=820,
            frameHeight=410,
            frameRate=30.0,
            focalLength=np.array([[None], [None]], dtype=np.float64),
            principlePoint=np.array([[None], [None]], dtype=np.float64),
            skew=None,
            position=np.array([[None], [None], [None]], dtype=np.float64),
            orientation=np.array(
                [[None,None,None], [None,None,None], [None,None,None]],
                dtype=np.float64),
            imageFormat = 0,
            brightness = None,
            contrast = None,
            gain = None,
            exposure = None,

        ):
        """Configures the 2D camera based on the cameraId provided.

        If you specify focal lengths, principle points, skew as well as
            camera position & orientation in the world/inertial frame,
            camera intrinsics/extrinsic matrices can also be extracted
            using corresponding methods in this class.

            image format defaults to 0. Outputs BGR images. If value is 1,
            will be set as greyscale.
        """
        self.url = "video://localhost:"+cameraId

        self.frameWidth = frameWidth
        self.frameHeight = frameHeight

        self.focalLength = 2*focalLength
        self.focalLength[0, 0] = -self.focalLength[0, 0]
        self.principlePoint = principlePoint
        self.skew = skew
        self.position = position
        self.orientation = orientation
        attributes = []

        if imageFormat == 0:
            self.imageFormat = ImageFormat.ROW_MAJOR_INTERLEAVED_BGR
            self.imageData = np.zeros((frameHeight, frameWidth, 3), dtype=np.uint8)
        else:
            self.imageFormat = ImageFormat.ROW_MAJOR_GREYSCALE
            self.imageData = np.zeros((frameHeight, frameWidth), dtype=np.uint8)

        if brightness is not None:
            attributes.append(VideoCaptureAttribute(VideoCapturePropertyCode.BRIGHTNESS, brightness, True))
        if contrast is not None:
            attributes.append(VideoCaptureAttribute(VideoCapturePropertyCode.CONTRAST, contrast, True))
        if gain:
            attributes.append(VideoCaptureAttribute(VideoCapturePropertyCode.GAIN, gain, True, False))
        if exposure is not None:
            attributes.append(VideoCaptureAttribute(VideoCapturePropertyCode.EXPOSURE, exposure, True, False))

        if not attributes:
            attributes = None
            numAttributes = 0
        else:
            numAttributes = len(attributes)


        try:
            self.capture = VideoCapture(
                self.url,
                frameRate,
                frameWidth,
                frameHeight,
                self.imageFormat,
                ImageDataType.UINT8,
                attributes,
                numAttributes
            )
            self.capture.start()
        except MediaError as me:
            print(me.get_error_message())

    def read(self):
        """Reads a frame, updating the corresponding image buffer. Returns a flag
        indicating whether the read was successful."""
        flag = False
        try:
            flag = self.capture.read(self.imageData)
        except MediaError as me:
            print(me.get_error_message())
        except KeyboardInterrupt:
            print('User Interrupted')
        finally:
            return flag

    def reset(self):
        """Resets the 2D camera stream by stopping and starting
            the capture service.
        """

        try:
            self.capture.stop()
            self.capture.start()
        except MediaError as me:
            print(me.get_error_message())

    def terminate(self):
        """Terminates the 2D camera operation. """
        try:
            self.capture.stop()
            self.capture.close()
        except MediaError as me:
            print(me.get_error_message())

    def extrinsics(self):
        """Provides the Extrinsic Matrix for the Camera"""
        # define rotation matrix from camera frame into the body frame
        transformFromCameraToBody = np.concatenate(
            (np.concatenate((self.orientation, self.position), axis=1),
                [[0, 0, 0, 1]]), axis=0)

        return np.linalg.inv(transformFromCameraToBody)

    def intrinsics(self):
        """Provides the Intrinsic Matrix for the Camera"""
        # construct the intrinsic matrix
        return np.array(
            [[self.focalLength[0,0], self.skew, self.principlePoint[0,0]],
             [0, self.focalLength[1,0], self.principlePoint[1,0]], [0, 0, 1]],
            dtype = np.float64
        )

    def __enter__(self):
        """Used for with statement."""
        return self

    def __exit__(self, type, value, traceback):
        """Used for with statement. Terminates the Camera"""
        self.terminate()



















this is the beginning of the pit folder:



    jetson:

    #!/usr/bin/python3

import jetson_inference
import jetson_utils
import time
import argparse
import numpy as np

class ImageNet():

    def __init__(
            self,network = 'googlenet',
            imageWidth=640,
            imageHeight=480, 
            threshold = 0.1, 
            showImage = True, 
            outputURI = "", 
            verbose=True
        ):
        '''
        threshold - used in combination with topK=0 for image tagging; 
        classes with confidence higher than threshold are shown
        network - pre-trained model to load, one of the following:
            alexnet
            googlenet
            googlenet-12
            resnet-18
            resnet-50
            resnet-101
            resnet-152
            vgg-16
            vgg-19
            inception-v4
        '''

        self.verbose=verbose
        self.font=jetson_utils.cudaFont()
        self.modelName = 'imageNet'
        
        if showImage:
            self._output = jetson_utils.videoOutput(outputURI)
        self.showImage = showImage
        
        self.net=jetson_inference.imageNet(network)
        self.net.SetThreshold(threshold)

        self.img=jetson_utils.cudaAllocMapped(width=imageWidth,
                                              height=imageHeight,
                                              format='rgb8')
        
        if not self.showImage:
                print('The parameter showImageOutput is set to False.',
                      'When running the render() function',
                      'it will never display the image despite being called.')
        
    def pre_process(self, inputImg):
        #https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-from-cv.py
        bgrImg = jetson_utils.cudaFromNumpy(inputImg, isBGR=True)
        jetson_utils.cudaConvertColor(bgrImg, self.img)
        return self.img

    def predict(self, inputImg, topK = 2, textOnImage = True): 

        self.predictions = self.net.Classify(inputImg, topK=topK)
        labelList = []
        confidenceList = []
        
        for n, (classID, confidence) in enumerate(self.predictions):
            classLabel = self.net.GetClassLabel(classID)
            confidence *= 100.0
            if self.verbose:
                print(f"{self.modelName}: {confidence:05.2f}%",
                        f"class #{classID} ({classLabel})")
            labelList.append(classLabel)
            confidenceList.append(confidence)

            if self.showImage:
                if textOnImage:
                    self.font.OverlayText(inputImg, 
                                text=f"{confidence:05.2f}% {classLabel}", 
                                x=5, 
                                y=5 + n * (self.font.GetSize() + 5),
                                color=self.font.White, 
                                background=self.font.Gray40)
            
        return labelList, confidenceList

    def render(self, printPerformance = False):
        if self.showImage:
            self._output.Render(self.img)
            # update the title bar
            self._output.SetStatus("{:s} | Network {:.0f} FPS".format(
                        self.net.GetNetworkName(), self.net.GetNetworkFPS()))
            
            if printPerformance:
                # print out performance info
                self.net.PrintProfilerTimes()
        else:
            pass #function will ony work 
                 #if showImageOutput was enabled on initialization

class ActionNet():
    def __init__(
            self,
            network = 'resnet-18',
            imageWidth=640,
            imageHeight=480, 
            threshold = 0.1, 
            skipFrames = 1, 
            showImage = True, 
            outputURI = "", 
            verbose=True
        ):
        '''
        threhold - class with confidence higher than threshold is shown
        skipFrames - number of frames to skip when 
                    using consecutive frames as input to actoinNet
        network - pre-trained model to load, one of the following:
            resnet-18
            resnet-34
        '''
        self.verbose=verbose
        self.font=jetson_utils.cudaFont()
        self.modelName = 'actionNet'
        
        if showImage:
            self._output = jetson_utils.videoOutput(outputURI)
        self.showImage = showImage
        
        self.net=jetson_inference.actionNet(network)
        self.net.SetThreshold(threshold)
        self.net.SetSkipFrames(skipFrames)

        self.img=jetson_utils.cudaAllocMapped(width=imageWidth,
                                              height=imageHeight,
                                              format='rgb8')
        
        if not self.showImage:
                print('The parameter showImageOutput is set to False.',
                      'When running the render() function',
                      'it will never display the image despite being called.')
                
    def pre_process(self, inputImg):
        #https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-from-cv.py
        bgrImg = jetson_utils.cudaFromNumpy(inputImg, isBGR=True)
        jetson_utils.cudaConvertColor(bgrImg, self.img)
        return self.img

    def predict(self, inputImg, textOnImage = True): 

        self.predictions = self.net.Classify(inputImg)
        labelList = []
        confidenceList = []

        
        classID, confidence = self.predictions
        confidence*=100

        if classID==-1:
            return labelList, confidenceList

        classLabel = self.net.GetClassDesc(classID)
        if self.verbose:
            print(f"{self.modelName}: {confidence:05.2f}%",
                        f"class #{classID} ({classLabel})")
        
        labelList.append(classLabel)
        confidenceList.append(confidence)

        if self.showImage and textOnImage:
            self.font.OverlayText(
                inputImg, 
                inputImg.width, 
                inputImg.height, 
                "{:05.2f}% {:s}".format(confidence, classLabel), 
                x=5, y=5,
                color=self.font.White, 
                background=self.font.Gray40)
                
        return labelList, confidenceList

    def render(self, printPerformance = False):
        if self.showImage:
            self._output.Render(self.img)
            # update the title bar
            self._output.SetStatus("{:s} | Network {:.0f} FPS".format(
                        self.net.GetNetworkName(), self.net.GetNetworkFPS()))
            
            if printPerformance:
                # print out performance info
                self.net.PrintProfilerTimes()
        else:
            pass #function will ony work 
                 #if showImageOutput was enabled on initialization
         
class DetectNet():
    def __init__(
            self,
            network = 'SSD-Mobilenet-v2',
            imageWidth=640,
            imageHeight=480, 
            threshold = 0.5, 
            alpha = 120, 
            lineWidth = 2.0, 
            showImage = True, 
            outputURI = "", 
            verbose=True
        ):
        """
        threshold - classes with confidence higher than threshold are shown
        alpha - overlay alpha blending value, range 0-255 (default: 120)
        lineWidth - used during overlay when 'lines' is used
        network - pre-trained model to load, one of the following:
            ssd-mobilenet-v1
            ssd-mobilenet-v2 (default)
            ssd-inception-v2
            peoplenet
            peoplenet-pruned
            dashcamnet
            trafficcamnet
            facedetect
        """
        self.network = network
        self.verbose=verbose
        self.modelName = 'detectNet'
        if showImage:
            self._output = jetson_utils.videoOutput(outputURI)
        self.showImage = showImage
        
        self.net=jetson_inference.detectNet(network)

        self.net.SetConfidenceThreshold(threshold)
        self.net.SetLineWidth(lineWidth)
        self.net.SetOverlayAlpha(alpha)

        self.img=jetson_utils.cudaAllocMapped(width=imageWidth,
                                              height=imageHeight,
                                              format='rgb8')
        
        if not self.showImage:
                print('The parameter showImageOutput is set to False.',
                      'When running the render() function',
                      'it will never display the image despite being called.')

    def pre_process(self, inputImg):
        #https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-from-cv.py
        bgrImg = jetson_utils.cudaFromNumpy(inputImg, isBGR=True)
        jetson_utils.cudaConvertColor(bgrImg, self.img)
        return self.img

    def predict(self, inputImg, overlay = 'box,labels,conf'): 
        # valid combinations are:  'box', 'lines', 'labels', 'conf', 'none'
        # it is possible to combine flags. See default.
	    # (bitwise OR) together with commas or pipe (|) symbol.

        self.predictions = self.net.Detect(inputImg, overlay=overlay)
        labelList = []
        confidenceList = []

        # if type(self.predictions) is tuple:
        #     self.predictions=[self.predictions]
        for n, detection in enumerate(self.predictions):
            classLabel = self.net.GetClassLabel(detection.ClassID)
            confidence = detection.Confidence*100.0
            if self.verbose:
                print(f"{self.modelName}: {confidence:05.2f}%",
                        f"class #{detection.ClassID} ({classLabel})")

            labelList.append(classLabel)
            confidenceList.append(confidence)
        
        return labelList, confidenceList
            
    def render(self, printPerformance = False):
        if self.showImage:
            self._output.Render(self.img)
            # update the title bar
            self._output.SetStatus("{:s} | Network {:.0f} FPS".format(
                                self.network, self.net.GetNetworkFPS()))
            if printPerformance:
                # print out performance info
                self.net.PrintProfilerTimes()
        else:
            pass #function will ony work 
                 #if showImageOutput was enabled on initialization
 
class PoseNet():
    def __init__(
            self, 
            network = 'resnet18-body',
            imageWidth=640,
            imageHeight=480, 
            threshold = 0.15, 
            keypointScale = 0.0052, 
            linkScale = 0.0013, 
            showImage = True, 
            outputURI = "", 
            verbose=True
        ):

        '''
        threshold - value which sets the minimum threshold for detection 
                    (the default is 0.15)
        keypointScale - value which controls the radius 
                        of the keypoint circles in the overlay 
                        (the default is 0.0052)
        linkScale - value which controls the line width 
                    of the link lines in the overlay 
                    (the default is 0.0013)
        network - pre-trained model to load, one of the following:
            resnet18-body
		    resnet18-hand
		    densenet121-body
        '''

        self.network = network
        self.verbose=verbose
        self.font=jetson_utils.cudaFont()
        self.modelName = 'poseNet'
        if showImage:
            self._output = jetson_utils.videoOutput(outputURI)
        self.showImage = showImage
        
        self.net=jetson_inference.poseNet(network)

        self.net.SetThreshold(threshold)
        self.net.SetKeypointScale(keypointScale)
        self.net.SetLinkScale(linkScale)

        self.img=jetson_utils.cudaAllocMapped(width=imageWidth,
                                              height=imageHeight,
                                              format='rgb8')
        
        if not self.showImage:
                print('The parameter showImageOutput is set to False.',
                      'When running the render() function',
                      'it will never display the image despite being called.')
        
    def pre_process(self, inputImg):
        #https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-from-cv.py
        bgrImg = jetson_utils.cudaFromNumpy(inputImg, isBGR=True)
        jetson_utils.cudaConvertColor(bgrImg, self.img)
        return self.img

    def predict(self, inputImg, overlay = 'links,keypoints'):
        # detection overlay flags (e.g. --overlay=links,keypoints)
		# valid combinations are:  'box', 'links', 'keypoints', 'none'

        self.poses = self.net.Process(inputImg, overlay=overlay)

        if self.verbose:
            # print the pose results
            print("detected {:d} objects in image".format(len(self.poses)))
            # for pose in self.poses:
            #     print(pose)
            #     print(pose.Keypoints)
            #     print('Links', pose.Links)  
        
        return self.poses
  
    def render(self, printPerformance = False):
        if self.showImage:
            self._output.Render(self.img)
            # update the title bar
            self._output.SetStatus("{:s} | Network {:.0f} FPS".format(
                                self.network, self.net.GetNetworkFPS()))
            
            if printPerformance:
                # print out performance info
                self.net.PrintProfilerTimes()
        else:
            pass #function will ony work 
                 # if showImageOutput was enabled on initialization

class DepthNet():
    def __init__(
            self, 
            network = 'fcn-mobilenet',
            imageWidth=640,
            imageHeight=480, 
            visualize = 'input,depth', 
            depthSize = 1, 
            showImage = True, 
            outputURI = "", 
            verbose=True
        ):
        
        '''
        visualize - decide what to desplay. Default displays input and depth
        images side-by-side. To view only depth use visualze='depth'
        visualize - visualization options 
        (can be 'input' 'depth' 'input,depth')
        depth-size - value which scales the size of the depth map 
        relative to the input (the default is 1.0)
        network - pre-trained model to load, one of the following:
            fcn-mobilenet
            fcn-resnet18
            fcn-resnet50
        '''

        self.network = network
        self.verbose=verbose
        self.modelName = 'depthNet'
        self.useInput = 'input' in visualize
        self.useDepth = 'depth' in visualize

        self.net=jetson_inference.depthNet(network)
        if showImage:
            self._output = jetson_utils.videoOutput(outputURI)
        self.showImage = showImage

        self.depth = None
        self.composite = None
        self.depthSize = depthSize

        depth_size = (imageHeight * self.depthSize, 
                      imageWidth * self.depthSize)
        
        composite_size = [0,0]

        if self.useDepth:
            composite_size[0] = depth_size[0]
            composite_size[1] += depth_size[1]
            
        if self.useDepth:
            composite_size[0] = imageHeight
            composite_size[1] += imageWidth

        self.depth = jetson_utils.cudaAllocMapped(width=depth_size[1], 
                                                  height=depth_size[0], 
                                                  format='rgb8')
        
        self.composite = jetson_utils.cudaAllocMapped(width=composite_size[1], 
                                                      height=composite_size[0], 
                                                      format='rgb8')

        self.img=jetson_utils.cudaAllocMapped(width=imageWidth,
                                              height=imageHeight,
                                              format='rgb8')
        
        if not self.showImage:
                print('The parameter showImageOutput is set to False.',
                      'When running the render() function',
                      'it will never display the image despite being called.')
  
    def pre_process(self, inputImg):
        #https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-from-cv.py
        bgrImg = jetson_utils.cudaFromNumpy(inputImg, isBGR=True)
        jetson_utils.cudaConvertColor(bgrImg, self.img)
        return self.img

    def predict(
            self, 
            inputImg, 
            filter_mode = 'linear', 
            colormap = 'viridis_inverted'
        ):
        
        # colormap - choices=["inferno", "inferno-inverted", 
        #           "magma", "magma-inverted", "parula", "parula-inverted", 
        #            "plasma", "plasma-inverted", "turbo", "turbo-inverted", 
        #            "viridis", "viridis-inverted"])
        # filtermode - filtering mode used during visualization, 
        #           options are: 'point' or 'linear' (default: 'linear')

        self.net.Process(inputImg, self.depth, colormap, filter_mode)

        if self.useInput:
            jetson_utils.cudaOverlay(inputImg, self.composite, 0, 0)

        if self.useDepth:
            jetson_utils.cudaOverlay(self.depth, 
                                     self.composite, 
                                     inputImg.width if self.useInput else 0, 0)

        return jetson_utils.cudaToNumpy(self.depth).squeeze()

    def render(self, printPerformance = False):
        # for consistency the input image argument is there,
        # but nothing happens with it, it grabs the correct 
        # image from the predict function
        if self.showImage:
            self._output.Render(self.composite)
            # update the title bar
            self._output.SetStatus("{:s} | Network {:.0f} FPS".format(
                    self.net.GetNetworkName(), self.net.GetNetworkFPS()))
            jetson_utils.cudaDeviceSynchronize()
            if printPerformance:
                # print out performance info
                self.net.PrintProfilerTimes()
        else:
            pass #function will ony work 
                 #if showImageOutput was enabled on initialization

class SegNet():
    def __init__(
            self,
            network = 'fcn-resnet18-voc',
            imageWidth=640,
            imageHeight=480, 
            visualize='overlay,mask', 
            filter_mode = 'linear', 
            alpha=120.0, 
            showImage = True, 
            outputURI = "",
            verbose=True
        ):
        '''
        visualize - flag accepts mask and/or overlay modes 
        (default is 'overlay,mask')
        alpha - flag sets the alpha blending value for overlay 
        (default is 120)
        filter_mode - flag accepts point or linear sampling 
        (default is linear)
        network - pre-trained model to load, one of the following:
            *If the resolution is omitted from the argument, 
            the lowest resolution model is loaded*
            fcn-resnet18-cityscapes
            fcn-resnet18-cityscapes-512x256
            fcn-resnet18-cityscapes-1024x512
            fcn-resnet18-cityscapes-2048x1024
            fcn-resnet18-deepscene
            fcn-resnet18-deepscene-576x320
            fcn-resnet18-deepscene-864x480
            fcn-resnet18-mhp
            fcn-resnet18-mhp-512x320
            fcn-resnet18-mhp-640x360
            fcn-resnet18-voc
            fcn-resnet18-voc-512x320
            fcn-resnet18-voc-320x320
            fcn-resnet18-sun
            fcn-resnet18-sun-512x400
            fcn-resnet18-sun-640x512
        '''
        self.modelname = 'segNet'
        self.filter_mode = filter_mode
        if showImage:
            self._output = jetson_utils.videoOutput(outputURI)
        self.showImage = showImage
        
        self.net=jetson_inference.segNet(network)
        self.img=jetson_utils.cudaAllocMapped(width=imageWidth,
                                              height=imageHeight,
                                              format='rgb8')
        self.mask = None
        self.overlay = None
        self.composite = None
        self.verbose = verbose
        self.use_mask = "mask" in visualize
        self.use_overlay = "overlay" in visualize
        self.use_composite = self.use_mask and self.use_overlay
        

        if not self.showImage:
                print('The parameter showImageOutput is set to False.',
                      'When running the render() function',
                      'it will never display the image despite being called.')

        if showImage and not self.use_overlay and not self.use_mask:
            raise Exception("invalid visualize flags - ",
                            "valid values are 'overlay' 'mask' 'overlay,mask'")
        
        self.net.SetOverlayAlpha(alpha)
        self.grid_width, self.grid_height = self.net.GetGridSize()	
        self.num_classes = self.net.GetNumClasses()

        self.class_mask = jetson_utils.cudaAllocMapped(width=self.grid_width, 
                                                       height=self.grid_height, 
                                                       format="gray8")
        
        self.class_mask_np = jetson_utils.cudaToNumpy(self.class_mask)

        if self.use_overlay:
            self.overlay = jetson_utils.cudaAllocMapped(width=imageWidth, 
                                                        height=imageHeight, 
                                                        format='rgb8')

        if self.use_mask:
            mask_downsample = 2 if self.use_overlay else 1
            self.mask = jetson_utils.cudaAllocMapped(
                                    width=imageWidth/mask_downsample, 
                                    height=imageHeight/mask_downsample, 
                                    format='rgb8') 

        if self.use_composite:
            self.composite = jetson_utils.cudaAllocMapped(
                                    width=self.overlay.width+self.mask.width, 
                                    height=self.overlay.height, 
                                    format='rgb8') 

    def pre_process(self, inputImg):
        #https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-from-cv.py
        bgrImg = jetson_utils.cudaFromNumpy(inputImg, isBGR=True)
        jetson_utils.cudaConvertColor(bgrImg, self.img)
        return self.img

    def predict(self, inputImg, ignore_class = "void"): 
        # ignore_class - class to ignore during segmentation 
        # (defalut = 'void')
        
        self.net.Process(inputImg, ignore_class=ignore_class)
        self.net.Mask(self.class_mask, self.grid_width, self.grid_height)
        # generate the overlay
        if self.overlay:
            self.net.Overlay(self.overlay, filter_mode=self.filter_mode)

        # generate the mask
        if self.mask:
            self.net.Mask(self.mask, filter_mode=self.filter_mode)

        # composite the images
        if self.composite:
            jetson_utils.cudaOverlay(self.overlay, self.composite, 0, 0)
            jetson_utils.cudaOverlay(self.mask, 
                                     self.composite, 
                                     self.overlay.width, 
                                     0)
        
        label_mask_np=self.class_mask_np.copy().squeeze()
        if self.verbose:
            self.net.Mask(self.class_mask, self.grid_width, self.grid_height)

            # compute the number of times each class occurs in the mask
            class_histogram, _ = np.histogram(self.class_mask_np, 
                                              bins=self.num_classes, 
                                              range=(0, self.num_classes-1))

            print('grid size:   {:d}x{:d}'.format(self.grid_width, 
                                                  self.grid_height))
            
            print('num classes: {:d}'.format(self.num_classes))

            print('-----------------------------------------')
            print(' ID  class name        count     %')
            print('-----------------------------------------')

            for n in range(self.num_classes):
                percentage = (
                    float(class_histogram[n]) / (
                        float(self.grid_width * self.grid_height)))
                
                print(' {:>2d}  {:<18s} {:>3d}   {:f}'.format(
                                                    n, 
                                                    self.net.GetClassDesc(n), 
                                                    class_histogram[n], 
                                                    percentage))
        
        return label_mask_np

    def render(self, printPerformance=False):
        
        if self.showImage:
            if printPerformance:
                # print out performance info
                self.net.PrintProfilerTimes()
            if self.use_overlay and self.use_mask:
                self._output.Render(self.composite)
            elif self.use_overlay:
                self._output.Render(self.overlay)
            elif self.use_mask:
                self._output.Render(self.mask)
        else:
            pass #function will ony work 
                 #if showImageOutput was enabled on initialization
        











# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

from .batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
from .replicate import DataParallelWithCallback, patch_replication_callback














# -*- coding: utf-8 -*-
# File   : batchnorm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
#
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import collections

import torch
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast

from .comm import SyncMaster

__all__ = ['SynchronizedBatchNorm1d', 'SynchronizedBatchNorm2d', 'SynchronizedBatchNorm3d']


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i*2:i*2+2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        return mean, bias_var.clamp(self.eps) ** -0.5


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    r"""Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm1d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm1d, self)._check_input_dim(input)


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 5d input that is seen as a mini-batch
    of 4d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm3d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchNorm
    or Spatio-temporal BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x depth x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm3d, self)._check_input_dim(input)










#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : batchnorm_reimpl.py
# Author : acgtyrant
# Date   : 11/01/2018
#
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = ['BatchNormReimpl']


class BatchNorm2dReimpl(nn.Module):
    """
    A re-implementation of batch normalization, used for testing the numerical
    stability.

    Author: acgtyrant
    See also:
    https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.empty(num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input_):
        batchsize, channels, height, width = input_.size()
        numel = batchsize * height * width
        input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
        sum_ = input_.sum(1)
        sum_of_square = input_.pow(2).sum(1)
        mean = sum_ / numel
        sumvar = sum_of_square - sum_ * mean

        self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * mean.detach()
        )
        unbias_var = sumvar / (numel - 1)
        self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * unbias_var.detach()
        )

        bias_var = sumvar / numel
        inv_std = 1 / (bias_var + self.eps).pow(0.5)
        output = (
                (input_ - mean.unsqueeze(1)) * inv_std.unsqueeze(1) *
                self.weight.unsqueeze(1) + self.bias.unsqueeze(1))

        return output.view(channels, batchsize, height, width).permute(1, 0, 2, 3).contiguous()









# -*- coding: utf-8 -*-
# File   : replicate.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import functools

from torch.nn.parallel.data_parallel import DataParallel

__all__ = [
    'CallbackContext',
    'execute_replication_callbacks',
    'DataParallelWithCallback',
    'patch_replication_callback'
]


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]

    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


def patch_replication_callback(data_parallel):
    """
    Monkey-patch an existing `DataParallel` object. Add the replication callback.
    Useful when you have customized `DataParallel` implementation.

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        > patch_replication_callback(sync_bn)
        # this is equivalent to
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    """

    assert isinstance(data_parallel, DataParallel)

    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    data_parallel.replicate = new_replicate









# -*- coding: utf-8 -*-
# File   : replicate.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import functools

from torch.nn.parallel.data_parallel import DataParallel

__all__ = [
    'CallbackContext',
    'execute_replication_callbacks',
    'DataParallelWithCallback',
    'patch_replication_callback'
]


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]

    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


def patch_replication_callback(data_parallel):
    """
    Monkey-patch an existing `DataParallel` object. Add the replication callback.
    Useful when you have customized `DataParallel` implementation.

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        > patch_replication_callback(sync_bn)
        # this is equivalent to
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    """

    assert isinstance(data_parallel, DataParallel)

    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    data_parallel.replicate = new_replicate















# -*- coding: utf-8 -*-
# File   : comm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import queue
import collections
import threading

__all__ = ['FutureResult', 'SlavePipe', 'SyncMaster']


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, 'Previous result has\'t been fetched.'
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()

            res = self._result
            self._result = None
            return res


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])
_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True

        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)



















# -*- coding: utf-8 -*-
# File   : unittest.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
#
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import unittest
import torch


class TorchTestCase(unittest.TestCase):
    def assertTensorClose(self, x, y):
        adiff = float((x - y).abs().max())
        if (y == 0).all():
            rdiff = 'NaN'
        else:
            rdiff = float((adiff / y).abs().max())

        message = (
            'Tensor close check failed\n'
            'adiff={}\n'
            'rdiff={}\n'
        ).format(adiff, rdiff)
        self.assertTrue(torch.allclose(x, y), message)












    '''deeplabv3_plus folder:'''

        '''_init_ had nothing, go to the next file'''









        # ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pit.LaneNet.architecture.backbone.deeplabv3_plus.sync_batchnorm import SynchronizedBatchNorm2d

class ASPP(nn.Module):
	
	def __init__(self, dim_in, dim_out, rate=1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				# SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
				nn.BatchNorm2d(dim_out),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate,bias=True),
				# SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
				nn.BatchNorm2d(dim_out),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate,bias=True),
				# SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
				nn.BatchNorm2d(dim_out),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate,bias=True),
				# SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
				nn.BatchNorm2d(dim_out),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out)   # SynchronizedBatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)
		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				# SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
				nn.BatchNorm2d(dim_out),
				nn.ReLU(inplace=True),		
		)
#		self.conv_cat = nn.Sequential(
#				nn.Conv2d(dim_out*4, dim_out, 1, 1, padding=0),
#				SynchronizedBatchNorm2d(dim_out),
#				nn.ReLU(inplace=True),		
#		)
	def forward(self, x):
		[b,c,row,col] = x.size()
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row,col), None, 'bilinear', True)
		
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
		result = self.conv_cat(feature_cat)
		return result


        








        # ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
import pit.LaneNet.architecture.backbone.deeplabv3_plus.resnet_atrous as atrousnet
import pit.LaneNet.architecture.backbone.deeplabv3_plus.xception as xception

def build_backbone(backbone_name, pretrained=True, os=16):
	if backbone_name == 'res50_atrous':
		net = atrousnet.resnet50_atrous(pretrained=pretrained, os=os)
		return net
	elif backbone_name == 'res101_atrous':
		net = atrousnet.resnet101_atrous(pretrained=pretrained, os=os)
		return net
	elif backbone_name == 'res152_atrous':
		net = atrousnet.resnet152_atrous(pretrained=pretrained, os=os)
		return net
	elif backbone_name == 'xception' or backbone_name == 'Xception':
		net = xception.xception(pretrained=pretrained, os=os)
		return net
	else:
		raise ValueError('backbone.py: The backbone named %s is not supported yet.'%backbone_name)
	

	


        









        # ----------------------------------------
# Written by Yude Wang
# Change by Iroh Cao
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from pit.LaneNet.architecture.backbone.deeplabv3_plus.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from pit.LaneNet.architecture.backbone.deeplabv3_plus.backbone import build_backbone
from pit.LaneNet.architecture.backbone.deeplabv3_plus.ASPP import ASPP

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class Deeplabv3plus_Encoder(nn.Module):
	def __init__(self):
		super(Deeplabv3plus_Encoder, self).__init__()
		self.backbone = None		
		self.backbone_layers = None
		input_channel = 2048		
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=256, 
				rate=16//16)
		self.dropout1 = nn.Dropout(0.5)

		indim = 256
		self.shortcut_conv = nn.Sequential(
				nn.Conv2d(indim, 48, 1, 1, padding=1//2,bias=True),
				# SynchronizedBatchNorm2d(48, momentum=0.0003),
				nn.BatchNorm2d(48),
				nn.ReLU(inplace=True),		
		)		

		# for m in self.modules():
		# 	if isinstance(m, nn.Conv2d):
		# 		nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
		# 	elif isinstance(m, SynchronizedBatchNorm2d):
		# 		nn.init.constant_(m.weight, 1)
		# 		nn.init.constant_(m.bias, 0)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				weights_init_kaiming(m)
			elif isinstance(m, nn.BatchNorm2d):
				weights_init_kaiming(m)

		self.backbone = build_backbone('res101_atrous', os=16)
		self.backbone_layers = self.backbone.get_layers()
	
	def forward(self, x):
		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
		feature_aspp = self.dropout1(feature_aspp)
		feature_shallow = self.shortcut_conv(layers[0])
		
		return feature_aspp, feature_shallow

class Deeplabv3plus_Decoder(nn.Module):
	def __init__(self, out_dim):
		super(Deeplabv3plus_Decoder, self).__init__()

		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=16//4)
	
		self.cat_conv = nn.Sequential(
				nn.Conv2d(304, 256, 3, 1, padding=1,bias=True),
				# SynchronizedBatchNorm2d(256, momentum=0.0003),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(256, 256, 3, 1, padding=1,bias=True),
				# SynchronizedBatchNorm2d(256, momentum=0.0003),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)
		self.cls_conv = nn.Conv2d(256, out_dim, 1, 1, padding=0)
		# for m in self.modules():
		# 	if isinstance(m, nn.Conv2d):
		# 		nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
		# 	elif isinstance(m, SynchronizedBatchNorm2d):
		# 		nn.init.constant_(m.weight, 1)
		# 		nn.init.constant_(m.bias, 0)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				weights_init_kaiming(m)
			elif isinstance(m, nn.BatchNorm2d):
				weights_init_kaiming(m)
	
	def forward(self, feature_aspp, feature_shallow):
    		
		feature_aspp = self.upsample_sub(feature_aspp)
		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		result = self.cat_conv(feature_cat) 
		result = self.cls_conv(result)
		result = self.upsample4(result)
		return result

# class deeplabv3plus(nn.Module):
# 	def __init__(self, cfg):
# 		super(deeplabv3plus, self).__init__()
# 		self.backbone = None		
# 		self.backbone_layers = None
# 		input_channel = 2048		
# 		self.aspp = ASPP(dim_in=input_channel, 
# 				dim_out=cfg.MODEL_ASPP_OUTDIM, 
# 				rate=16//cfg.MODEL_OUTPUT_STRIDE,
# 				bn_mom = cfg.TRAIN_BN_MOM)
# 		self.dropout1 = nn.Dropout(0.5)
# 		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
# 		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE//4)

# 		indim = 256
# 		self.shortcut_conv = nn.Sequential(
# 				nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1, padding=cfg.MODEL_SHORTCUT_KERNEL//2,bias=True),
# 				SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
# 				nn.ReLU(inplace=True),		
# 		)		
# 		self.cat_conv = nn.Sequential(
# 				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
# 				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
# 				nn.ReLU(inplace=True),
# 				nn.Dropout(0.5),
# 				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
# 				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
# 				nn.ReLU(inplace=True),
# 				nn.Dropout(0.1),
# 		)
# 		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
# 		for m in self.modules():
# 			if isinstance(m, nn.Conv2d):
# 				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
# 			elif isinstance(m, SynchronizedBatchNorm2d):
# 				nn.init.constant_(m.weight, 1)
# 				nn.init.constant_(m.bias, 0)
# 		self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
# 		self.backbone_layers = self.backbone.get_layers()

# 	def forward(self, x):
# 		x_bottom = self.backbone(x)
# 		layers = self.backbone.get_layers()
# 		feature_aspp = self.aspp(layers[-1])
# 		feature_aspp = self.dropout1(feature_aspp)
# 		feature_aspp = self.upsample_sub(feature_aspp)

# 		feature_shallow = self.shortcut_conv(layers[0])
# 		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
# 		result = self.cat_conv(feature_cat) 
# 		result = self.cls_conv(result)
# 		result = self.upsample4(result)
# 		return result












import torch.nn as nn
import math
import torchvision.models as models

class ResNet(nn.modules):
	
	def __init__(self, layers, atrous, pretrained=True):
		super(ResNet, self).__init__()
		self.inner_layer = []
		if layers == 18:
			self.backbone = models.resnet18(pretrained=pretrained)
		elif layers == 34:
			self.backbone = models.resnet34(pretrained=pretrained)
		elif layers == 50:
			self.backbone = models.resnet50(pretrained=pretrained)
		elif layers == 101:
			self.backbone = models.resnet101(pretrained=pretrained)
		elif layers == 152:
			self.backbone = models.resnet152(pretrained=pretrained)
		else:
			raise ValueError('resnet.py: network layers is no support yet')
		
		def hook_func(module, input, output):
			self.inner_layer.append(output)

		self.backbone.layer1.register_forward_hook(hook_func)	
		self.backbone.layer2.register_forward_hook(hook_func)
		self.backbone.layer3.register_forward_hook(hook_func)
		self.backbone.layer4.register_forward_hook(hook_func)

	def forward(self,x):
		self.inner_layer = []
		

        
















        import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from pit.LaneNet.architecture.backbone.deeplabv3_plus.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

bn_mom = 0.0003
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1, atrous=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1*atrous, dilation=atrous, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, atrous)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.bn1 = SynchronizedBatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = SynchronizedBatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.bn1 = SynchronizedBatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1*atrous, dilation=atrous, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = SynchronizedBatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        # self.bn3 = SynchronizedBatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Atrous(nn.Module):

    def __init__(self, block, layers, atrous=None, os=16):
        super(ResNet_Atrous, self).__init__()
        stride_list = None
        if os == 8:
            stride_list = [2,1,1]
        elif os == 16:
            stride_list = [2,2,1]
        else:
            raise ValueError('resnet_atrous.py: output stride=%d is not supported.'%os) 
            
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
#        self.conv1 =  nn.Sequential(
#                          nn.Conv2d(3,64,kernel_size=3, stride=2, padding=1),
#                          nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1),
#                          nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1),
#                      )
        # self.bn1 = SynchronizedBatchNorm2d(64, momentum=bn_mom)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 256, 128, layers[1], stride=stride_list[0])
        self.layer3 = self._make_layer(block, 512, 256, layers[2], stride=stride_list[1], atrous=16//os)
        self.layer4 = self._make_layer(block, 1024, 512, layers[3], stride=stride_list[2], atrous=[item*16//os for item in atrous])
        #self.layer5 = self._make_layer(block, 2048, 512, layers[3], stride=1, atrous=[item*16//os for item in atrous])
        #self.layer6 = self._make_layer(block, 2048, 512, layers[3], stride=1, atrous=[item*16//os for item in atrous])
        #self.layer7 = self._make_layer(block, 2048, 512, layers[3], stride=1, atrous=[item*16//os for item in atrous])
        self.layers = []

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, SynchronizedBatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weights_init_kaiming(m)
            elif isinstance(m, nn.BatchNorm2d):
                weights_init_kaiming(m)

    def get_layers(self):
        return self.layers

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, atrous=None):
        downsample = None
        if atrous == None:
            atrous = [1]*blocks
        elif isinstance(atrous, int):
            atrous_list = [atrous]*blocks
            atrous = atrous_list
        if stride != 1 or inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # SynchronizedBatchNorm2d(planes * block.expansion, momentum=bn_mom),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes, planes, stride=stride, atrous=atrous[0], downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes*block.expansion, planes, stride=1, atrous=atrous[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        self.layers.append(x)
        x = self.layer2(x)
        self.layers.append(x)
        x = self.layer3(x)
        self.layers.append(x)
        x = self.layer4(x)
        #x = self.layer5(x)
        #x = self.layer6(x)
        #x = self.layer7(x)
        self.layers.append(x)

        return x

def resnet50_atrous(pretrained=True, os=16, **kwargs):
    """Constructs a atrous ResNet-50 model."""
    model = ResNet_Atrous(Bottleneck, [3, 4, 6, 3], atrous=[1,2,1], os=os, **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        old_dict = {k: v for k,v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict) 
    return model


def resnet101_atrous(pretrained=True, os=16, **kwargs):
    """Constructs a atrous ResNet-101 model."""
    model = ResNet_Atrous(Bottleneck, [3, 4, 23, 3], atrous=[2,2,2], os=os, **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        old_dict = {k: v for k,v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict) 
    return model


def resnet152_atrous(pretrained=True, os=16, **kwargs):
    """Constructs a atrous ResNet-152 model."""
    model = ResNet_Atrous(Bottleneck, [3, 8, 36, 3], atrous=[1,2,1], os=os, **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet152'])
        model_dict = model.state_dict()
        old_dict = {k: v for k,v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict) 
    return model

    















    """ 
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)
@author: tstandley
Adapted by cadene
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from pit.LaneNet.architecture.backbone.deeplabv3_plus.sync_batchnorm import SynchronizedBatchNorm2d

bn_mom = 0.0003
__all__ = ['xception']

model_urls = {
    'xception': '/home/wangyude/.torch/models/xception_pytorch_imagenet.pth'#'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
}

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False,activate_first=True,inplace=True):
        super(SeparableConv2d,self).__init__()
        self.relu0 = nn.ReLU(inplace=inplace)
        self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.bn1 = SynchronizedBatchNorm2d(in_channels, momentum=bn_mom)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
        self.bn2 = SynchronizedBatchNorm2d(out_channels, momentum=bn_mom)
        self.relu2 = nn.ReLU(inplace=True)
        self.activate_first = activate_first
    def forward(self,x):
        if self.activate_first:
            x = self.relu0(x)
        x = self.depthwise(x)
        x = self.bn1(x)
        if not self.activate_first:
            x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        if not self.activate_first:
            x = self.relu2(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,strides=1,atrous=None,grow_first=True,activate_first=True,inplace=True):
        super(Block, self).__init__()
        if atrous == None:
            atrous = [1]*3
        elif isinstance(atrous, int):
            atrous_list = [atrous]*3
            atrous = atrous_list
        idx = 0
        self.head_relu = True
        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = SynchronizedBatchNorm2d(out_filters, momentum=bn_mom)
            self.head_relu = False
        else:
            self.skip=None
        
        self.hook_layer = None
        if grow_first:
            filters = out_filters
        else:
            filters = in_filters
        self.sepconv1 = SeparableConv2d(in_filters,filters,3,stride=1,padding=1*atrous[0],dilation=atrous[0],bias=False,activate_first=activate_first,inplace=self.head_relu)
        self.sepconv2 = SeparableConv2d(filters,out_filters,3,stride=1,padding=1*atrous[1],dilation=atrous[1],bias=False,activate_first=activate_first)
        self.sepconv3 = SeparableConv2d(out_filters,out_filters,3,stride=strides,padding=1*atrous[2],dilation=atrous[2],bias=False,activate_first=activate_first,inplace=inplace)

    def forward(self,inp):
        
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = self.sepconv1(inp)
        x = self.sepconv2(x)
        self.hook_layer = x
        x = self.sepconv3(x)

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, os):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        stride_list = None
        if os == 8:
            stride_list = [2,1,1]
        elif os == 16:
            stride_list = [2,2,1]
        else:
            raise ValueError('xception.py: output stride=%d is not supported.'%os) 
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(32, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32,64,3,1,1,bias=False)
        self.bn2 = SynchronizedBatchNorm2d(64, momentum=bn_mom)
        #do relu here

        self.block1=Block(64,128,2)
        self.block2=Block(128,256,stride_list[0],inplace=False)
        self.block3=Block(256,728,stride_list[1])

        rate = 16//os
        self.block4=Block(728,728,1,atrous=rate)
        self.block5=Block(728,728,1,atrous=rate)
        self.block6=Block(728,728,1,atrous=rate)
        self.block7=Block(728,728,1,atrous=rate)

        self.block8=Block(728,728,1,atrous=rate)
        self.block9=Block(728,728,1,atrous=rate)
        self.block10=Block(728,728,1,atrous=rate)
        self.block11=Block(728,728,1,atrous=rate)

        self.block12=Block(728,728,1,atrous=rate)
        self.block13=Block(728,728,1,atrous=rate)
        self.block14=Block(728,728,1,atrous=rate)
        self.block15=Block(728,728,1,atrous=rate)

        self.block16=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        self.block17=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        self.block18=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        self.block19=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        
        self.block20=Block(728,1024,stride_list[2],atrous=rate,grow_first=False)
        #self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1*rate,dilation=rate,activate_first=False)
        # self.bn3 = SynchronizedBatchNorm2d(1536, momentum=bn_mom)

        self.conv4 = SeparableConv2d(1536,1536,3,1,1*rate,dilation=rate,activate_first=False)
        # self.bn4 = SynchronizedBatchNorm2d(1536, momentum=bn_mom)

        #do relu here
        self.conv5 = SeparableConv2d(1536,2048,3,1,1*rate,dilation=rate,activate_first=False)
        # self.bn5 = SynchronizedBatchNorm2d(2048, momentum=bn_mom)
        self.layers = []

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------

    def forward(self, input):
        self.layers = []
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        #self.layers.append(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        self.layers.append(self.block2.hook_layer)
        x = self.block3(x)
        # self.layers.append(self.block3.hook_layer)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)       
        # self.layers.append(self.block20.hook_layer)

        x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        
        x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.relu(x)
        self.layers.append(x)

        return x

    def get_layers(self):
        return self.layers

def xception(pretrained=True, os=16):
    model = Xception(os=os)
    if pretrained:
        old_dict = torch.load(model_urls['xception'])
        # old_dict = model_zoo.load_url(model_urls['xception'])
        # for name, weights in old_dict.items():
        #     if 'pointwise' in name:
        #         old_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model_dict = model.state_dict()
        old_dict = {k: v for k,v in old_dict.items() if ('itr' not in k and 'tmp' not in k and 'track' not in k)}
        model_dict.update(old_dict)
        
        model.load_state_dict(model_dict) 

    return model

    










    '''backbone folder'''

    '''_init_.py was empty, go the next file'''





        # coding: utf-8
import torch
from torch.nn import init
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
Refence: https://arxiv.org/pdf/1606.02147.pdf

Code is written by Iroh Cao
'''

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class InitialBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InitialBlock, self).__init__()
        self.input_channel = in_ch
        self.conv_channel = out_ch - in_ch

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch - in_ch, kernel_size = 3, stride = 2, padding=1),
            nn.BatchNorm2d(out_ch - in_ch),
            nn.PReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        conv_branch = self.conv(x)
        maxp_branch = self.maxpool(x)
        return torch.cat([conv_branch, maxp_branch], 1)

class BottleneckModule(nn.Module):
    def __init__(self, in_ch, out_ch, module_type, padding = 1, dilated = 0, asymmetric = 5, dropout_prob = 0):
        super(BottleneckModule, self).__init__()
        self.input_channel = in_ch
        self.activate = nn.PReLU()

        self.module_type = module_type
        if self.module_type == 'downsampling':
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 2, stride = 2),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_prob)
            )
        elif self.module_type == 'upsampling':
            self.maxunpool = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)    # Use upsample instead of maxunpooling
            )
            
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_prob)
            )
        elif self.module_type == 'regular':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_prob)
            )
        elif self.module_type == 'asymmetric':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, (asymmetric, 1), stride=1, padding=(padding, 0)),
                nn.Conv2d(out_ch, out_ch, (1, asymmetric), stride=1, padding=(0, padding)),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_prob)
            )
        elif self.module_type == 'dilated':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=padding, dilation=dilated),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_prob)
            )
        else:
            raise("Module Type error")

    def forward(self, x):
        if self.module_type == 'downsampling':
            conv_branch = self.conv(x)
            maxp_branch = self.maxpool(x)
            bs, conv_ch, h, w = conv_branch.size()
            maxp_ch = maxp_branch.size()[1]
            padding = torch.zeros(bs, conv_ch - maxp_ch, h, w).to(DEVICE)

            maxp_branch = torch.cat([maxp_branch, padding], 1).to(DEVICE)
            output = maxp_branch + conv_branch
        elif self.module_type == 'upsampling':
            conv_branch = self.conv(x)
            maxunp_branch = self.maxunpool(x)
            output = maxunp_branch + conv_branch
        else:
            output = self.conv(x) + x
        
        return self.activate(output)

class ENet_Encoder(nn.Module):
    
    def __init__(self, in_ch=3, dropout_prob=0):
        super(ENet_Encoder, self).__init__()

        # Encoder

        self.initial_block = InitialBlock(in_ch, 16)

        self.bottleneck1_0 = BottleneckModule(16, 64, module_type = 'downsampling', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck1_1 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck1_2 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck1_3 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck1_4 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)

        self.bottleneck2_0 = BottleneckModule(64, 128, module_type = 'downsampling', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck2_1 = BottleneckModule(128, 128, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck2_2 = BottleneckModule(128, 128, module_type = 'dilated', padding = 2, dilated = 2, dropout_prob = dropout_prob)
        self.bottleneck2_3 = BottleneckModule(128, 128, module_type = 'asymmetric', padding = 2, asymmetric=5, dropout_prob = dropout_prob)
        self.bottleneck2_4 = BottleneckModule(128, 128, module_type = 'dilated', padding = 4, dilated = 4, dropout_prob = dropout_prob)
        self.bottleneck2_5 = BottleneckModule(128, 128, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck2_6 = BottleneckModule(128, 128, module_type = 'dilated', padding = 8, dilated = 8, dropout_prob = dropout_prob)
        self.bottleneck2_7 = BottleneckModule(128, 128, module_type = 'asymmetric', padding = 2, asymmetric=5, dropout_prob = dropout_prob)
        self.bottleneck2_8 = BottleneckModule(128, 128, module_type = 'dilated', padding = 16, dilated = 16, dropout_prob = dropout_prob)

        self.bottleneck3_0 = BottleneckModule(128, 128, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck3_1 = BottleneckModule(128, 128, module_type = 'dilated', padding = 2, dilated = 2, dropout_prob = dropout_prob)
        self.bottleneck3_2 = BottleneckModule(128, 128, module_type = 'asymmetric', padding = 2, asymmetric=5, dropout_prob = dropout_prob)
        self.bottleneck3_3 = BottleneckModule(128, 128, module_type = 'dilated', padding = 4, dilated = 4, dropout_prob = dropout_prob)
        self.bottleneck3_4 = BottleneckModule(128, 128, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck3_5 = BottleneckModule(128, 128, module_type = 'dilated', padding = 8, dilated = 8, dropout_prob = dropout_prob)
        self.bottleneck3_6 = BottleneckModule(128, 128, module_type = 'asymmetric', padding = 2, asymmetric=5, dropout_prob = dropout_prob)
        self.bottleneck3_7 = BottleneckModule(128, 128, module_type = 'dilated', padding = 16, dilated = 16, dropout_prob = dropout_prob)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weights_init_kaiming(m)
            elif isinstance(m, nn.BatchNorm2d):
                weights_init_kaiming(m)
    
    def forward(self, x):
        x = self.initial_block(x)

        x = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)

        x = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        x = self.bottleneck2_8(x)

        x = self.bottleneck3_0(x)
        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_5(x)
        x = self.bottleneck3_6(x)
        x = self.bottleneck3_7(x)

        return x


class ENet_Decoder(nn.Module):
    
    def __init__(self, out_ch=1, dropout_prob=0):
        super(ENet_Decoder, self).__init__()


        self.bottleneck4_0 = BottleneckModule(128, 64, module_type = 'upsampling', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck4_1 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck4_2 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)

        self.bottleneck5_0 = BottleneckModule(64, 16, module_type = 'upsampling', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck5_1 = BottleneckModule(16, 16, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)

        self.fullconv = nn.ConvTranspose2d(16, out_ch, kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weights_init_kaiming(m)
            elif isinstance(m, nn.BatchNorm2d):
                weights_init_kaiming(m)
    
    def forward(self, x):

        x = self.bottleneck4_0(x)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)

        x = self.bottleneck5_0(x)
        x = self.bottleneck5_1(x)

        x = self.fullconv(x)

        return x


class ENet(nn.Module):
    
    def __init__(self, in_ch=3, out_ch=1):
        super(ENet, self).__init__()

        # Encoder

        self.encoder = ENet_Encoder(in_ch)

        # self.initial_block = InitialBlock(in_ch, 16)

        # self.bottleneck1_0 = BottleneckModule(16, 64, module_type = 'downsampling', padding = 1, dropout_prob = 0.1)
        # self.bottleneck1_1 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = 0.1)
        # self.bottleneck1_2 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = 0.1)
        # self.bottleneck1_3 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = 0.1)
        # self.bottleneck1_4 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = 0.1)

        # self.bottleneck2_0 = BottleneckModule(64, 128, module_type = 'downsampling', padding = 1, dropout_prob = 0.1)
        # self.bottleneck2_1 = BottleneckModule(128, 128, module_type = 'regular', padding = 1, dropout_prob = 0.1)
        # self.bottleneck2_2 = BottleneckModule(128, 128, module_type = 'dilated', padding = 2, dilated = 2, dropout_prob = 0.1)
        # self.bottleneck2_3 = BottleneckModule(128, 128, module_type = 'asymmetric', padding = 2, asymmetric=5, dropout_prob = 0.1)
        # self.bottleneck2_4 = BottleneckModule(128, 128, module_type = 'dilated', padding = 4, dilated = 4, dropout_prob = 0.1)
        # self.bottleneck2_5 = BottleneckModule(128, 128, module_type = 'regular', padding = 1, dropout_prob = 0.1)
        # self.bottleneck2_6 = BottleneckModule(128, 128, module_type = 'dilated', padding = 8, dilated = 8, dropout_prob = 0.1)
        # self.bottleneck2_7 = BottleneckModule(128, 128, module_type = 'asymmetric', padding = 2, asymmetric=5, dropout_prob = 0.1)
        # self.bottleneck2_8 = BottleneckModule(128, 128, module_type = 'dilated', padding = 16, dilated = 16, dropout_prob = 0.1)

        # self.bottleneck3_0 = BottleneckModule(128, 128, module_type = 'regular', padding = 1, dropout_prob = 0.1)
        # self.bottleneck3_1 = BottleneckModule(128, 128, module_type = 'dilated', padding = 2, dilated = 2, dropout_prob = 0.1)
        # self.bottleneck3_2 = BottleneckModule(128, 128, module_type = 'asymmetric', padding = 2, asymmetric=5, dropout_prob = 0.1)
        # self.bottleneck3_3 = BottleneckModule(128, 128, module_type = 'dilated', padding = 4, dilated = 4, dropout_prob = 0.1)
        # self.bottleneck3_4 = BottleneckModule(128, 128, module_type = 'regular', padding = 1, dropout_prob = 0.1)
        # self.bottleneck3_5 = BottleneckModule(128, 128, module_type = 'dilated', padding = 8, dilated = 8, dropout_prob = 0.1)
        # self.bottleneck3_6 = BottleneckModule(128, 128, module_type = 'asymmetric', padding = 2, asymmetric=5, dropout_prob = 0.1)
        # self.bottleneck3_7 = BottleneckModule(128, 128, module_type = 'dilated', padding = 16, dilated = 16, dropout_prob = 0.1)

        # Decoder

        self.decoder = ENet_Decoder(out_ch)

        # self.bottleneck4_0 = BottleneckModule(128, 64, module_type = 'upsampling', padding = 1, dropout_prob = 0.1)
        # self.bottleneck4_1 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = 0.1)
        # self.bottleneck4_2 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = 0.1)

        # self.bottleneck5_0 = BottleneckModule(64, 16, module_type = 'upsampling', padding = 1, dropout_prob = 0.1)
        # self.bottleneck5_1 = BottleneckModule(16, 16, module_type = 'regular', padding = 1, dropout_prob = 0.1)

        # self.fullconv = nn.ConvTranspose2d(16, out_ch, kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weights_init_kaiming(m)
            elif isinstance(m, nn.BatchNorm2d):
                weights_init_kaiming(m)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)


        # x = self.initial_block(x)

        # x = self.bottleneck1_0(x)
        # x = self.bottleneck1_1(x)
        # x = self.bottleneck1_2(x)
        # x = self.bottleneck1_3(x)
        # x = self.bottleneck1_4(x)

        # x = self.bottleneck2_0(x)
        # x = self.bottleneck2_1(x)
        # x = self.bottleneck2_2(x)
        # x = self.bottleneck2_3(x)
        # x = self.bottleneck2_4(x)
        # x = self.bottleneck2_5(x)
        # x = self.bottleneck2_6(x)
        # x = self.bottleneck2_7(x)
        # x = self.bottleneck2_8(x)

        # x = self.bottleneck3_0(x)
        # x = self.bottleneck3_1(x)
        # x = self.bottleneck3_2(x)
        # x = self.bottleneck3_3(x)
        # x = self.bottleneck3_4(x)
        # x = self.bottleneck3_5(x)
        # x = self.bottleneck3_6(x)
        # x = self.bottleneck3_7(x)

        # x = self.bottleneck4_0(x)
        # x = self.bottleneck4_1(x)
        # x = self.bottleneck4_2(x)

        # x = self.bottleneck5_0(x)
        # x = self.bottleneck5_1(x)

        # x = self.fullconv(x)

        return x

#########################################################################
'''
============================================================================
Test the module type.
============================================================================

'''

if __name__ == "__main__":
    input_var = Variable(torch.randn(5, 3, 512, 512))
    # model = BottleneckModule(128, 64, module_type = 'upsampling', padding = 2, dilated = 2, asymmetric = 5, dropout_prob = 0.1)
    model = ENet(3, 2)
    print(model)
    output = model(input_var)
    # print(output)
    print(output.shape)












# coding: utf-8
"""
U-Net model encoder and decoder
"""

import torch
from torch.nn import init
from torch import nn

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class UNet_Encoder(nn.Module):
    def __init__(self, in_ch):
        super(UNet_Encoder, self).__init__()
        self.n_channels = in_ch
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weights_init_kaiming(m)
            elif isinstance(m, nn.BatchNorm2d):
                weights_init_kaiming(m)
        
    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)     
        return c1, c2, c3, c4, c5


class UNet_Decoder(nn.Module):
    def __init__(self, out_ch):
        super(UNet_Decoder, self).__init__()
        self.n_classes = out_ch
        
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weights_init_kaiming(m)
            elif isinstance(m, nn.BatchNorm2d):
                weights_init_kaiming(m)
        
    def forward(self, c1, c2, c3, c4, c5):
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9, c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        
        return c10
    















    '''architecture folder'''



        '''_init_.py was empty, go to the next file'''





        # coding: utf-8
"""
LaneNet model
https://arxiv.org/pdf/1807.01726.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pit.LaneNet.architecture.backbone.UNet import UNet_Encoder, UNet_Decoder
from pit.LaneNet.architecture.backbone.ENet import ENet_Encoder, ENet_Decoder
from pit.LaneNet.architecture.backbone.deeplabv3_plus.deeplabv3plus import Deeplabv3plus_Encoder, Deeplabv3plus_Decoder

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LaneNet(nn.Module):
    def __init__(self, in_ch = 3, arch="ENet"):
        super(LaneNet, self).__init__()
        # no of instances for segmentation
        self.no_of_instances = 3  # if you want to output RGB instance map, it should be 3.
        print("Use {} as backbone".format(arch))
        self._arch = arch
        if self._arch == 'UNet':
            self._encoder = UNet_Encoder(in_ch)
            self._encoder.to(DEVICE)

            self._decoder_binary = UNet_Decoder(2)
            self._decoder_instance = UNet_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'ENet':
            self._encoder = ENet_Encoder(in_ch)
            self._encoder.to(DEVICE)

            self._decoder_binary = ENet_Decoder(2)
            self._decoder_instance = ENet_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'DeepLabv3+':
            self._encoder = Deeplabv3plus_Encoder()
            self._encoder.to(DEVICE)

            self._decoder_binary = Deeplabv3plus_Decoder(2)
            self._decoder_instance = Deeplabv3plus_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        else:
            raise("Please select right model.")

        self.relu = nn.ReLU().to(DEVICE)
        self.sigmoid = nn.Sigmoid().to(DEVICE)

    def forward(self, input_tensor):
        if self._arch == 'UNet':
            c1, c2, c3, c4, c5 = self._encoder(input_tensor)
            binary = self._decoder_binary(c1, c2, c3, c4, c5)
            instance = self._decoder_instance(c1, c2, c3, c4, c5)
        elif self._arch == 'ENet':
            c = self._encoder(input_tensor)
            binary = self._decoder_binary(c)
            instance = self._decoder_instance(c)
        elif self._arch == 'DeepLabv3+':
            c1, c2 = self._encoder(input_tensor)
            binary = self._decoder_binary(c1, c2)
            instance = self._decoder_instance(c1, c2)
        else:
            raise("Please select right model.")

        binary_seg_ret = torch.argmax(F.softmax(binary, dim=1), dim=1, keepdim=True)

        pix_embedding = self.sigmoid(instance)

        return {
            'instance_seg_logits': pix_embedding,
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': binary
        }














MIT License

Copyright (c) 2021 Iroh Cao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.













import numpy as np
import cv2
import torch 
import os
import torchvision.transforms as transforms
import tensorrt as trt
import requests
from tqdm import tqdm
from pit.LaneNet.utils import NP_TO_TORCH_DICT
import time 

class LaneNet():

    def __init__(
        self,
        modelPath = None,
        imageHeight = 480,
        imageWidth = 640,
        rowUpperBound = 228
        ):

        self.defaultPath = os.path.normpath(os.path.join(
                              os.path.dirname(__file__), 
                              '../../../resources/pretrained_models/lanenet.engine'))
        self.modelPath = self.__check_path(modelPath)
        self.rowUpperBound = rowUpperBound
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.imgTransforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # use the same nomalization as the training set
            ])
        self.__allocate_buffers()
        self.engine = self.__load_engine()
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream(device='cuda:0')
        print('LaneNet loaded')
    
    def pre_process(self, inputImg):
        self.imgClone = inputImg.copy()
        self.imgTensor = self.imgTransforms(self.imgClone[self.rowUpperBound:,:,:])
        return self.imgTensor
    
    def predict(self, inputImg):
        if not inputImg.dtype == torch.float32:
            raise SystemExit('input image data type error, need to be torch.float32')
        self.inputBuffer[:] = inputImg.flatten()[:]
        bindings = [self.inputBuffer.data_ptr()] +\
                   [self.binaryLogitsBuffer.data_ptr()] +\
                   [self.binaryBuffer.data_ptr()] +\
                   [self.instanceBuffer.data_ptr()]
        start = time.time()
        self.context.execute_async_v2(bindings=bindings, 
                                      stream_handle=self.stream.cuda_stream)
        end=time.time()
        self.stream.synchronize()
        self.FPS = 1/(end-start)
        self.binaryPred = (self.binaryBuffer.cpu().numpy().reshape((256,512))*255).astype(np.uint8)
        self.instancePred = self.instanceBuffer.cpu().numpy().reshape((3,256,512)).transpose((1, 2, 0))
        return (self.binaryPred,self.instancePred)
    
    def render(self,showFPS = True):
        binary3d = np.dstack((self.binaryPred,self.binaryPred,self.binaryPred))
        # instanceVisual = (self.instancePred*255).clip(max=255).astype(np.uint8)
        instanceVisual = (self.instancePred*255).astype(np.uint8)
        lanes= cv2.bitwise_and(instanceVisual,binary3d)
        overlaid=cv2.addWeighted(lanes,
                                 1,
                                 self.imgTensor.numpy().transpose((1, 2, 0))[:,:,[2,1,0]],
                                 1,
                                 0,
                                 dtype=cv2.CV_32F)
        resized=cv2.resize(overlaid, 
                           (self.imageWidth, self.imageHeight - self.rowUpperBound), 
                           interpolation = cv2.INTER_LINEAR)
        annotatedImg = self.imgClone.copy()
        annotatedImg[self.rowUpperBound:,:,:]=(resized*255).clip(max=255).astype(np.uint8)[:,:,[2,1,0]]
        if showFPS:
            cv2.putText(annotatedImg, 'FPS: '+str(round(self.FPS)), 
                        (565,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,255,255), 2)
        return annotatedImg
    
    @staticmethod
    def convert_to_trt(path):
        print('Converting to teneorRT engine')

        #convert to onnx format
        model = torch.load(path)
        model.eval()
        dummy_input = torch.rand((1,3,256,512)).cuda()
        onnx_path=os.path.join(os.path.split(path)[0],'lanenet.onnx')
        torch.onnx.export(model, dummy_input, onnx_path)
        
        #convert to tensorrt engine
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        EXPLICIT_BATCH = []
        if trt.__version__[0] >= '7':
            EXPLICIT_BATCH.append(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(*EXPLICIT_BATCH)
        parser= trt.OnnxParser(network, logger)

        success = parser.parse_from_file(onnx_path)
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        if not success:
            print('Parser read failed')

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 22) # 1 MiB
        config.set_flag(trt.BuilderFlag.FP16)
        serialized_engine = builder.build_serialized_network(network, config)
        enginePath=os.path.join(os.path.split(path)[0],'lanenet.engine')
        with open(enginePath, 'wb') as f:
            f.write(serialized_engine)
        return enginePath

    def __check_path(self,modelPath):
    
        if modelPath:
            enginePath = os.path.splitext(modelPath)[0]+'.engine'
            if os.path.exists(enginePath):
                return enginePath
            if os.path.splitext(modelPath)[1] != '.engine':
                try:
                    enginePath = self.convert_to_trt(modelPath)
                except:
                    errorMsg = modelPath + ' does not exist, or is in unsupported format, please ensure the model is a .pt file.'
                    raise SystemExit(errorMsg) 
            else:
                enginePath = modelPath
        else: 
            if not os.path.exists(self.defaultPath):
                self.__download_model()
                ptPath = os.path.splitext(self.defaultPath)[0]+'.pt'
                enginePath = self.convert_to_trt(ptPath)
            else:
                enginePath = self.defaultPath
        return enginePath

    def __load_engine(self):
        self.logger = trt.Logger()
        if not os.path.isfile(self.modelPath):
            raise SystemExit('ERROR: file (%s) not found!' % self.modelPath)
        with open(self.modelPath,'rb') as f, trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def __allocate_buffers(self):
        self.inputBuffer          = torch.empty((512*256*3),
                                             dtype=torch.float32,
                                             device='cuda:0')
        self.binaryLogitsBuffer         = torch.empty((512*256*2),
                                             dtype=torch.float32,
                                             device='cuda:0')
        self.binaryBuffer   = torch.empty((512*256*1),
                                             dtype=torch.int32,
                                             device='cuda:0')
        self.instanceBuffer       = torch.empty((512*256*3),
                                             dtype=torch.float32,
                                             device='cuda:0')
        
    def __download_model(self):
        url = 'https://quanserinc.box.com/shared/static/c19pjultyikcgzlbzu6vs8tu5vuqhl2n.pt'
        filepath = os.path.splitext(self.defaultPath)[0]+'.pt'
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        print('Downloading lanenet.pt')
        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(filepath, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Could not download file")













import numpy as np
import torch

NP_TO_TORCH_DICT = {
        np.uint8      : torch.uint8,
        np.int8       : torch.int8,
        np.int16      : torch.int16,
        np.int32      : torch.int32,
        np.int64      : torch.int64,
        np.float16    : torch.float16,
        np.float32    : torch.float32,
        np.float64    : torch.float64,
        np.complex64  : torch.complex64,
        np.complex128 : torch.complex128
    }













        '''YOLO folder is below:'''

        




        import numpy as np
import cv2
from ultralytics import YOLO
import torch 
import os
from pit.YOLO.utils import TrafficLight,Obstacle,MASK_COLORS_RGB
import requests
from tqdm import tqdm

class YOLOv8():

    def __init__(
        self,
        imageWidth = 640,
        imageHeight = 480,
        modelPath = None,
        ):

        self.defaultPath = os.path.normpath(os.path.join(
                        os.path.dirname(__file__), 
                        '../../../resources/pretrained_models/yolov8s-seg.engine'))
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.modelPath = self.__check_path(modelPath)
        self.img=np.empty((480,640,3),dtype=np.uint8)
        self.objectsDetected=None
        self.processedResults=[]
        self._calc_distence = False
        self.net=YOLO(self.modelPath,task='segment')
        print('YOLOv8 model loaded')   

    def pre_process(self,inputImg):
        inputImgClone=inputImg.copy()
        if inputImgClone.shape[:2] != (self.imageHeight,self.imageWidth):
            inputImgClone=cv2.resize(inputImgClone,
                                     (self.imageWidth,self.imageHeight))
        self.img[:,:,:]=inputImgClone[:,:,:]
        return self.img
    
    def predict(self, inputImg, classes = [2,9,11], confidence = 0.3, verbose = False, half = False):
        self.predictions = self.net.predict(inputImg,
                                            verbose = verbose,
                                            imgsz = (self.imageHeight,
                                                     self.imageWidth),
                                            classes = classes,
                                            conf = confidence,
                                            half = half
                                            )
        self.objectsDetected=self.predictions[0].boxes.cls.cpu().numpy()
        self.FPS=1000/self.predictions[0].speed['inference']
        return self.predictions[0]
    
    def render(self):
        
        annotatedImg = self.predictions[0].plot()

        return annotatedImg

    def post_processing(self,alignedDepth=None,clippingDistance=5):
        '''
        a depth image aligned to the rgb input is needed for computing the 
        distnace of the detected obstacle
        '''
        self.processedResults = []
        if len (self.objectsDetected) == 0:
            return self.processedResults
        self.bounding = self.predictions[0].boxes.xyxy.cpu().numpy().astype(int)
        if alignedDepth is not None:
            depth3D = np.dstack((alignedDepth,alignedDepth,alignedDepth))
            bgRemoved = np.where((depth3D > clippingDistance)| 
                                 (depth3D <= 0), 0, depth3D)
            self._calc_distence = True
            self.depthTensor=torch.as_tensor(bgRemoved,device="cuda:0")
        for i in range(len(self.objectsDetected)):
            if self.objectsDetected[i]==9:
                trafficBox = self.bounding[i]
                traficLightColor = self.check_traffic_light(trafficBox,self.img)
                result=TrafficLight(color=traficLightColor)
                result.name+=(' ('+traficLightColor+')')
            else:
                name=self.predictions[0].names[self.objectsDetected[i]]
                result=Obstacle(name=name)
            if alignedDepth is not None:
                mask=self.predictions[0].masks.data.cuda()[i]
                distance=self.check_distance(mask,self.depthTensor[:,:,:1])
                result.distance=distance.cpu().numpy().round(3)
            points=self.predictions[0].boxes.xyxy.cpu()[i]
            x=int(points.numpy()[0])
            y=int(points.numpy()[1])
            result.x=x
            result.y=y
            self.processedResults.append(result)
        return self.processedResults

    def post_process_render(self, showFPS = False, bbox_thickness = 4):
        if not self.processedResults:
            return self.img
        colors=[]
        masks = self.predictions[0].masks.data.cuda()
        boxes = self.predictions[0].boxes.xyxy.cpu().numpy().astype(int)
        imgClone=self.img.copy()
        for i in range(len(self.objectsDetected)):
            colors.append(MASK_COLORS_RGB[self.objectsDetected[i].astype(int)])
            name=self.processedResults[i].name
            x=self.processedResults[i].x
            y=self.processedResults[i].y
            distance=self.processedResults[i].distance
            cv2.rectangle(imgClone,(boxes[i,:2]),(boxes[i,2:4]),colors[i],bbox_thickness)
            cv2.putText(imgClone, name, 
                        (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        colors[i], 2)
            if self._calc_distence:
                cv2.putText(imgClone,str(distance) + " m",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            colors[i], 2)
        if showFPS:
            cv2.putText(imgClone, 'FPS: '+str(round(self.FPS)), 
                        (565,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,255,255), 2)
        imgTensor=torch.from_numpy(imgClone).to("cuda:0")
        imgMask=self.mask_color(masks, imgTensor,colors)
        return imgMask

    @staticmethod
    def convert_to_trt(path,
                       imageWidth = 640,
                       imageHeight = 480,
                       half = True,
                       dynamic = False,
                       batch = 1,
                       int8 = False,
                       simplify = True
                       ):
        print('Converting to teneorRT engine')
        model = YOLO(path)
        model.export(format="engine",
             imgsz=(imageHeight,imageWidth),
             half=half,
             dynamic=dynamic,
             batch=batch,
             int8=int8,
             simplify=simplify)
        enginePath = os.path.splitext(path)[0]+'.engine'
        return enginePath
        
    @staticmethod
    def mask_color(masks, im_gpu,colors, alpha=0.5):
        colors = torch.tensor(colors, device="cuda:0", dtype=torch.float32) / 255.0 
        colors = colors[:, None, None]
        masks = masks.unsqueeze(3)
        masks_color = masks * (colors * alpha)
        inv_alpha_masks = (1 - masks * alpha).cumprod(0) 
        mcs = masks_color.max(dim=0).values  
        im_gpu = im_gpu/255
        im_gpu = im_gpu * inv_alpha_masks[-1] + mcs 
        im_mask = im_gpu * 255
        im_mask_np = im_mask.squeeze().byte().cpu().numpy()
        return im_mask_np

    @staticmethod
    def check_traffic_light(traffic_box,im_cpu):
        mask = np.zeros((480,640),dtype='uint8')
        x1,y1,x2,y2=(traffic_box[0], traffic_box[1], traffic_box[2], traffic_box[3])
        d = 0.3*(x2-x1)
        R_center=(int(x1/2+x2/2),int(3*y1/4+y2/4))
        Y_center=(int(x1/2+x2/2),int(y1/2+y2/2))
        G_center=(int(x1/2+x2/2),int(y1/4+3*y2/4))
        maskR=cv2.circle(mask.copy(),R_center,int(d/2),1,-1)
        maskY=cv2.circle(mask.copy(),Y_center,int(d/2),1,-1)
        maskG=cv2.circle(mask.copy(),G_center,int(d/2),1,-1)
        maskR_gpu=torch.tensor(maskR,device="cuda:0").unsqueeze(2)
        maskY_gpu=torch.tensor(maskY,device="cuda:0").unsqueeze(2)
        maskG_gpu=torch.tensor(maskG,device="cuda:0").unsqueeze(2)
        im_hsv=cv2.cvtColor(im_cpu, cv2.COLOR_RGB2HSV)
        im_hsv_gpu = torch.tensor(im_hsv,device="cuda:0")
        masked_red = im_hsv_gpu*maskR_gpu
        masked_yellow = im_hsv_gpu*maskY_gpu
        masked_green = im_hsv_gpu*maskG_gpu
        value_R=torch.sum(masked_red[:,:,2])/torch.count_nonzero(masked_red[:,:,2])
        value_Y=torch.sum(masked_yellow[:,:,2])/torch.count_nonzero(masked_yellow[:,:,2])
        value_G=torch.sum(masked_green[:,:,2])/torch.count_nonzero(masked_green[:,:,2])
        mean = (value_R+value_Y+value_G)/3
        threshold_perc=0.25
        min= torch.min(torch.tensor([value_R,value_Y,value_G]))
        max= torch.max(torch.tensor([value_R,value_Y,value_G]))
        if (max-min)<30:
            return 'idle'
        threshold=(max-min)*threshold_perc
        # print('red',value_R,'yellow',value_Y,'green',value_G,mean,threshold)
        redOn=(value_R>mean) and (value_R-mean)>threshold
        yellowOn=(value_Y>mean) and (value_Y-mean)>threshold
        greenOn=(value_G>mean) and (value_G-mean)>threshold
        traffic_light_status=[redOn.cpu().numpy(),yellowOn.cpu().numpy(),greenOn.cpu().numpy()]
        colors=['red','yellow','green']
        traffic_light_color=''
        for i in range(len(traffic_light_status)):
            if traffic_light_status[i]:
                traffic_light_color+=colors[i]
                traffic_light_color+=' '
        return traffic_light_color

    @staticmethod
    def check_distance(mask,depth_gpu):
        mask=mask.unsqueeze(2)
        isolated_depth = mask*depth_gpu
        # distance = torch.sum(isolated_depth)/torch.count_nonzero(isolated_depth)
        distance = torch.median(isolated_depth[isolated_depth.nonzero(as_tuple=True)])
        return distance
    
    @staticmethod
    def reshape_for_matlab_server(frame):
        frame=frame.copy()[:,:,[2,1,0]]
        flatten = frame.flatten(order='F').copy() 
        return flatten.reshape(frame.shape,order='C')
    
    def __download_model(self):
        url = 'https://quanserinc.box.com/shared/static/ce0gxomeg4b12wlcch9cmlh0376nditf.pt'
        filepath = os.path.splitext(self.defaultPath)[0]+'.pt'
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        print('Downloading yolov8s-seg.pt')
        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(filepath, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Could not download file")

    def __check_path(self, modelPath):
        
        if modelPath:

            enginePath = os.path.splitext(modelPath)[0]+'.engine'
            if os.path.exists(enginePath):
                return enginePath
            
            if os.path.splitext(modelPath)[1] != '.engine':
                try:
                    enginePath = self.convert_to_trt(path = modelPath,
                                                     imageWidth = self.imageWidth,
                                                     imageHeight = self.imageHeight)
                except:
                    errorMsg = modelPath + ' does not exist, or is in unsupported formats.'
                    raise SystemExit(errorMsg) 
            else:
                enginePath = modelPath
        else: 
            if not os.path.exists(self.defaultPath):
                ptPath = os.path.splitext(self.defaultPath)[0]+'.pt'
                if not os.path.exists(ptPath):
                    self.__download_model()
                enginePath = self.convert_to_trt(path = ptPath,
                                                 imageWidth = self.imageWidth,
                                                 imageHeight = self.imageHeight)
            else:
                enginePath = self.defaultPath
        return enginePath
        # if modelPath:
        #     self.modelPath=modelPath
        # else: 
        #     self.modelPath=os.path.normpath(os.path.join(
        #     os.path.dirname(__file__), '../../../resources/pretrained_models/yolov8s-seg.engine'))
        # if not os.path.exists(self.modelPath):
        #     self.convert_to_trt(imageWidth=self.imageWidth,imageHeight=self.imageHeight)















        import numpy as np
import time
from quanser.common import Timeout
from pal.utilities.stream import BasicStream
import os


#top 80 colors from xkcd color survey https://xkcd.com/color/rgb/
MASK_COLORS_HEX=[
    "7e1e9c",
    "15b01a",
    "0343df",
    "ff81c0",
    "653700",
    "e50000",
    "95d0fc",
    "029386",
    "f97306",
    "96f97b",
    "c20078",
    "ffff14",
    "75bbfd",
    "929591",
    "89fe05",
    "bf77f6",
    "9a0eea",
    "033500",
    "06c2ac",
    "c79fef",
    "00035b",
    "d1b26f",
    "00ffff",
    "13eac9",
    "06470c",
    "ae7181",
    "35063e",
    "01ff07",
    "650021",
    "6e750e",
    "ff796c",
    "e6daa6",
    "0504aa",
    "001146",
    "cea2fd",
    "000000",
    "ff028d",
    "ad8150",
    "c7fdb5",
    "ffb07c",
    "677a04",
    "cb416b",
    "8e82fe",
    "53fca1",
    "aaff32",
    "380282",
    "ceb301",
    "ffd1df",
    "cf6275",
    "0165fc",
    "0cff0c",
    "c04e01",
    "04d8b2",
    "01153e",
    "3f9b0b",
    "d0fefe",
    "840000",
    "be03fd",
    "c0fb2d",
    "a2cffe",
    "dbb40c",
    "8fff9f",
    "580f41",
    "4b006e",
    "8f1402",
    "014d4e",
    "610023",
    "aaa662",
    "137e6d",
    "7af9ab",
    "02ab2e",
    "9aae07",
    "8eab12",
    "b9a281",
    "341c02",
    "36013f",
    "c1f80a",
    "fe01b1",
    "fdaa48",
    "9ffeb0",
    ]
MASK_COLORS_RGB=[list(int(h[i:i+2], 16) for i in (0, 2, 4)) for h in MASK_COLORS_HEX]

class Obstacle():
    def __init__(self,
                 name = 'QCar',
                 distance = 0,
                 x=0,
                 y=0      
                 ):
      
        self.name = name
        self.distance = distance 
        self.x=x
        self.y=y

class TrafficLight(Obstacle):
    def __init__(self,
                 color = 'idle',
                 distance = 0
                 ):
        super().__init__(name='traffic light', distance=distance)
        self.lightColor = color

class QCar2DepthAligned():
    def __init__(self,ip='localhost',nonBlocking=True,manualStart=False,port='18777'):
        self.depth  = np.empty((480,640,1), dtype = np.float32)
        self.rgb  = np.empty((480,640,3), dtype = np.uint8) 
        if not manualStart:
            self.__initDepthAlign()
        self.uri='tcpip://'+ip+':'+port
        self._timeout = Timeout(seconds=0, nanoseconds=1000000)
        self._handle = BasicStream(uri=self.uri,
                                    agent='C',
                                    receiveBuffer=np.zeros((480,640,4),
                                                           dtype=np.float32),
                                    sendBufferSize=480*640*3,
                                    recvBufferSize=480*640*4*4,
                                    nonBlocking=nonBlocking,
                                    reshapeOrder='F')
        self._sendPacket = np.zeros((480,640,3),dtype=np.uint8)
        self.status_check('', iterations=20)
    
    def __initDepthAlign(self):
        self.__stopDepthAlign()
        depthAlignPath = os.path.normpath(os.path.join(
            os.path.dirname(__file__),
            '../../../resources/applications/QCarDepthAlign/'
                + 'QCar2DepthAlign.rt-linux_qcar2 '
        ))
        os.system(
            'quarc_run -r -t tcpip://localhost:17000 '
            + depthAlignPath + '-uri tcpip://localhost:17003'
        )

        time.sleep(4)
        print('Aligned Depth image is streaming')

    def __stopDepthAlign(self):
        # Quietly stop qcarLidarToGPS if it is already running:
        # the -q flag kills the executable
        # the -Q flag kills quietly (no errors thrown if its not running)
        os.system(
            'quarc_run -t tcpip://localhost:17000 -q -Q '
            + 'QCar2DepthAlign.rt-linux_qcar2'
        )

    def status_check(self, message, iterations=10):
        # blocking method to establish connection to the server stream.
        self._timeout = Timeout(seconds=0, nanoseconds=1000) #1000000
        counter = 0
        while not self._handle.connected:
            self._handle.checkConnection(timeout=self._timeout)
            counter += 1
            if self._handle.connected:
                print(message)
                break
            elif counter >= iterations:
                print('Server error: status check failed.')
                break

    def read(self):
        new = False
        self._timeout = Timeout(seconds=0, nanoseconds=100)
        if self._handle.connected:
            new, bytesReceived = self._handle.receive(timeout=self._timeout, iterations=5)
            # print('read:',new, bytesReceived)
            # if new is True, full packet was received
            if new:
                self.depth[:,:,:] = self._handle.receiveBuffer[:,:,:1]
                self.rgb[:,:,:] = self._handle.receiveBuffer[:,:,[3,2,1]].astype(np.uint8)

        else:
            self.status_check('Reconnected to Server')
        return new
    
    def read_reply(self,annotated_frame):

        # data received flag
        new = False

        # 1 us timeout parameter
        self._timeout = Timeout(seconds=0, nanoseconds=10000000)

        # set remaining packet to send
        self._sendPacket = annotated_frame

        # if connected to driver, send/receive
        if self._handle.connected:
            self._handle.send(self._sendPacket)
            new, bytesReceived = self._handle.receive(timeout=self._timeout, iterations=5)
            # print(new, bytesReceived)
            # if new is True, full packet was received
            if new:
                self.depth = self._handle.receiveBuffer[:,:,:1]
                self.rgb = self._handle.receiveBuffer[:,:,[3,2,1]].astype(np.uint8)

        else:
            self.status_check('Reconnected to QBot Platform Driver.')

        # if new is False, data is stale, else all is good
        return new

    def terminate(self):
        self.__stopDepthAlign()
        self._handle.terminate()




































        '''Below is the qvl folder: '''




        '''_init_ is empty'''



    from qvl.qlabs import CommModularContainer

import math
import struct
import cv2
import numpy as np


######################### MODULAR CONTAINER CLASS #########################

class QLabsActor:
    """ This the base actor class."""

    FCN_UNKNOWN = 0
    """Function ID is not recognized."""
    FCN_REQUEST_PING = 1
    """Request a response from an actor to test if it is present."""
    FCN_RESPONSE_PING = 2
    """Response from an actor to confirming it is present."""
    FCN_REQUEST_WORLD_TRANSFORM = 3
    """Request a world transform from the actor to read its current location, rotation, and scale."""
    FCN_RESPONSE_WORLD_TRANSFORM = 4
    """Response from an actor with its current location, rotation, and scale."""
    FCN_SET_CUSTOM_PROPERTIES = 5
    """Set custom properties of measured mass, ID, and/or property string."""
    FCN_SET_CUSTOM_PROPERTIES_ACK = 6
    """Set custom properties acknowledgment"""
    FCN_REQUEST_CUSTOM_PROPERTIES = 7
    """Request the custom properties of measured mass, ID, and/or property string previously assigned to the actor."""
    FCN_RESPONSE_CUSTOM_PROPERTIES = 8
    """Response containing the custom properties of measured mass, ID, and/or property string previously assigned to the actor."""
    

    actorNumber = None
    """ The current actor number of this class to be addressed. This will be set by spawn methods and cleared by destroy methods. It will not be modified by the destroy all actors.  This can be manually altered at any time to use one object to address multiple actors. """
    _qlabs = None
    _verbose = False
    classID = 0

    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       return

    def _is_actor_number_valid(self):
        if self.actorNumber == None:
            if (self._verbose):
                print('actorNumber object variable None. Use a spawn function to assign an actor or manually assign the actorNumber variable.')

            return False
        else:
            return True

    def destroy(self):
        """Find and destroy a specific actor. This is a blocking operation.

        :return:
            - **numActorsDestroyed** - The number of actors destroyed. -1 if failed.
        :rtype: int32

        """
        if (not self._is_actor_number_valid()):
            return -1

        c = CommModularContainer()

        c.classID = CommModularContainer.ID_GENERIC_ACTOR_SPAWNER
        c.actorNumber = 0
        c.actorFunction = CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_DESTROY_ACTOR
        c.payload = bytearray(struct.pack(">II", self.classID, self.actorNumber))

        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(CommModularContainer.ID_GENERIC_ACTOR_SPAWNER, 0, CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_DESTROY_ACTOR_ACK)
            if (c == None):
                return -1

            if len(c.payload) == 4:
                numActorsDestroyed, = struct.unpack(">I", c.payload[0:4])
                self.actorNumber = None
                return numActorsDestroyed
            else:
                return -1
        else:
            return -1

    def destroy_all_actors_of_class(self):
        """Find and destroy all actors of this class. This is a blocking operation.

        :return:
            - **numActorsDestroyed** - The number of actors destroyed. -1 if failed.
        :rtype: int32

        """

        c = CommModularContainer()

        c.classID = CommModularContainer.ID_GENERIC_ACTOR_SPAWNER
        c.actorNumber = 0
        c.actorFunction = CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_DESTROY_ALL_ACTORS_OF_CLASS
        c.payload = bytearray(struct.pack(">I", self.classID))

        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(CommModularContainer.ID_GENERIC_ACTOR_SPAWNER, 0, CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_DESTROY_ALL_ACTORS_OF_CLASS_ACK)
            if (c == None):
                return -1

            if len(c.payload) == 4:
                numActorsDestroyed, = struct.unpack(">I", c.payload[0:4])
                self.actorNumber = None
                return numActorsDestroyed
            else:
                return -1
        else:
            return -1

    def spawn_id(self, actorNumber, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, waitForConfirmation=True):
        """Spawns a new actor.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used.
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type actorNumber: uint32
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 2 actor number not available or already in use, 3 unknown error, -1 communications error
        :rtype: int32

        """

        c = CommModularContainer()
        c.classID = CommModularContainer.ID_GENERIC_ACTOR_SPAWNER
        c.actorNumber = 0
        c.actorFunction = CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_SPAWN_ID
        c.payload = bytearray(struct.pack(">IIfffffffffI", self.classID, actorNumber, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], scale[0], scale[1], scale[2], configuration))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)


        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):

            if waitForConfirmation:
                c = self._qlabs.wait_for_container(CommModularContainer.ID_GENERIC_ACTOR_SPAWNER, 0, CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_SPAWN_ID_ACK)
                if (c == None):
                    if (self._verbose):
                        print('spawn_id: Communication timeout (classID {}, actorNumber {}).'.format(self.classID, actorNumber))
                    return -1
                if len(c.payload) == 1:
                    status, = struct.unpack(">B", c.payload[0:1])
                    if (status == 0):
                        self.actorNumber = actorNumber

                    elif (self._verbose):
                        if (status == 1):
                            print('spawn_id: Class not available (classID {}, actorNumber {}).'.format(self.classID, actorNumber))
                        elif (status == 2):
                            print('spawn_id: Actor number not available or already in use (classID {}, actorNumber {}).'.format(self.classID, actorNumber))
                        elif (status == -1):
                            print('spawn_id: Communication error (classID {}, actorNumber {}).'.format(self.classID, actorNumber))
                        else:
                            print('spawn_id: Unknown error (classID {}, actorNumber {}).'.format(self.classID, actorNumber))
                    return status
                else:
                    if (self._verbose):
                        print("spawn: Communication error (classID {}, actorNumber {}).".format(self.classID, actorNumber))
                    return -1

            self.actorNumber = actorNumber
            return 0
        else:
            if (self._verbose):
                print('spawn_id: Communication failed (classID {}, actorNumber {}).'.format(self.classID, actorNumber))
            return -1

    def spawn_id_degrees(self, actorNumber, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, waitForConfirmation=True):
        """Spawns a new actor.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used.
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type actorNumber: uint32
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 2 actor number not available or already in use, 3 unknown error, -1 communications error
        :rtype: int32

        """

        return self.spawn_id(actorNumber, location, [rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi], scale, configuration, waitForConfirmation)

    def spawn(self, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, waitForConfirmation=True):
        """Spawns a new actor with the next available actor number within this class.

        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used.
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred. Note that if this is False, the returned actor number will be invalid.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 3 unknown error, -1 communications error.
            - **actorNumber** - An actor number to use for future references.
        :rtype: int32, int32

        """
        c = CommModularContainer()
        c.classID = CommModularContainer.ID_GENERIC_ACTOR_SPAWNER
        c.actorNumber = 0
        c.actorFunction = CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_SPAWN
        c.payload = bytearray(struct.pack(">IfffffffffI", self.classID, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], scale[0], scale[1], scale[2], configuration))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):

            if waitForConfirmation:
                c = self._qlabs.wait_for_container(CommModularContainer.ID_GENERIC_ACTOR_SPAWNER, 0, CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_SPAWN_RESPONSE)
                if (c == None):
                    if (self._verbose):
                        print('spawn: Communication timeout (classID {}).'.format(self.classID))
                    return -1, -1
                if len(c.payload) == 5:
                    status, actorNumber, = struct.unpack(">BI", c.payload[0:5])
                    if (status == 0):
                        self.actorNumber = actorNumber

                    elif (self._verbose):
                        if (status == 1):
                            print('spawn: Class not available (classID {}).'.format(self.classID))
                        elif (status == -1):
                            print('spawn: Communication error (classID {}).'.format(self.classID))
                        else:
                            print('spawn: Unknown error (classID {}).'.format(self.classID))

                    return status, actorNumber
                else:
                    if (self._verbose):
                        print('spawn: Communication error (classID {}).'.format(self.classID))
                    return -1, -1

            return 0, -1
        else:
            if (self._verbose):
                print('spawn: Communication failed (classID {}).'.format(self.classID))
            return -1, -1

    def spawn_degrees(self, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, waitForConfirmation=True):
        """Spawns a new actor with the next available actor number within this class.

        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in degrees
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used.
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred. Note that if this is False, the returned actor number will be invalid.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 3 unknown error, -1 communications error.
            - **actorNumber** - An actor number to use for future references.
        :rtype: int32, int32

        """
        return self.spawn(location, [rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi], scale, configuration, waitForConfirmation)

    def spawn_id_and_parent_with_relative_transform(self, actorNumber, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, parentClassID=0, parentActorNumber=0, parentComponent=0, waitForConfirmation=True):
        """Spawns a new actor relative to an existing actor and creates a kinematic relationship.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used.
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param parentClassID: See the ID variables in the respective library classes for the class identifier
        :param parentActorNumber: User defined unique identifier for the class actor in QLabs
        :param parentComponent: `0` for the origin of the parent actor, see the parent class for additional reference frame options
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type actorNumber: uint32
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type parentClassID: uint32
        :type parentActorNumber: uint32
        :type parentComponent: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 2 actor number not available or already in use, 3 cannot find the parent actor, 4 unknown error, -1 communications error
        :rtype: int32

        """
        c = CommModularContainer()
        c.classID = CommModularContainer.ID_GENERIC_ACTOR_SPAWNER
        c.actorNumber = 0
        c.actorFunction = CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_SPAWN_AND_PARENT_RELATIVE
        c.payload = bytearray(struct.pack(">IIfffffffffIIII", self.classID, actorNumber, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], scale[0], scale[1], scale[2], configuration, parentClassID, parentActorNumber, parentComponent))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):

            if waitForConfirmation:
                c = self._qlabs.wait_for_container(CommModularContainer.ID_GENERIC_ACTOR_SPAWNER, 0, CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_SPAWN_AND_PARENT_RELATIVE_ACK)
                if (c == None):
                    if (self._verbose):
                        print("spawn_id_and_parent_with_relative_transform (classID {}, actorNumber {}): Communication timeout.".format(self.classID, actorNumber))
                    return -1

                if len(c.payload) == 1:
                    status, = struct.unpack(">B", c.payload[0:1])
                    if (status == 0):
                        self.actorNumber = actorNumber

                    elif (self._verbose):
                        if (status == 1):
                            print('spawn_id_and_parent_with_relative_transform (classID {}, actorNumber {}): Class not available.'.format(self.classID, actorNumber))
                        elif (status == 2):
                            print('spawn_id_and_parent_with_relative_transform (classID {}, actorNumber {}): Actor number not available or already in use.'.format(self.classID, actorNumber))
                        elif (status == 3):
                            print('spawn_id_and_parent_with_relative_transform (classID {}, actorNumber {}): Cannot find parent.'.format(self.classID, actorNumber))
                        elif (status == -1):
                            print('spawn_id_and_parent_with_relative_transform (classID {}, actorNumber {}): Communication error.'.format(self.classID, actorNumber))
                        else:
                            print('spawn_id_and_parent_with_relative_transform (classID {}, actorNumber {}): Unknown error.'.format(self.classID, actorNumber))

                    return status
                else:
                    if (self._verbose):
                        print("spawn_id_and_parent_with_relative_transform (classID {}, actorNumber {}): Communication error.".format(self.classID, actorNumber))
                    return -1

            self.actorNumber = actorNumber
            return 0
        else:
            if (self._verbose):
                print("spawn_id_and_parent_with_relative_transform (classID {}, actorNumber {}): Communication failed.".format(self.classID, actorNumber))
            return -1

    def spawn_id_and_parent_with_relative_transform_degrees(self, actorNumber, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, parentClassID=0, parentActorNumber=0, parentComponent=0, waitForConfirmation=True):
        """Spawns a new actor relative to an existing actor and creates a kinematic relationship.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in degrees
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used.
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param parentClassID: See the ID variables in the respective library classes for the class identifier
        :param parentActorNumber: User defined unique identifier for the class actor in QLabs
        :param parentComponent: `0` for the origin of the parent actor, see the parent class for additional reference frame options
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type actorNumber: uint32
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type parentClassID: uint32
        :type parentActorNumber: uint32
        :type parentComponent: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 2 actor number not available or already in use, 3 cannot find the parent actor, 4 unknown error, -1 communications error
        :rtype: int32

        """
        return self.spawn_id_and_parent_with_relative_transform(actorNumber, location, [rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi], scale, configuration, parentClassID, parentActorNumber, parentComponent, waitForConfirmation)

    def ping(self):
        """Checks if the actor is still present in the environment. Note that if you did not spawn
        the actor with one of the spawn functions, you may need to manually set the actorNumber member variable.


        :return:
            - **status** - `True` if successful, `False` otherwise
        :rtype: boolean
        """

        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.classID
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_REQUEST_PING
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):

            c = self._qlabs.wait_for_container(self.classID, self.actorNumber, self.FCN_RESPONSE_PING)
            if (c == None):
                return False

            if c.payload[0] > 0:
                return True
            else:
                return False
        else:
            if (self._verbose):
                print("ping: Communication failed.")
            return False

    def get_world_transform(self):
        """Get the location, rotation, and scale in world coordinates of the actor.

        :return:
            - **status** - True if successful, False otherwise
            - **location**
            - **rotation**
            - **scale**
        :rtype: boolean, float array[3], float array[3], float array[3]
        """

        if (not self._is_actor_number_valid()):
            return False, [0,0,0], [0,0,0], [0,0,0]

        c = CommModularContainer()
        c.classID = self.classID
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_REQUEST_WORLD_TRANSFORM
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        location = [0,0,0]
        rotation = [0,0,0]
        scale = [0,0,0]

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):

            c = self._qlabs.wait_for_container(self.classID, self.actorNumber, self.FCN_RESPONSE_WORLD_TRANSFORM)
            if (c == None):
                return False, location, rotation, scale

            if len(c.payload) == 36:
                location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], scale[0], scale[1], scale[2], = struct.unpack(">fffffffff", c.payload[0:36])
                return True, location, rotation, scale
            else:
                if (self._verbose):
                    print("get_world_transform: Communication error (classID {}, actorNumber {}).".format(self.classID, self.actorNumber))
                return False, location, rotation, scale
        else:
            if (self._verbose):
                print("get_world_transform: Communication failed (classID {}, actorNumber {}).".format(self.classID, self.actorNumber))
            return False, location, rotation, scale

    def get_world_transform_degrees(self):
        """Get the location, rotation, and scale in world coordinates of the actor.

        :return:
            - **status** - True if successful, False otherwise
            - **location**
            - **rotation**
            - **scale**
        :rtype: boolean, float array[3], float array[3], float array[3]
        """
        success, location, rotation, scale = self.get_world_transform()
        rotation_deg = [rotation[0]/math.pi*180, rotation[1]/math.pi*180, rotation[2]/math.pi*180]

        return  success, location, rotation_deg, scale

    def parent_with_relative_transform(self, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], parentClassID=0, parentActorNumber=0, parentComponent=0, waitForConfirmation=True):
        """Parents one existing actor to another to create a kinematic relationship.

        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used.
        :param parentClassID: See the ID variables in the respective library classes for the class identifier
        :param parentActorNumber: User defined unique identifier for the class actor in QLabs
        :param parentComponent: `0` for the origin of the parent actor, see the parent class for additional reference frame options
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type parentClassID: uint32
        :type parentActorNumber: uint32
        :type parentComponent: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 cannot find this actor, 2 cannot find the parent actor, 3 unknown error, -1 communications error
        :rtype: int32

        """
        c = CommModularContainer()
        c.classID = CommModularContainer.ID_GENERIC_ACTOR_SPAWNER
        c.actorNumber = 0
        c.actorFunction = CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_PARENT_RELATIVE
        c.payload = bytearray(struct.pack(">IIfffffffffIII", self.classID, self.actorNumber, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], scale[0], scale[1], scale[2], parentClassID, parentActorNumber, parentComponent))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):

            if waitForConfirmation:
                c = self._qlabs.wait_for_container(CommModularContainer.ID_GENERIC_ACTOR_SPAWNER, 0, CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_PARENT_RELATIVE_ACK)
                if (c == None):
                    if (self._verbose):
                        print("parent_with_relative_transform (classID {}, actorNumber {}): Communication timeout.".format(self.classID, self.actorNumber))
                    return -1

                if len(c.payload) == 1:
                    status, = struct.unpack(">B", c.payload[0:1])
                    if (status == 0):
                        pass

                    elif (self._verbose):
                        if (status == 1):
                            print('parent_with_relative_transform (classID {}, actorNumber {}): Cannot find this actor.'.format(self.classID, self.actorNumber))
                        elif (status == 2):
                            print('parent_with_relative_transform (classID {}, actorNumber {}): Cannot find parent (classID {}, actorNumber {}).'.format(self.classID, self.actorNumber, parentClassID, parentActorNumber))
                        elif (status == -1):
                            print('parent_with_relative_transform (classID {}, actorNumber {}): Communication error.'.format(self.classID, self.actorNumber))
                        else:
                            print('parent_with_relative_transform (classID {}, actorNumber {}): Unknown error.'.format(self.classID, self.actorNumber))

                    return status
                else:
                    if (self._verbose):
                        print("parent_with_relative_transform (classID {}, actorNumber {}): Communication error.".format(self.classID, self.actorNumber))
                    return -1

            return 0
        else:
            if (self._verbose):
                print("parent_with_relative_transform (classID {}, actorNumber {}): Communication failed.".format(self.classID, self.actorNumber))
            return -1

    def parent_with_relative_transform_degrees(self, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], parentClassID=0, parentActorNumber=0, parentComponent=0, waitForConfirmation=True):
        """Parents one existing actor to another to create a kinematic relationship.

        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in degrees
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used.
        :param parentClassID: (Optional) See the ID variables in the respective library classes for the class identifier
        :param parentActorNumber: (Optional) User defined unique identifier for the class actor in QLabs
        :param parentComponent: (Optional) `0` for the origin of the parent actor, see the parent class for additional reference frame options
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type parentClassID: uint32
        :type parentActorNumber: uint32
        :type parentComponent: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 cannot find this actor, 2 cannot find the parent actor, 3 unknown error, -1 communications error
        :rtype: int32

        """

        return self.parent_with_relative_transform(location, [rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi], scale, parentClassID, parentActorNumber, parentComponent, waitForConfirmation)

    def parent_with_current_world_transform(self, parentClassID=0, parentActorNumber=0, parentComponent=0, waitForConfirmation=True):
        """Parents one existing actor to another to create a kinematic relationship while preserving the current world transform of the child actor.

        :param parentClassID: See the ID variables in the respective library classes for the class identifier
        :param parentActorNumber: User defined unique identifier for the class actor in QLabs
        :param parentComponent: `0` for the origin of the parent actor, see the parent class for additional reference frame options
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type parentClassID: uint32
        :type parentActorNumber: uint32
        :type parentComponent: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 cannot find this actor, 2 cannot find the parent actor, 3 unknown error, -1 communications error
        :rtype: int32

        """
        c = CommModularContainer()
        c.classID = CommModularContainer.ID_GENERIC_ACTOR_SPAWNER
        c.actorNumber = 0
        c.actorFunction = CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_PARENT_CURRENT_WORLD
        c.payload = bytearray(struct.pack(">IIIII", self.classID, self.actorNumber, parentClassID, parentActorNumber, parentComponent))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):

            if waitForConfirmation:
                c = self._qlabs.wait_for_container(CommModularContainer.ID_GENERIC_ACTOR_SPAWNER, 0, CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_PARENT_CURRENT_WORLD_ACK)
                if (c == None):
                    if (self._verbose):
                        print("parent_with_current_world_transform (classID {}, actorNumber {}): Communication timeout.".format(self.classID, self.actorNumber))
                    return -1

                if len(c.payload) == 1:
                    status, = struct.unpack(">B", c.payload[0:1])
                    if (status == 0):
                        pass

                    elif (self._verbose):
                        if (status == 1):
                            print('parent_with_current_world_transform (classID {}, actorNumber {}): Cannot find this actor.'.format(self.classID, self.actorNumber))
                        elif (status == 2):
                            print('parent_with_current_world_transform (classID {}, actorNumber {}): Cannot find parent (classID {}, actorNumber {}).'.format(self.classID, self.actorNumber, parentClassID, parentActorNumber))
                        elif (status == -1):
                            print('parent_with_current_world_transform (classID {}, actorNumber {}): Communication error.'.format(self.classID, self.actorNumber))
                        else:
                            print('parent_with_current_world_transform (classID {}, actorNumber {}): Unknown error.'.format(self.classID, self.actorNumber))

                    return status
                else:
                    if (self._verbose):
                        print("parent_with_current_world_transform (classID {}, actorNumber {}): Communication error.".format(self.classID, self.actorNumber))
                    return -1

            return 0
        else:
            if (self._verbose):
                print("parent_with_current_world_transform (classID {}, actorNumber {}): Communication failed.".format(self.classID, self.actorNumber))
            return -1

    def parent_break(self, waitForConfirmation=True):
        """Breaks any relationship with a parent actor (if it exists) and preserves the current world transform

        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 cannot find this actor, -1 communications error
        :rtype: int32

        """
        c = CommModularContainer()
        c.classID = CommModularContainer.ID_GENERIC_ACTOR_SPAWNER
        c.actorNumber = 0
        c.actorFunction = CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_PARENT_BREAK_WITH_CURRENT_WORLD
        c.payload = bytearray(struct.pack(">II", self.classID, self.actorNumber))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):

            if waitForConfirmation:
                c = self._qlabs.wait_for_container(CommModularContainer.ID_GENERIC_ACTOR_SPAWNER, 0, CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_PARENT_BREAK_WITH_CURRENT_WORLD_ACK)
                if (c == None):
                    if (self._verbose):
                        print("parent_break (classID {}, actorNumber {}): Communication timeout.".format(self.classID, self.actorNumber))
                    return -1

                if len(c.payload) == 1:
                    status, = struct.unpack(">B", c.payload[0:1])
                    if (status == 0):
                        pass

                    elif (self._verbose):
                        if (status == 1):
                            print('parent_break (classID {}, actorNumber {}): Cannot find this actor.'.format(self.classID, self.actorNumber))
                        else:
                            print('parent_break (classID {}, actorNumber {}): Unknown error.'.format(self.classID, self.actorNumber))

                    return status
                else:
                    if (self._verbose):
                        print("parent_break (classID {}, actorNumber {}): Communication error.".format(self.classID, self.actorNumber))
                    return -1

            self.actorNumber = c.actorNumber
            return 0
        else:
            if (self._verbose):
                print("parent_break (classID {}, actorNumber {}): Communication failed.".format(self.classID, self.actorNumber))
            return -1

    def set_custom_properties(self, measuredMass=0, IDTag=0, properties="", waitForConfirmation=True):
        """Assigns custom properties to an actor.

        :param measuredMass: A float value for use with mass sensor instrumented actors. This does not alter the dynamic properties.
        :param IDTag: An integer value for use with IDTag sensor instrumented actors or for custom use.
        :param properties: A string for use with properties sensor instrumented actors. This can contain any string that is available for use to parse user-customized parameters.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type measuredMass: float
        :type IDTag: uint32
        :type properties: string
        :type waitForConfirmation: boolean
        :return:
            - **status** - `True` if successful, `False` otherwise
        :rtype: boolean

        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.classID
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_SET_CUSTOM_PROPERTIES
        c.payload = bytearray(struct.pack(">fII", measuredMass, IDTag, len(properties)))
        c.payload = c.payload + bytearray(properties.encode('utf-8'))

        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):

            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.classID, self.actorNumber, self.FCN_SET_CUSTOM_PROPERTIES_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False

    def get_custom_properties(self):
        """Gets previously assigned custom properties to an actor.

        :return:
            - **status** - `True` if successful, `False` otherwise
            - **measuredMass** - float value
            - **IDTag** - integer value
            - **properties** - UTF-8 string
        :rtype: boolean, float, int32, string

        """

        measuredMass = 0.0
        IDTag = 0
        properties = ""


        if (not self._is_actor_number_valid()):
            return False, measuredMass, IDTag, properties

        c = CommModularContainer()
        c.classID = self.classID
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_REQUEST_CUSTOM_PROPERTIES
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        
        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            
            c = self._qlabs.wait_for_container(self.classID, self.actorNumber, self.FCN_RESPONSE_CUSTOM_PROPERTIES)
            if (c == None):
                pass
            else:

                if len(c.payload) >= 12:
                    measuredMass, IDTag, stringLength, = struct.unpack(">fII", c.payload[0:12])

                    if (stringLength > 0):

                        if (len(c.payload) == (12 + stringLength)):
                            properties = c.payload[12:(12+stringLength)].decode('utf-8')

                            return True, measuredMass, IDTag, properties
                    else:
                        return True, measuredMass, IDTag, properties
        
        return False, measuredMass, IDTag, properties
    

















    from qvl.qlabs import CommModularContainer
from qvl.character import QLabsCharacter
import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsAnimal(QLabsCharacter):
    """ This class implements spawning and AI navigation of the environment for animals."""

    ID_ANIMAL = 10031

    GOAT = 0
    """ Configuration constant. """
    SHEEP = 1
    """ Configuration constant. """
    COW = 2
    """ Configuration constant. """


    GOAT_STANDING = 0
    """ Speed constant for the move_to method. """
    GOAT_WALK = 0.8
    """ Speed constant for the move_to method. """
    GOAT_RUN = 4.0
    """ Speed constant for the move_to method. """

    SHEEP_STANDING = 0
    """ Speed constant for the move_to method. """
    SHEEP_WALK = 0.60
    """ Speed constant for the move_to method. """
    SHEEP_RUN = 3.0
    """ Speed constant for the move_to method. """

    COW_STANDING = 0
    """ Speed constant for the move_to method. """
    COW_WALK = 1.0
    """ Speed constant for the move_to method. """
    COW_RUN = 6.0
    """ Speed constant for the move_to method. """


    def __init__(self, qlabs, verbose=False):
       """ Constructor method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_ANIMAL
       return


from qvl.qlabs import QuanserInteractiveLabs, CommModularContainer
from quanser.common import GenericError
import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsAutoclave:

    ID_AUTOCLAVE = 140
    FCN_AUTOCLAVE_SET_DRAWER = 10
    FCN_AUTOCLAVE_SET_DRAWER_ACK = 11

    RED = 0
    GREEN = 1
    BLUE = 2


    # Initilize class
    def __init__(self):

       return

    def spawn(self, qlabs, actorNumber, location, rotation, configuration=0, waitForConfirmation=True):
        return qlabs.spawn(actorNumber, self.ID_AUTOCLAVE, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], 1, 1, 1, configuration, waitForConfirmation)

    def spawn_degrees(self, qlabs, actorNumber, location, rotation, configuration=0, waitForConfirmation=True):

        return qlabs.spawn(actorNumber, self.ID_AUTOCLAVE, location[0], location[1], location[2], rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi, 1, 1, 1, configuration, waitForConfirmation)

    def set_drawer(self, qlabs, actorNumber, open_drawer, waitForConfirmation=True):
        c = CommModularContainer()
        c.classID = self.ID_AUTOCLAVE
        c.actorNumber = actorNumber
        c.actorFunction = self.FCN_AUTOCLAVE_SET_DRAWER
        c.payload = bytearray(struct.pack(">B", open_drawer ))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            qlabs.flush_receive()

        if (qlabs.send_container(c)):
            if waitForConfirmation:
                c = qlabs.wait_for_container(self.ID_AUTOCLAVE, actorNumber, self.FCN_AUTOCLAVE_SET_DRAWER_ACK)
                return c

            return True
        else:
            return False















from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import numpy as np
import math
import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsBasicShape(QLabsActor):
    """ This class is for spawning both static and dynamic basic shapes."""

    ID_BASIC_SHAPE = 200
    """Class ID"""

    SHAPE_CUBE = 0
    """See configurations"""
    SHAPE_CYLINDER = 1
    """See configurations"""
    SHAPE_SPHERE = 2
    """See configurations"""
    SHAPE_CONE = 3
    """See configurations"""

    COMBINE_AVERAGE = 0
    COMBINE_MIN = 1
    COMBINE_MULTIPLY = 2
    COMBINE_MAX = 3


    FCN_BASIC_SHAPE_SET_MATERIAL_PROPERTIES = 10
    FCN_BASIC_SHAPE_SET_MATERIAL_PROPERTIES_ACK = 11
    FCN_BASIC_SHAPE_GET_MATERIAL_PROPERTIES = 30
    FCN_BASIC_SHAPE_GET_MATERIAL_PROPERTIES_RESPONSE = 31
    
    FCN_BASIC_SHAPE_SET_PHYSICS_PROPERTIES = 20
    FCN_BASIC_SHAPE_SET_PHYSICS_PROPERTIES_ACK = 21
    
    FCN_BASIC_SHAPE_ENABLE_DYNAMICS = 14
    FCN_BASIC_SHAPE_ENABLE_DYNAMICS_ACK = 15
    FCN_BASIC_SHAPE_SET_TRANSFORM = 16
    FCN_BASIC_SHAPE_SET_TRANSFORM_ACK = 17
    FCN_BASIC_SHAPE_ENABLE_COLLISIONS = 18
    FCN_BASIC_SHAPE_ENABLE_COLLISIONS_ACK = 19

    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_BASIC_SHAPE
       return

    def set_material_properties(self, color, roughness=0.4, metallic=False, waitForConfirmation=True):
        """Sets the visual surface properties of the shape.

        :param color: Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :param roughness: A value between 0.0 (completely smooth and reflective) to 1.0 (completely rough and diffuse). Note that reflections are rendered using screen space reflections. Only objects visible in the camera view will be rendered in the reflection of the object.
        :param metallic: Metallic (True) or non-metallic (False)
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.
        :type color: float array[3]
        :type roughness: float
        :type metallic: boolean
        :type waitForConfirmation: boolean
        :return: True if successful, False otherwise
        :rtype: boolean

        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_BASIC_SHAPE
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_BASIC_SHAPE_SET_MATERIAL_PROPERTIES
        c.payload = bytearray(struct.pack(">ffffB", color[0], color[1], color[2], roughness, metallic))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_BASIC_SHAPE, self.actorNumber, self.FCN_BASIC_SHAPE_SET_MATERIAL_PROPERTIES_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False
            
    def get_material_properties(self):
        """Gets the visual surface properties of the shape.

        :return:
            - **status** - True if successful or False otherwise
            - **color** - Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
            - **roughness** - A value between 0.0 (completely smooth and reflective) to 1.0 (completely rough and diffuse). 
            - **metallic** - Metallic (True) or non-metallic (False)
        :rtype: boolean, float array[3], float, boolean

        """
        color = [0,0,0]
        roughness = 0
        metallic = False
        
        if (self._is_actor_number_valid()):
            
            c = CommModularContainer()
            c.classID = self.ID_BASIC_SHAPE
            c.actorNumber = self.actorNumber
            c.actorFunction = self.FCN_BASIC_SHAPE_GET_MATERIAL_PROPERTIES
            c.payload = bytearray()
            c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

            self._qlabs.flush_receive()

            if (self._qlabs.send_container(c)):

                c = self._qlabs.wait_for_container(self.ID_BASIC_SHAPE, self.actorNumber, self.FCN_BASIC_SHAPE_GET_MATERIAL_PROPERTIES_RESPONSE)
                if (c == None):
                    pass
                    
                elif len(c.payload) == 17:
                    color[0], color[1], color[2], roughness, metallic, = struct.unpack(">ffff?", c.payload[0:17])
                    return True, color, roughness, metallic          
        

        return False, color, roughness, metallic          

    def set_enable_dynamics(self, enableDynamics, waitForConfirmation=True):
        """Sets the visual surface properties of the shape.

        :param enableDynamics: Enable (True) or disable (False) the shape dynamics. A dynamic actor can be pushed with other static or dynamic actors.  A static actor will generate collisions, but will not be affected by interactions with other actors.
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.
        :type enableDynamics: boolean
        :type waitForConfirmation: boolean
        :return: True if successful, False otherwise
        :rtype: boolean

        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_BASIC_SHAPE
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_BASIC_SHAPE_ENABLE_DYNAMICS
        c.payload = bytearray(struct.pack(">B", enableDynamics))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_BASIC_SHAPE, self.actorNumber, self.FCN_BASIC_SHAPE_ENABLE_DYNAMICS_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False

    def set_enable_collisions(self, enableCollisions, waitForConfirmation=True):
        """Enables and disables physics collisions. When disabled, other physics or velocity-based actors will be able to pass through.

        :param enableCollisions: Enable (True) or disable (False) the collision.
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.
        :type enableCollisions: boolean
        :type waitForConfirmation: boolean
        :return: True if successful, False otherwise
        :rtype: boolean

        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_BASIC_SHAPE
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_BASIC_SHAPE_ENABLE_COLLISIONS
        c.payload = bytearray(struct.pack(">B", enableCollisions))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_BASIC_SHAPE, self.actorNumber, self.FCN_BASIC_SHAPE_ENABLE_COLLISIONS_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False

    def set_physics_properties(self, enableDynamics, mass=1.0, linearDamping=0.01, angularDamping=0.0, staticFriction=0.0, dynamicFriction=0.7, frictionCombineMode=COMBINE_AVERAGE, restitution=0.3, restitutionCombineMode=COMBINE_AVERAGE, waitForConfirmation=True):
        """Sets the dynamic properties of the shape.

        :param enableDynamics: Enable (True) or disable (False) the shape dynamics. A dynamic actor can be pushed with other static or dynamic actors.  A static actor will generate collisions, but will not be affected by interactions with other actors.
        :param mass: (Optional) Sets the mass of the actor in kilograms.
        :param linearDamping: (Optional) Sets the damping of the actor for linear motions.
        :param angularDamping: (Optional) Sets the damping of the actor for angular motions.
        :param staticFriction: (Optional) Sets the coefficient of friction when the actor is at rest. A value of 0.0 is frictionless.
        :param dynamicFriction: (Optional) Sets the coefficient of friction when the actor is moving relative to the surface it is on. A value of 0.0 is frictionless.
        :param frictionCombineMode: (Optional) Defines how the friction between two surfaces with different coefficients should be calculated (see COMBINE constants).
        :param restitution: (Optional) The coefficient of restitution defines how plastic or elastic a collision is. A value of 0.0 is plastic and will absorb all energy. A value of 1.0 is elastic and will bounce forever. A value greater than 1.0 will add energy with each collision.
        :param restitutionCombineMode: (Optional) Defines how the restitution between two surfaces with different coefficients should be calculated (see COMBINE constants).
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.

        :type enableDynamics: boolean
        :type mass: float
        :type linearDamping: float
        :type angularDamping: float
        :type staticFriction: float
        :type dynamicFriction: float
        :type frictionCombineMode: byte
        :type restitution: float
        :type restitutionCombineMode: byte
        :type waitForConfirmation: boolean
        :return: True if successful, False otherwise
        :rtype: boolean

        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_BASIC_SHAPE
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_BASIC_SHAPE_SET_PHYSICS_PROPERTIES
        c.payload = bytearray(struct.pack(">BfffffBfB", enableDynamics, mass, linearDamping, angularDamping, staticFriction, dynamicFriction, frictionCombineMode, restitution, restitutionCombineMode))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_BASIC_SHAPE, self.actorNumber, self.FCN_BASIC_SHAPE_SET_PHYSICS_PROPERTIES_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False

    def set_transform(self, location, rotation, scale, waitForConfirmation=True):
        """Sets the location, rotation in radians, and scale. If a shape is parented to another actor then the location, rotation, and scale are relative to the parent actor.

        :param location: An array of floats for x, y and z coordinates in full-scale units. 
        :param rotation: An array of floats for the roll, pitch, and yaw in radians
        :param scale: An array of floats for the scale in the x, y, and z directions.
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type waitForConfirmation: boolean
        :return: True if successful or False otherwise
        :rtype: boolean
        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_BASIC_SHAPE
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_BASIC_SHAPE_SET_TRANSFORM
        c.payload = bytearray(struct.pack(">fffffffff", location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], scale[0], scale[1], scale[2]))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_BASIC_SHAPE, self.actorNumber, self.FCN_BASIC_SHAPE_SET_TRANSFORM_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False

    def set_transform_degrees(self, location, rotation, scale, waitForConfirmation=True):
        """Sets the location, rotation in degrees, and scale. If a shape is parented to another actor then the location, rotation, and scale are relative to the parent actor.

        :param location: An array of floats for x, y and z coordinates in full-scale units.
        :param rotation: An array of floats for the roll, pitch, and yaw in degrees
        :param scale: An array of floats for the scale in the x, y, and z directions.
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type waitForConfirmation: boolean
        :return: True if successful or False otherwise
        :rtype: boolean
        """

        return self.set_transform(location, [rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi], scale, waitForConfirmation)

    def _rotate_vector_2d_degrees(self, vector, angle):
        """Internal helper function to rotate a vector on the z plane.

        :param vector: Vector to rotate
        :param angle: Rotation angle in radians
        :type vector: float array[3]
        :type angle: float
        :return: Rotated vector
        :rtype: float array[3]
        """

        result = [0,0,vector[2]]

        result[0] = math.cos(angle)*vector[0] - math.sin(angle)*vector[1]
        result[1] = math.sin(angle)*vector[0] + math.cos(angle)*vector[1]

        return result

    def spawn_id_box_walls_from_end_points(self, actorNumber, startLocation, endLocation, height, thickness, color=[1,1,1], waitForConfirmation=True):
        """Given a start and end point, this helper method calculates the position, rotation, and scale required to place a box on top of this line.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param startLocation: An array of floats for x, y and z coordinates.
        :param endLocation: An array of floats for x, y and z coordinates.
        :param height: The height of the wall.
        :param thickness: The width or thickness of the wall.
        :param color: Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.
        :type actorNumber: uint32
        :type startLocation: float array[3]
        :type endLocation: float array[3]
        :type height: float
        :type thickness: float
        :type color: float array[3]
        :type waitForConfirmation: boolean
        :return: True if successful or False otherwise
        :rtype: boolean
        """


        length = math.sqrt(pow(startLocation[0] - endLocation[0],2) + pow(startLocation[1] - endLocation[1],2) + pow(startLocation[2] - endLocation[2],2))
        location = [(startLocation[0] + endLocation[0])/2, (startLocation[1] + endLocation[1])/2, (startLocation[2] + endLocation[2])/2]

        yRotation = -math.asin( (endLocation[2] - startLocation[2])/(length) )
        zRotation = math.atan2( (endLocation[1] - startLocation[1]), (endLocation[0] - startLocation[0]))

        shiftedLocation = [location[0]+math.sin(yRotation)*math.cos(zRotation)*height/2, location[1]+math.sin(yRotation)*math.sin(zRotation)*height/2, location[2]+math.cos(yRotation)*height/2]

        if (0 == self.spawn_id(actorNumber, shiftedLocation, [0, yRotation, zRotation], [length, thickness, height], self.SHAPE_CUBE, waitForConfirmation)):
            if (True == self.set_material_properties(color, 1, False, waitForConfirmation)):
                return True
            else:
                return False

        else:
            return False

    def spawn_id_box_walls_from_center(self, actorNumbers, centerLocation, yaw, xSize, ySize, zHeight, wallThickness, floorThickness=0, wallColor=[1,1,1], floorColor=[1,1,1], waitForConfirmation=True):
        """Creates a container-like box with 4 walls and an optional floor.

        :param actorNumbers: An array of 5 user defined unique identifiers for the class actors in QLabs.
        :param centerLocation: An array of floats for x, y and z coordinates.
        :param yaw: Rotation about the z axis in radians.
        :param xSize: Size of the box in the x direction.
        :param ySize: Size of the box in the y direction.
        :param zSize: Size of the box in the z direction.
        :param wallThickness: The thickness of the walls.
        :param floorThickness: (Optional) The thickness of the floor. Setting this to 0 will spawn a box without a floor.
        :param wallColor: (Optional) Red, Green, Blue components of the wall color on a 0.0 to 1.0 scale.
        :param floorColor: (Optional) Red, Green, Blue components of the floor color on a 0.0 to 1.0 scale.
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.

        :type actorNumbers: uint32 array[5]
        :type centerLocation: float array[3]
        :type yaw: float
        :type xSize: float
        :type ySize: float
        :type zSize: float
        :type wallThickness: float
        :type floorThickness: float
        :type wallColor: float array[3]
        :type floorColor: float array[3]
        :type waitForConfirmation: boolean

        :return: True if successful or False otherwise
        :rtype: boolean
        """
        origin = [centerLocation[0],  centerLocation[1], centerLocation[2] + zHeight/2 + floorThickness]

        location = np.add(origin, self._rotate_vector_2d_degrees([xSize/2 + wallThickness/2, 0, 0], yaw) )
        if (0 != self.spawn_id(actorNumbers[0], location, [0, 0, yaw], [wallThickness, ySize, zHeight], self.SHAPE_CUBE, waitForConfirmation)):
            return False
        if (True != self.set_material_properties(wallColor, 1, False, waitForConfirmation)):
            return False

        location = np.add(origin, self._rotate_vector_2d_degrees([ - xSize/2 - wallThickness/2, 0, 0], yaw) )
        if (0 != self.spawn_id(actorNumbers[1], location, [0, 0, yaw], [wallThickness, ySize, zHeight], self.SHAPE_CUBE, waitForConfirmation)):
            return False
        if (True != self.set_material_properties(wallColor, 1, False, waitForConfirmation)):
            return False


        location = np.add(origin, self._rotate_vector_2d_degrees([0, ySize/2 + wallThickness/2, 0], yaw) )
        if (0 != self.spawn_id(actorNumbers[2], location, [0, 0, yaw], [xSize + wallThickness*2, wallThickness, zHeight], self.SHAPE_CUBE, waitForConfirmation)):
            return False
        if (True != self.set_material_properties(wallColor, 1, False, waitForConfirmation)):
            return False


        location = np.add(origin, self._rotate_vector_2d_degrees([0, - ySize/2 - wallThickness/2, 0], yaw) )
        if (0 != self.spawn_id(actorNumbers[3], location, [0, 0, yaw], [xSize + wallThickness*2, wallThickness, zHeight], self.SHAPE_CUBE, waitForConfirmation)):
            return False
        if (True != self.set_material_properties(wallColor, 1, False, waitForConfirmation)):
            return False

        if (floorThickness > 0):
            if (0 != self.spawn_id(actorNumbers[4], [centerLocation[0], centerLocation[1], centerLocation[2]+ floorThickness/2], [0, 0, yaw], [xSize+wallThickness*2, ySize+wallThickness*2, floorThickness], self.SHAPE_CUBE, waitForConfirmation)):
                return False
            if (True != self.set_material_properties(floorColor, 1, False, waitForConfirmation)):
                return False

        return True

    def spawn_id_box_walls_from_center_degrees(self, actorNumbers, centerLocation, yaw, xSize, ySize, zHeight, wallThickness, floorThickness=0, wallColor=[1,1,1], floorColor=[1,1,1], waitForConfirmation=True):
        """Creates a container-like box with 4 walls and an optional floor.

        :param actorNumbers: An array of 5 user defined unique identifiers for the class actors in QLabs.
        :param centerLocation: An array of floats for x, y and z coordinates.
        :param yaw: Rotation about the z axis in degrees.
        :param xSize: Size of the box in the x direction.
        :param ySize: Size of the box in the y direction.
        :param zSize: Size of the box in the z direction.
        :param wallThickness: The thickness of the walls.
        :param floorThickness: (Optional) The thickness of the floor. Setting this to 0 will spawn a box without a floor.
        :param wallColor: (Optional) Red, Green, Blue components of the wall color on a 0.0 to 1.0 scale.
        :param floorColor: (Optional) Red, Green, Blue components of the floor color on a 0.0 to 1.0 scale.
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.

        :type actorNumbers: uint32 array[5]
        :type centerLocation: float array[3]
        :type yaw: float
        :type xSize: float
        :type ySize: float
        :type zSize: float
        :type wallThickness: float
        :type floorThickness: float
        :type wallColor: float array[3]
        :type floorColor: float array[3]
        :type waitForConfirmation: boolean

        :return: True if successful or False otherwise
        :rtype: boolean
        """
        return self.spawn_id_box_walls_from_center(actorNumbers, centerLocation, yaw/180*math.pi, xSize, ySize, zHeight, wallThickness, floorThickness, wallColor, floorColor, waitForConfirmation)













from qvl.qlabs import QuanserInteractiveLabs, CommModularContainer
from qvl.widget import QLabsWidget
from quanser.common import GenericError
import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsBottleTableAttachment:


    ID_BOTTLE_TABLE_ATTACHMENT = 101

    FCN_BOTTLE_TABLE_ATTACHMENT_REQUEST_LOAD_MASS = 91
    FCN_BOTTLE_TABLE_ATTACHMENT_RESPONSE_LOAD_MASS = 92

    # Initialize class
    def __init__(self):

       return

    def spawn(self, qlabs, actorNumber, location, rotation, waitForConfirmation=True):
        return qlabs.spawn(actorNumber, self.ID_BOTTLE_TABLE_ATTACHMENT, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], 1, 1, 1, 0, waitForConfirmation)

    def spawn_degrees(self, qlabs, actorNumber, location, rotation, waitForConfirmation=True):

        return qlabs.spawn(actorNumber, self.ID_BOTTLE_TABLE_ATTACHMENT, location[0], location[1], location[2], rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi, 1, 1, 1, 0, waitForConfirmation)

    def spawn_and_parent_with_relative_transform(self, qlabs, actorNumber, location, rotation, parentClass, parentActorNumber, parentComponent, waitForConfirmation=True):
        return qlabs.spawn_and_parent_with_relative_transform(actorNumber, self.ID_BOTTLE_TABLE_ATTACHMENT, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], 1, 1, 1, 0, parentClass, parentActorNumber, parentComponent, waitForConfirmation)

    def get_measured_mass(self, qlabs, actorNumber):
        c = CommModularContainer()
        c.classID = self.ID_BOTTLE_TABLE_ATTACHMENT
        c.actorNumber = actorNumber
        c.actorFunction = self.FCN_BOTTLE_TABLE_ATTACHMENT_REQUEST_LOAD_MASS
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        qlabs.flush_receive()

        if (qlabs.send_container(c)):
            c = qlabs.wait_for_container(self.ID_BOTTLE_TABLE_ATTACHMENT, actorNumber, self.FCN_BOTTLE_TABLE_ATTACHMENT_RESPONSE_LOAD_MASS)

            if (len(c.payload) == 4):
                mass,  = struct.unpack(">f", c.payload)
                return mass
            else:
                return -1.0

        else:
            return -1.0

class QLabsBottleTableSupport:


    ID_BOTTLE_TABLE_SUPPORT = 102



    # Initialize class
    def __init__(self):

       return

    def spawn(self, qlabs, actorNumber, location, rotation, waitForConfirmation=True):
        return qlabs.spawn(actorNumber, self.ID_BOTTLE_TABLE_SUPPORT, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], 1, 1, 1, 0, waitForConfirmation)

    def spawn_degrees(self, qlabs, actorNumber, location, rotation, waitForConfirmation=True):

        return qlabs.spawn(actorNumber, self.ID_BOTTLE_TABLE_SUPPORT, location[0], location[1], location[2], rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi, 1, 1, 1, 0, waitForConfirmation)

    def spawn_and_parent_with_relative_transform(self, qlabs, actorNumber, location, rotation, parentClass, parentActorNumber, parentComponent, waitForConfirmation=True):
        return qlabs.spawn_and_parent_with_relative_transform(actorNumber, self.ID_BOTTLE_TABLE_SUPPORT, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], 1, 1, 1, 0, parentClass, parentActorNumber, parentComponent, waitForConfirmation)

class QLabsBottleTableSensorTowerShort:


    ID_BOTTLE_TABLE_SENSOR_TOWER_SHORT = 103

    FCN_BOTTLE_TABLE_SENSOR_TOWER_SHORT_REQUEST_PROXIMITY = 17
    FCN_BOTTLE_TABLE_SENSOR_TOWER_SHORT_RESPONSE_PROXIMITY = 18

    # Initialize class
    def __init__(self):

       return

    def spawn(self, qlabs, actorNumber, location, rotation, waitForConfirmation=True):
        return qlabs.spawn(actorNumber, self.ID_BOTTLE_TABLE_SENSOR_TOWER_SHORT, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], 1, 1, 1, 0, waitForConfirmation)

    def spawn_degrees(self, qlabs, actorNumber, location, rotation, waitForConfirmation=True):

        return qlabs.spawn(actorNumber, self.ID_BOTTLE_TABLE_SENSOR_TOWER_SHORT, location[0], location[1], location[2], rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi, 1, 1, 1, 0, waitForConfirmation)

    def spawn_and_parent_with_relative_transform(self, qlabs, actorNumber, location, rotation, parentClass, parentActorNumber, parentComponent, waitForConfirmation=True):
        return qlabs.spawn_and_parent_with_relative_transform(actorNumber, self.ID_BOTTLE_TABLE_SENSOR_TOWER_SHORT, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], 1, 1, 1, 0, parentClass, parentActorNumber, parentComponent, waitForConfirmation)

    def spawn_and_parent_with_relative_transform_degrees(self, qlabs, actorNumber, location, rotation, parentClass, parentActorNumber, parentComponent, waitForConfirmation=True):

        return qlabs.spawn_and_parent_with_relative_transform(actorNumber, self.ID_BOTTLE_TABLE_SENSOR_TOWER_SHORT, location[0], location[1], location[2], rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi, 1, 1, 1, 0, parentClass, parentActorNumber, parentComponent, waitForConfirmation)

    def get_proximity(self, qlabs, actorNumber):
        c = CommModularContainer()
        c.classID = self.ID_BOTTLE_TABLE_SENSOR_TOWER_SHORT
        c.actorNumber = actorNumber
        c.actorFunction = self.FCN_BOTTLE_TABLE_SENSOR_TOWER_SHORT_REQUEST_PROXIMITY
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        qlabs.flush_receive()

        relative_x = 0.0
        relative_y = 0.0
        relative_z = 0.0
        properties = ""
        properties_size = 0

        if (qlabs.send_container(c)):
            c = qlabs.wait_for_container(self.ID_BOTTLE_TABLE_SENSOR_TOWER_SHORT, actorNumber, self.FCN_BOTTLE_TABLE_SENSOR_TOWER_SHORT_RESPONSE_PROXIMITY)

            if (len(c.payload) >= 16):
                relative_x, relative_y, relative_z, properties_size, = struct.unpack(">fffI", c.payload[0:16])

                if (properties_size > 0):
                    properties = c.payload[16:(16+properties_size)].decode("utf-8")

        return [relative_x, relative_y, relative_z], properties

class QLabsBottleTableSensorTowerTall:


    ID_BOTTLE_TABLE_SENSOR_TOWER_TALL = 104

    FCN_BOTTLE_TABLE_SENSOR_TOWER_TALL_REQUEST_TOF = 15
    FCN_BOTTLE_TABLE_SENSOR_TOWER_TALL_RESPONSE_TOF = 16
    FCN_BOTTLE_TABLE_SENSOR_TOWER_TALL_REQUEST_PROXIMITY = 19
    FCN_BOTTLE_TABLE_SENSOR_TOWER_TALL_RESPONSE_PROXIMITY = 20


    # Initialize class
    def __init__(self):

       return

    def spawn(self, qlabs, actorNumber, location, rotation, waitForConfirmation=True):
        return qlabs.spawn(actorNumber, self.ID_BOTTLE_TABLE_SENSOR_TOWER_TALL, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], 1, 1, 1, 0, waitForConfirmation)

    def spawn_degrees(self, qlabs, actorNumber, location, rotation, waitForConfirmation=True):

        return qlabs.spawn(actorNumber, self.ID_BOTTLE_TABLE_SENSOR_TOWER_TALL, location[0], location[1], location[2], rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi, 1, 1, 1, 0, waitForConfirmation)

    def spawn_and_parent_with_relative_transform(self, qlabs, actorNumber, location, rotation, parentClass, parentActorNumber, parentComponent, waitForConfirmation=True):
        return qlabs.spawn_and_parent_with_relative_transform(actorNumber, self.ID_BOTTLE_TABLE_SENSOR_TOWER_TALL, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], 1, 1, 1, 0, parentClass, parentActorNumber, parentComponent, waitForConfirmation)

    def spawn_and_parent_with_relative_transform_degrees(self, qlabs, actorNumber, location, rotation, parentClass, parentActorNumber, parentComponent, waitForConfirmation=True):

        return qlabs.spawn_and_parent_with_relative_transform(actorNumber, self.ID_BOTTLE_TABLE_SENSOR_TOWER_TALL, location[0], location[1], location[2], rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi, 1, 1, 1, 0, parentClass, parentActorNumber, parentComponent, waitForConfirmation)

    def get_proximity(self, qlabs, actorNumber):
        c = CommModularContainer()
        c.classID = self.ID_BOTTLE_TABLE_SENSOR_TOWER_TALL
        c.actorNumber = actorNumber
        c.actorFunction = self.FCN_BOTTLE_TABLE_SENSOR_TOWER_TALL_REQUEST_PROXIMITY
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        qlabs.flush_receive()

        relative_x = 0.0
        relative_y = 0.0
        relative_z = 0.0
        properties = ""
        properties_size = 0

        if (qlabs.send_container(c)):
            c = qlabs.wait_for_container(self.ID_BOTTLE_TABLE_SENSOR_TOWER_TALL, actorNumber, self.FCN_BOTTLE_TABLE_SENSOR_TOWER_TALL_RESPONSE_PROXIMITY)

            if (len(c.payload) >= 16):
                relative_x, relative_y, relative_z, properties_size, = struct.unpack(">fffI", c.payload[0:16])

                if (properties_size > 0):
                    properties = c.payload[16:(16+properties_size)].decode("utf-8")

        return [relative_x, relative_y, relative_z], properties



    def get_tof(self, qlabs, actorNumber):
        c = CommModularContainer()
        c.classID = self.ID_BOTTLE_TABLE_SENSOR_TOWER_TALL
        c.actorNumber = actorNumber
        c.actorFunction = self.FCN_BOTTLE_TABLE_SENSOR_TOWER_TALL_REQUEST_TOF
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        qlabs.flush_receive()

        tof_distance = 0.0

        if (qlabs.send_container(c)):
            c = qlabs.wait_for_container(self.ID_BOTTLE_TABLE_SENSOR_TOWER_TALL, actorNumber, self.FCN_BOTTLE_TABLE_SENSOR_TOWER_TALL_RESPONSE_TOF)

            if (len(c.payload) == 4):
                tof_distance, = struct.unpack(">f", c.payload[0:4])


        return tof_distance



















from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor
import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsCharacter(QLabsActor):
    """ This base class implements spawning and AI navigation of the environment for characters."""

    FCN_CHARACTER_MOVE_TO = 10
    FCN_CHARACTER_MOVE_TO_ACK = 11


    def __init__(self, qlabs, verbose=False):
       """ Constructor method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       return

    def move_to(self, location, speed, waitForConfirmation=True):
        """Commands an actor to move from the present location to a new target location by using AI path navigation.

        :param location: A target destination as an array of floats for x, y and z coordinates in full-scale units.
        :param speed: The speed at which the person should walk to the destination (refer to the constants for recommended speeds)
        :param waitForConfirmation: (Optional) Wait for confirmation before proceeding. This makes the method a blocking operation, but only until the command is received. The time for the actor to traverse to the destination is always non-blocking.
        :type location: float array[3]
        :type speed: float
        :type waitForConfirmation: boolean
        :return:
            - **status** - `True` if successful, `False` otherwise
        :rtype: boolean

        .. tip::

            Ensure the start and end locations are in valid navigation areas so the actor can find a path to the destination.

        """

        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.classID
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_CHARACTER_MOVE_TO
        c.payload = bytearray(struct.pack(">ffff", location[0], location[1], location[2], speed))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.classID, self.actorNumber, self.FCN_CHARACTER_MOVE_TO_ACK)
                if (c == None):
                    return False
                else:
                    return True
            return True
        else:
            return False














from qvl.qlabs import QuanserInteractiveLabs, CommModularContainer
from quanser.common import GenericError
from qvl.actor import QLabsActor
import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsConveyorCurved(QLabsActor):


    ID_CONVEYOR_CURVED = 211

    FCN_CONVEYOR_CURVED_SET_SPEED = 10
    FCN_CONVEYOR_CURVED_SET_SPEED_ACK = 11


    # Initialize class
    def __init__(self, qlabs, verbose=False):
        """ Constructor Method

        :param qlabs: A QuanserInteractiveLabs object
        :param verbose: (Optional) Print error information to the console.
        :type qlabs: object
        :type verbose: boolean
        """

        self._qlabs = qlabs
        self._verbose = verbose
        self.classID = self.ID_CONVEYOR_CURVED
        return

    def set_speed(self, speed):
        c = CommModularContainer()
        c.classID = self.ID_CONVEYOR_CURVED
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_CONVEYOR_CURVED_SET_SPEED
        c.payload = bytearray(struct.pack(">f", speed))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_CONVEYOR_CURVED, self.actorNumber, self.FCN_CONVEYOR_CURVED_SET_SPEED_ACK)

            return True
        else:
            return False
        
















        from qvl.qlabs import QuanserInteractiveLabs, CommModularContainer
from quanser.common import GenericError
from qvl.actor import QLabsActor
import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsConveyorStraight(QLabsActor):


    ID_CONVEYOR_STRAIGHT = 210

    FCN_CONVEYOR_STRAIGHT_SET_SPEED = 10
    FCN_CONVEYOR_STRAIGHT_SET_SPEED_ACK = 11


    # Initialize class
    def __init__(self, qlabs, verbose=False):
        """ Constructor Method

        :param qlabs: A QuanserInteractiveLabs object
        :param verbose: (Optional) Print error information to the console.
        :type qlabs: object
        :type verbose: boolean
        """

        self._qlabs = qlabs
        self._verbose = verbose
        self.classID = self.ID_CONVEYOR_STRAIGHT
        return

    def set_speed(self, speed):
        c = CommModularContainer()
        c.classID = self.ID_CONVEYOR_STRAIGHT
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_CONVEYOR_STRAIGHT_SET_SPEED
        c.payload = bytearray(struct.pack(">f", speed))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_CONVEYOR_STRAIGHT, self.actorNumber, self.FCN_CONVEYOR_STRAIGHT_SET_SPEED_ACK)

            return True
        else:
            return False
        













        from qvl.actor import QLabsActor

import math
import struct

class QLabsCrosswalk(QLabsActor):
    """This class is for spawning crosswalks."""

    ID_CROSSWALK = 10010
    """Class ID"""

    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_CROSSWALK
       return


    def spawn_id(self, actorNumber, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, waitForConfirmation=True):
        """Spawns a new crosswalk actor.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type actorNumber: uint32
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 2 actor number not available or already in use, 3 unknown error, -1 communications error
        :rtype: int32

        """
        scale = [scale[2], scale[1], scale[0]]
        return super().spawn_id(actorNumber, location, rotation, scale, configuration, waitForConfirmation)
        
    def spawn_id_degrees(self, actorNumber, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, waitForConfirmation=True):
        """Spawns a new crosswalk actor.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type actorNumber: uint32
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 2 actor number not available or already in use, 3 unknown error, -1 communications error
        :rtype: int32

        """
        scale = [scale[2], scale[1], scale[0]]
        return super().spawn_id_degrees(actorNumber, location, rotation, scale, configuration, waitForConfirmation)
        
    def spawn(self, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, waitForConfirmation=True):
        """Spawns a new crosswalk actor with the next available actor number within this class.

        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred. Note that if this is False, the returned actor number will be invalid.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 3 unknown error, -1 communications error.
            - **actorNumber** - An actor number to use for future references.
        :rtype: int32, int32

        """   
        
        scale = [scale[2], scale[1], scale[0]]
        return super().spawn(location, rotation, scale, configuration, waitForConfirmation)
               
    def spawn_degrees(self, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, waitForConfirmation=True):
        """Spawns a new crosswalk actor with the next available actor number within this class.

        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in degrees
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred. Note that if this is False, the returned actor number will be invalid.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 3 unknown error, -1 communications error.
            - **actorNumber** - An actor number to use for future references.
        :rtype: int32, int32
 
        """

        scale = [scale[2], scale[1], scale[0]]
        return super().spawn_degrees(location, rotation, scale, configuration, waitForConfirmation)
        
    def spawn_id_and_parent_with_relative_transform(self, actorNumber, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, parentClassID=0, parentActorNumber=0, parentComponent=0, waitForConfirmation=True):
        """Spawns a new crosswalk actor relative to an existing actor and creates a kinematic relationship.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param parentClassID: See the ID variables in the respective library classes for the class identifier
        :param parentActorNumber: User defined unique identifier for the class actor in QLabs
        :param parentComponent: `0` for the origin of the parent actor, see the parent class for additional reference frame options
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type actorNumber: uint32
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type parentClassID: uint32
        :type parentActorNumber: uint32
        :type parentComponent: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 2 actor number not available or already in use, 3 cannot find the parent actor, 4 unknown error, -1 communications error
        :rtype: int32

        """
        scale = [scale[2], scale[1], scale[0]]
        return super().spawn_id_and_parent_with_relative_transform(actorNumber, location, rotation, scale, configuration, parentClassID, parentActorNumber, parentComponent, waitForConfirmation)

    def spawn_id_and_parent_with_relative_transform_degrees(self, actorNumber, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, parentClassID=0, parentActorNumber=0, parentComponent=0, waitForConfirmation=True):
        """Spawns a new crosswalk actor relative to an existing actor and creates a kinematic relationship.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in degrees
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param parentClassID: See the ID variables in the respective library classes for the class identifier
        :param parentActorNumber: User defined unique identifier for the class actor in QLabs
        :param parentComponent: `0` for the origin of the parent actor, see the parent class for additional reference frame options
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type actorNumber: uint32
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type parentClassID: uint32
        :type parentActorNumber: uint32
        :type parentComponent: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 2 actor number not available or already in use, 3 cannot find the parent actor, 4 unknown error, -1 communications error
        :rtype: int32

        """
        scale = [scale[2], scale[1], scale[0]]
        return super().spawn_id_and_parent_with_relative_transform_degrees(actorNumber, location, rotation, scale, configuration, parentClassID, parentActorNumber, parentComponent, waitForConfirmation)


















from qvl.qlabs import QuanserInteractiveLabs, CommModularContainer
from quanser.common import GenericError
from qvl.actor import QLabsActor
import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsDeliveryTube(QLabsActor):


    ID_DELIVERY_TUBE = 80

    FCN_DELIVERY_TUBE_SPAWN_BLOCK = 10
    FCN_DELIVERY_TUBE_SPAWN_BLOCK_ACK = 11
    FCN_DELIVERY_TUBE_SET_HEIGHT = 12
    FCN_DELIVERY_TUBE_SET_HEIGHT_ACK = 13

    BLOCK_CUBE = 0
    BLOCK_CYLINDER = 1
    BLOCK_SPHERE = 2
    BLOCK_GEOSPHERE = 3

    CONFIG_HOVER = 0
    CONFIG_NO_HOVER = 1

    # Initialize class
    def __init__(self, qlabs, verbose=False):
        """ Constructor Method

        :param qlabs: A QuanserInteractiveLabs object
        :param verbose: (Optional) Print error information to the console.
        :type qlabs: object
        :type verbose: boolean
        """

        self._qlabs = qlabs
        self._verbose = verbose
        self.classID = self.ID_DELIVERY_TUBE
        return

    def spawn_block(self, blockType, mass, yawRotation, color):
        c = CommModularContainer()
        c.classID = self.ID_DELIVERY_TUBE
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_DELIVERY_TUBE_SPAWN_BLOCK
        c.payload = bytearray(struct.pack(">Ifffff", blockType, mass, yawRotation, color[0], color[1], color[2]))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_DELIVERY_TUBE, self.actorNumber, self.FCN_DELIVERY_TUBE_SPAWN_BLOCK_ACK)

            return True
        else:
            return False

    def set_height(self, height):
        c = CommModularContainer()
        c.classID = self.ID_DELIVERY_TUBE
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_DELIVERY_TUBE_SET_HEIGHT
        c.payload = bytearray(struct.pack(">f", height))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_DELIVERY_TUBE, self.actorNumber, self.FCN_DELIVERY_TUBE_SET_HEIGHT_ACK)

            return True
        else:
            return False


class QLabsDeliveryTubeBottles(QLabsActor):


    ID_DELIVERY_TUBE_BOTTLES = 81

    FCN_DELIVERY_TUBE_SPAWN_CONTAINER = 10
    FCN_DELIVERY_TUBE_SPAWN_CONTAINER_ACK = 11
    FCN_DELIVERY_TUBE_SET_HEIGHT = 12
    FCN_DELIVERY_TUBE_SET_HEIGHT_ACK = 13

    PLASTIC_BOTTLE = 4
    METAL_CAN = 5

    CONFIG_HOVER = 0
    CONFIG_NO_HOVER = 1

    # Initialize class
    def __init__(self, qlabs, verbose=False):
        """ Constructor Method

        :param qlabs: A QuanserInteractiveLabs object
        :param verbose: (Optional) Print error information to the console.
        :type qlabs: object
        :type verbose: boolean
        """

        self._qlabs = qlabs
        self._verbose = verbose
        self.classID = self.ID_DELIVERY_TUBE_BOTTLES
        return

    def spawn_container(self, metallic, color, mass, propertyString="", height = 0.1, diameter = 0.65, roughness = 0.65):
        c = CommModularContainer()
        c.classID = self.ID_DELIVERY_TUBE_BOTTLES
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_DELIVERY_TUBE_SPAWN_CONTAINER
        c.payload = bytearray(struct.pack(">ffBffffffI", height, diameter, metallic, color[0], color[1], color[2], 1.0, roughness, mass, len(propertyString)))
        c.payload = c.payload + bytearray(propertyString.encode('utf-8'))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_DELIVERY_TUBE_BOTTLES, self.actorNumber, self.FCN_DELIVERY_TUBE_SPAWN_CONTAINER_ACK)

            return True
        else:
            return False

    def set_height(self, height):
        c = CommModularContainer()
        c.classID = self.ID_DELIVERY_TUBE_BOTTLES
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_DELIVERY_TUBE_SET_HEIGHT
        c.payload = bytearray(struct.pack(">f", height))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_DELIVERY_TUBE_BOTTLES, self.actorNumber, self.FCN_DELIVERY_TUBE_SET_HEIGHT_ACK)

            return True
        else:
            return False












from qvl.qlabs import QuanserInteractiveLabs, CommModularContainer
from quanser.common import GenericError
from qvl.actor import QLabsActor
import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsDeliveryTube(QLabsActor):


    ID_DELIVERY_TUBE = 80

    FCN_DELIVERY_TUBE_SPAWN_BLOCK = 10
    FCN_DELIVERY_TUBE_SPAWN_BLOCK_ACK = 11
    FCN_DELIVERY_TUBE_SET_HEIGHT = 12
    FCN_DELIVERY_TUBE_SET_HEIGHT_ACK = 13

    BLOCK_CUBE = 0
    BLOCK_CYLINDER = 1
    BLOCK_SPHERE = 2
    BLOCK_GEOSPHERE = 3

    CONFIG_HOVER = 0
    CONFIG_NO_HOVER = 1

    # Initialize class
    def __init__(self, qlabs, verbose=False):
        """ Constructor Method

        :param qlabs: A QuanserInteractiveLabs object
        :param verbose: (Optional) Print error information to the console.
        :type qlabs: object
        :type verbose: boolean
        """

        self._qlabs = qlabs
        self._verbose = verbose
        self.classID = self.ID_DELIVERY_TUBE
        return

    def spawn_block(self, blockType, mass, yawRotation, color):
        c = CommModularContainer()
        c.classID = self.ID_DELIVERY_TUBE
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_DELIVERY_TUBE_SPAWN_BLOCK
        c.payload = bytearray(struct.pack(">Ifffff", blockType, mass, yawRotation, color[0], color[1], color[2]))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_DELIVERY_TUBE, self.actorNumber, self.FCN_DELIVERY_TUBE_SPAWN_BLOCK_ACK)

            return True
        else:
            return False

    def set_height(self, height):
        c = CommModularContainer()
        c.classID = self.ID_DELIVERY_TUBE
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_DELIVERY_TUBE_SET_HEIGHT
        c.payload = bytearray(struct.pack(">f", height))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_DELIVERY_TUBE, self.actorNumber, self.FCN_DELIVERY_TUBE_SET_HEIGHT_ACK)

            return True
        else:
            return False


class QLabsDeliveryTubeBottles(QLabsActor):


    ID_DELIVERY_TUBE_BOTTLES = 81

    FCN_DELIVERY_TUBE_SPAWN_CONTAINER = 10
    FCN_DELIVERY_TUBE_SPAWN_CONTAINER_ACK = 11
    FCN_DELIVERY_TUBE_SET_HEIGHT = 12
    FCN_DELIVERY_TUBE_SET_HEIGHT_ACK = 13

    PLASTIC_BOTTLE = 4
    METAL_CAN = 5

    CONFIG_HOVER = 0
    CONFIG_NO_HOVER = 1

    # Initialize class
    def __init__(self, qlabs, verbose=False):
        """ Constructor Method

        :param qlabs: A QuanserInteractiveLabs object
        :param verbose: (Optional) Print error information to the console.
        :type qlabs: object
        :type verbose: boolean
        """

        self._qlabs = qlabs
        self._verbose = verbose
        self.classID = self.ID_DELIVERY_TUBE_BOTTLES
        return

    def spawn_container(self, metallic, color, mass, propertyString="", height = 0.1, diameter = 0.65, roughness = 0.65):
        c = CommModularContainer()
        c.classID = self.ID_DELIVERY_TUBE_BOTTLES
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_DELIVERY_TUBE_SPAWN_CONTAINER
        c.payload = bytearray(struct.pack(">ffBffffffI", height, diameter, metallic, color[0], color[1], color[2], 1.0, roughness, mass, len(propertyString)))
        c.payload = c.payload + bytearray(propertyString.encode('utf-8'))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_DELIVERY_TUBE_BOTTLES, self.actorNumber, self.FCN_DELIVERY_TUBE_SPAWN_CONTAINER_ACK)

            return True
        else:
            return False

    def set_height(self, height):
        c = CommModularContainer()
        c.classID = self.ID_DELIVERY_TUBE_BOTTLES
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_DELIVERY_TUBE_SET_HEIGHT
        c.payload = bytearray(struct.pack(">f", height))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_DELIVERY_TUBE_BOTTLES, self.actorNumber, self.FCN_DELIVERY_TUBE_SET_HEIGHT_ACK)

            return True
        else:
            return False

















from qvl.qlabs import CommModularContainer
from quanser.common import GenericError
import math
import os
import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsEnvironmentOutdoors:
    """ This class modifies QLabs open worlds with outdoor environments."""

    ID_ENVIRONMENT_OUTDOORS = 1100
    """Class ID"""

    FCN_SET_TIME_OF_DAY = 10
    FCN_SET_TIME_OF_DAY_ACK = 11
    FCN_OVERRIDE_OUTDOOR_LIGHTING = 12
    FCN_OVERRIDE_OUTDOOR_LIGHTING_ACK = 13
    FCN_SET_WEATHER_PRESET = 14
    FCN_SET_WEATHER_PRESET_ACK = 15

    CLEAR_SKIES = 0
    PARTLY_CLOUDY = 1
    CLOUDY = 2
    OVERCAST = 3
    FOGGY = 4
    LIGHT_RAIN = 5
    RAIN = 6
    THUNDERSTORM = 7
    LIGHT_SNOW = 8
    SNOW = 9
    BLIZZARD = 10


    _qlabs = None
    _verbose = False

    def __init__(self, qlabs, verbose=False):
       """ Constructor method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       return

    def set_time_of_day(self, time):
        """
        Set the time of day for an outdoor environment.

        :param time: A value from 0 to 24. Midnight is a value 0 or 24. Noon is a value of 12.
        :type time: float
        :return: `True` if setting the time was successful, `False` otherwise
        :rtype: boolean

        """
        c = CommModularContainer()
        c.classID = self.ID_ENVIRONMENT_OUTDOORS
        c.actorNumber = 0
        c.actorFunction = self.FCN_SET_TIME_OF_DAY
        c.payload = bytearray(struct.pack(">f", time))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_ENVIRONMENT_OUTDOORS, 0, self.FCN_SET_TIME_OF_DAY_ACK)
            if (c == None):
                return False
            else:
                return True
        else:
            return False

    def set_outdoor_lighting(self, state):
        """
        Overrides the outdoor lighting set by other environment functions

        :param state: 0 force lights off, 1 force lights on
        :type time: int32
        :return: `True` if setting the time was successful, `False` otherwise
        :rtype: boolean

        """
        c = CommModularContainer()
        c.classID = self.ID_ENVIRONMENT_OUTDOORS
        c.actorNumber = 0
        c.actorFunction = self.FCN_OVERRIDE_OUTDOOR_LIGHTING
        c.payload = bytearray(struct.pack(">I", state))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_ENVIRONMENT_OUTDOORS, 0, self.FCN_OVERRIDE_OUTDOOR_LIGHTING_ACK)
            if (c == None):
                return False
            else:
                return True
        else:
            return False

    def set_weather_preset(self, weather_preset):
        """
        Set the weather conditions for an outdoor environment with a preset value

        :param weather_preset: A preset index (see defined constants for weather types)
        :type time: int32
        :return: `True` if setting the time was successful, `False` otherwise
        :rtype: boolean

        """
        c = CommModularContainer()
        c.classID = self.ID_ENVIRONMENT_OUTDOORS
        c.actorNumber = 0
        c.actorFunction = self.FCN_SET_WEATHER_PRESET
        c.payload = bytearray(struct.pack(">I", weather_preset))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_ENVIRONMENT_OUTDOORS, 0, self.FCN_SET_WEATHER_PRESET_ACK)
            if (c == None):
                return False
            else:
                return True
        else:
            return False
        














        from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import numpy as np
import math
import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsFlooring(QLabsActor):
    """ This is a deprecated class.  Please see the product-specific flooring classes."""

    ID_FLOORING = 10090
    """Class ID"""

    FLOORING_QCAR_MAP_LARGE = 0
    FLOORING_QCAR_MAP_SMALL = 1


    def __init__(self, qlabs, verbose=False):
        """ Constructor Method

        :param qlabs: A QuanserInteractiveLabs object
        :param verbose: (Optional) Print error information to the console.
        :type qlabs: object
        :type verbose: boolean
        """
        print('The class QLabsFlooring will be deprecated starting 2025. Please use QLabsQCarFlooring or QLabsQBotPlatformFlooring.')
        self._qlabs = qlabs
        self._qlabs = qlabs
        self._verbose = verbose
        self.classID = self.ID_FLOORING
        return
    












    from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import math
import struct
import cv2
import numpy as np


######################### MODULAR CONTAINER CLASS #########################

class QLabsFreeCamera(QLabsActor):
    """ This class supports the spawning and control of free movement cameras in QLabs open worlds."""

    ID_FREE_CAMERA = 170
    """Class ID"""
    FCN_FREE_CAMERA_POSSESS = 10
    FCN_FREE_CAMERA_POSSESS_ACK = 11
    FCN_FREE_CAMERA_SET_CAMERA_PROPERTIES = 12
    FCN_FREE_CAMERA_SET_CAMERA_PROPERTIES_ACK = 13
    FCN_FREE_CAMERA_SET_TRANSFORM = 14
    FCN_FREE_CAMERA_SET_TRANSFORM_ACK = 15
    FCN_FREE_CAMERA_SET_IMAGE_RESOLUTION = 90
    FCN_FREE_CAMERA_SET_IMAGE_RESOLUTION_RESPONSE = 91
    FCN_FREE_CAMERA_REQUEST_IMAGE = 100
    FCN_FREE_CAMERA_RESPONSE_IMAGE = 101

    """ The current actor number of this class to be addressed. This can be modified at any time. """


    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_FREE_CAMERA
       return

    def possess(self):
        """
        Possess (take control of) a camera in QLabs.

        :return: `True` if possessing the camera was successful, `False` otherwise
        :rtype: boolean

        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_FREE_CAMERA
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_FREE_CAMERA_POSSESS
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_FREE_CAMERA, self.actorNumber, self.FCN_FREE_CAMERA_POSSESS_ACK)
            if (c == None):
                return False
            else:
                return True
        else:
            return False

    def set_camera_properties(self, fieldOfView, depthOfField, aperture, focusDistance):
        """
        Sets the camera properties. When depthOfField is enabled, the camera will produce more realistic (and cinematic) results by adding some blur to the view at distances closer and further away from a given focal distance. For more blur, use a large aperture (small value) and a narrow field of view.

        :param fieldOfView: The field of view that the camera can see (range:5-150 degrees). When depthOfField is True, smaller values will increase focal blur at distances relative to the focusDistance.
        :param depthOfField: Enable or disable the depth of field visual effect
        :param aperture: The amount of light allowed into the camera sensor (range:2.0-22.0). Smaller values (larger apertures) increase the light and decrease the depth of field. This parameter is only active when depthOfField is True.
        :param focusDistance: The distance to the focal plane of the camera. (range:0.1-50.0 meters).  This parameter is only active when depthOfField is True.
        :type fieldOfView: int
        :type depthOfField: boolean
        :type aperture: float
        :type focusDistance: float
        :return: `True` if setting the camera properties was successful, `False` otherwise
        :rtype: boolean

        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_FREE_CAMERA
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_FREE_CAMERA_SET_CAMERA_PROPERTIES
        c.payload = bytearray(struct.pack(">fBff", fieldOfView, depthOfField, aperture, focusDistance))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_FREE_CAMERA, self.actorNumber, self.FCN_FREE_CAMERA_SET_CAMERA_PROPERTIES_ACK)
            if (c == None):
                return False
            else:
                return True
        else:
            return False

    def set_transform(self, location, rotation):
        """
        Change the location and rotation of a spawned camera in radians

        :param location: An array of floats for x, y and z coordinates
        :param rotation: An array of floats for the roll, pitch, yaw in radians
        :type location: array[3]
        :type rotation: array[3]
        :return: `True` if spawn was successful, `False` otherwise
        :rtype: boolean

        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_FREE_CAMERA
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_FREE_CAMERA_SET_TRANSFORM
        c.payload = bytearray(struct.pack(">ffffff", location[0], location[1], location[2], rotation[0], rotation[1], rotation[2]))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_FREE_CAMERA, self.actorNumber, self.FCN_FREE_CAMERA_SET_TRANSFORM_ACK)
            if (c == None):
                return False
            else:
                return True
        else:
            return False

    def set_transform_degrees(self, location, rotation):
        """
        Change the location and rotation of a spawned camera in degrees

        :param location: An array of floats for x, y and z coordinates
        :param rotation: An array of floats for the roll, pitch, yaw in degrees
        :type location: array[3]
        :type rotation: array[3]
        :return: `True` if spawn was successful, `False` otherwise
        :rtype: boolean

        """
        return self.set_transform(location, [rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi])

    def set_image_capture_resolution(self, width=640, height=480):
        """Change the default width and height of image resolution for capture

        :param width: Must be an even number. Default 640
        :param height: Must be an even number. Default 480
        :type width: uint32
        :type height: uint32
        :return: `True` if spawn was successful, `False` otherwise
        :rtype: boolean
        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_FREE_CAMERA
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_FREE_CAMERA_SET_IMAGE_RESOLUTION
        c.payload = bytearray(struct.pack(">II", width, height))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_FREE_CAMERA, self.actorNumber, self.FCN_FREE_CAMERA_SET_IMAGE_RESOLUTION_RESPONSE)
            if (c == None):
                return False
            else:
                return True
        else:
            return False

    def get_image(self):
        """Request an image from the camera actor. Note, set_image_capture_resolution must be set once per camera otherwise this method will fail.

        :return: Success, RGB image data
        :rtype: boolean, byte array[variable]
        """

        if (not self._is_actor_number_valid()):
            return False, None

        c = CommModularContainer()
        c.classID = self.ID_FREE_CAMERA
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_FREE_CAMERA_REQUEST_IMAGE
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_FREE_CAMERA, self.actorNumber, self.FCN_FREE_CAMERA_RESPONSE_IMAGE)
            if (c == None):
                return False, None

            data_size, = struct.unpack(">I", c.payload[0:4])

            jpg_buffer = cv2.imdecode(np.frombuffer(bytearray(c.payload[4:len(c.payload)]), dtype=np.uint8, count=-1, offset=0), 1)


            return True, jpg_buffer
        else:
            return False, None


















from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import numpy as np
import math
import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsGenericSensor(QLabsActor):
    """ This class is for spawning both generic distance sensing sensors."""

    ID_GENERIC_SENSOR = 220
    """Class ID"""

   
    FCN_GENERIC_SENSOR_SHOW_SENSOR = 10
    FCN_GENERIC_SENSOR_SHOW_SENSOR_ACK = 11
    FCN_GENERIC_SENSOR_SET_BEAM_SIZE = 12
    FCN_GENERIC_SENSOR_SET_BEAM_SIZE_ACK = 13
    FCN_GENERIC_SENSOR_TEST_BEAM_HIT = 14
    FCN_GENERIC_SENSOR_TEST_BEAM_HIT_RESPONSE = 15
    FCN_GENERIC_SENSOR_SET_TRANSFORM = 16
    FCN_GENERIC_SENSOR_SET_TRANSFORM_ACK = 17
    FCN_GENERIC_SENSOR_TEST_BEAM_HIT_WIDGET = 18
    FCN_GENERIC_SENSOR_TEST_BEAM_HIT_WIDGET_RESPONSE = 19
    

    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_GENERIC_SENSOR
       return

    def set_transform(self, location, rotation, scale, waitForConfirmation=True):
        """Sets the location, rotation in radians, and scale. If a sensor is parented to another actor then the location, rotation, and scale are relative to the parent actor.

        :param location: An array of floats for x, y and z coordinates in full-scale units. 
        :param rotation: An array of floats for the roll, pitch, and yaw in radians
        :param scale: An array of floats for the scale in the x, y, and z directions.
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type waitForConfirmation: boolean
        :return: True if successful or False otherwise
        :rtype: boolean
        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_GENERIC_SENSOR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_GENERIC_SENSOR_SET_TRANSFORM
        c.payload = bytearray(struct.pack(">fffffffff", location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], scale[0], scale[1], scale[2]))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_GENERIC_SENSOR, self.actorNumber, self.FCN_GENERIC_SENSOR_SET_TRANSFORM_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False

    def set_transform_degrees(self, location, rotation, scale, waitForConfirmation=True):
        """Sets the location, rotation in degrees, and scale. If a shape is parented to another actor then the location, rotation, and scale are relative to the parent actor.

        :param location: An array of floats for x, y and z coordinates in full-scale units. 
        :param rotation: An array of floats for the roll, pitch, and yaw in degrees
        :param scale: An array of floats for the scale in the x, y, and z directions.
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type waitForConfirmation: boolean
        :return: True if successful or False otherwise
        :rtype: boolean
        """

        return self.set_transform(location, [rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi], scale, waitForConfirmation)

    def show_sensor(self, showBeam=True, showOriginIcon=True, iconScale=0.1, waitForConfirmation=True):
        """Displays the beam and sensor location for debugging purposes.

        :param showBeam: Make the beam shape visible. Note this will be visible to all cameras and may affect depth sensors.
        :param showOriginIcon: Display a cone representing the projecting location of the beam.
        :param iconScale: A scale factor for the cone icon. A value of one will make a cone 1m x 1m.
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.
        :type showBeam: boolean
        :type showOriginIcon: boolean
        :type iconScale: float
        :type waitForConfirmation: boolean
        :return: True if successful or False otherwise
        :rtype: boolean
        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_GENERIC_SENSOR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_GENERIC_SENSOR_SHOW_SENSOR
        c.payload = bytearray(struct.pack(">??f", showBeam, showOriginIcon, iconScale))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_GENERIC_SENSOR, self.actorNumber, self.FCN_GENERIC_SENSOR_SHOW_SENSOR_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False

    def set_beam_size(self, startDistance=0.0, endDistance=1.0, heightOrRadius=0.1, width=0.1, waitForConfirmation=True):
        """Adjusts the beam shape parameters

        :param startDistance: Forward distance from the beam origin to start sensing
        :param endDistance: Maximum distance from the beam origin to end sensing
        :param heightOrRadius: For rectangular beam shapes the height. For round beam shapes, the radius. 
        :param width: For rectangular beam shapes the width. Ignored for round beam shapes.
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.
        :type startDistance: float
        :type endDistance: float
        :type heightOrRadius: float
        :type width: float
        :type waitForConfirmation: boolean
        :return: True if successful or False otherwise
        :rtype: boolean
        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_GENERIC_SENSOR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_GENERIC_SENSOR_SET_BEAM_SIZE
        c.payload = bytearray(struct.pack(">ffff", startDistance, endDistance, heightOrRadius, width))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_GENERIC_SENSOR, self.actorNumber, self.FCN_GENERIC_SENSOR_SET_BEAM_SIZE_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False

    def test_beam_hit(self):
        """Queries the beam to test if it hits something in its path.

        :return:
            - **status** - `True` communication was successful, `False` otherwise
            - **hit** - `True` if a hit occurred, `False` otherwise
            - **actorClass** - ID of the actor class.  If the value is 0 this indicates an actor which cannot be queried further or an environmental object.
            - **actorNumber** - If the actor is a valid actor class that can be queried, this will return the actor ID.
            - **distance** - Distance to the hit surface.
        :rtype: boolean, boolean, int32, int32, float
        """

        hit = False
        actorClass = 0
        actorNumber = 0
        distance = 0.0

        if (not self._is_actor_number_valid()):
            return False, hit, actorClass, actorNumber

        c = CommModularContainer()
        c.classID = self.ID_GENERIC_SENSOR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_GENERIC_SENSOR_TEST_BEAM_HIT
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
        
            c = self._qlabs.wait_for_container(self.ID_GENERIC_SENSOR, self.actorNumber, self.FCN_GENERIC_SENSOR_TEST_BEAM_HIT_RESPONSE)
            if (c == None):
                pass
            else:
                hit, actorClass, actorNumber, distance, = struct.unpack(">?IIf", c.payload[0:13])

                return True, hit, actorClass, actorNumber, distance

        
        
        return False, hit, actorClass, actorNumber, distance
           
    def test_beam_hit_widget(self):
        """Queries the beam to test if it hits a widget and if so returns the widget properties

        :return:
            - **status** - `True` communication was successful, `False` otherwise
            - **hit** - `True` if a hit occurred on a widget, `False` otherwise including non-widget actors
            - **distance** - Distance to the hit surface.
            - **IDTag** - User assigned byte identifier
            - **mass** - User assigned mass value
            - **properties** - User assigned custom property string
        :rtype: boolean, boolean, float, byte, float, string
        """

        hit = False
        distance = 0.0
        IDTag = 0
        mass = 0.0
        properties = ""
        

        if (not self._is_actor_number_valid()):
            return False, hit, distance, IDTag, mass, properties

        c = CommModularContainer()
        c.classID = self.ID_GENERIC_SENSOR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_GENERIC_SENSOR_TEST_BEAM_HIT_WIDGET
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
        
            c = self._qlabs.wait_for_container(self.ID_GENERIC_SENSOR, self.actorNumber, self.FCN_GENERIC_SENSOR_TEST_BEAM_HIT_WIDGET_RESPONSE)
            if (c == None):
                pass
            else:
                if (len(c.payload) >= 14):
                    hit, distance, IDTag, mass, properties_length, = struct.unpack(">?fBfI", c.payload[0:14])
                    
                    if properties_length > 0:
                        if (len(c.payload) == (14 + properties_length)):
                            properties = c.payload[14:(14+properties_length)].decode('utf-8')


                    return True, hit, distance, IDTag, mass, properties

        
        
        return False, hit, distance, IDTag, mass, properties
    
















    import numpy as np
import cv2

# Convert a color image to HSV, filter for hues within a certain width of center
# hueCenter - The hue being searched for
# hueWidth - The width of the range of hue values to accept
# hueGamut - The max hue value. Open CV uses 180, set this to 255 for normal hue values
def hue_threshold(image, hueCenter = 0, hueWidth = 20, hueGamut = 180):

    invert = False
    # Scale for incompatible hue gamut
    if hueGamut != 180:
        scale = 180/hueGamut
        hueCenter = scale * hueCenter
        hueWidth = scale * hueWidth

    # Set min and max hue values and general limits for saturation and value
    hMin = (hueCenter - (hueWidth/2)) % 180
    hMax = (hueCenter + (hueWidth/2)) % 180
    svMin = 64.0
    svMax = 255.0

    # Convert and threshold image
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bottomBounds = np.array([0.0,svMin,svMin])
    lowerBounds = np.array([hMin,svMin,svMin])
    upperBounds = np.array([hMax,svMax,svMax])
    topBounds = np.array([179.9,svMax,svMax])

    #Threshold for hue, special case for values that wrap past zero
    if hMin < hMax:
        binary = cv2.inRange(imageHSV, lowerBounds, upperBounds)
    else:
        binaryLow = cv2.inRange(imageHSV, bottomBounds, upperBounds)
        binaryHigh = cv2.inRange(imageHSV, lowerBounds, topBounds)
        binary = cv2.bitwise_or(binaryLow, binaryHigh)
	
    return binary

# Crop an image to extract a region of interest using x and y ranges of pixels
def crop_rect(image, xRange = [0,0], yRange = [0,0]):
    if xRange[1] > xRange[0] and yRange[1] > yRange[0]:
        imageCrop = image[yRange[0]:yRange[1], xRange[0]:xRange[1]]
    else:
        imageCrop = image

    return imageCrop

# Draw a rectangle over an image to indicate the region of interest
def show_ROI(image, xRange = [0,0], yRange = [0,0]):

    image = cv2.rectangle(image, (xRange[0], yRange[0]), (xRange[1], yRange[1]), (255, 128, 128), 2)

    return image

# Draw the rectangle ROI with a vertical line indicating the target center
def show_ROI_target(image, xRange = [0,0], yRange = [0,0], targ = -1):
    
    image = show_ROI(image, xRange, yRange)
    
    tRange = [yRange[0] - 10, yRange[1] + 10]

    if targ == -1:
        image = cv2.line(image, (320, tRange[0]), (320, tRange[1]), (0, 0, 255), 4)
    else:
        tX = int(round(targ))
        image = cv2.line(image, (tX, tRange[0]), (tX, tRange[1]), (128, 255, 128), 4)

    return image

# Find the center of a line in a thresholded image
def extract_line_ctr(image):

    center = -1

    # Average the pixels in each column and find the maximum value
    columnVals = np.mean(image, axis = 0)
    maxCol = np.amax(columnVals)

    #Check for lost line and return the average x position of max values. Else, return -1
    if maxCol > 64:
        center = np.mean(np.argwhere(columnVals == maxCol))

    return center



















from qvl.qlabs import CommModularContainer
from qvl.character import QLabsCharacter
import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsPerson(QLabsCharacter):
    """ This class implements spawning and AI navigation of the environment for human pedestrians."""

    ID_PERSON = 10030

    STANDING = 0
    """ Speed constant for the move_to method. """
    WALK = 1.2
    """ Speed constant for the move_to method. """
    JOG = 3.6
    """ Speed constant for the move_to method. """
    RUN = 6.0
    """ Speed constant for the move_to method. """


    def __init__(self, qlabs, verbose=False):
       """ Constructor method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_PERSON
       return






















from qvl.qlabs import QuanserInteractiveLabs, CommModularContainer
from quanser.common import GenericError
from qvl.actor import QLabsActor
import math

import sys
import struct
import os

sys.path.append('../Common/')

######################### MODULAR CONTAINER CLASS #########################

class QLabsQArm(QLabsActor):

    ID_QARM = 10

    # Initialize class
    def __init__(self, qlabs, verbose=False):
        """ Constructor Method

        :param qlabs: A QuanserInteractiveLabs object
        :param verbose: (Optional) Print error information to the console.
        :type qlabs: object
        :type verbose: boolean
        """

        self._qlabs = qlabs
        self._verbose = verbose
        self.classID = self.ID_QARM
        return


















from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

from quanser.common import GenericError
import math
import os
import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsQbot(QLabsActor):
    # This is a deprecated class.  Please see the product-specific qbot classes.
    
    ID_QBOT = 20

    FCN_QBOT_COMMAND_AND_REQUEST_STATE = 10
    FCN_QBOT_COMMAND_AND_REQUEST_STATE_RESPONSE = 11
    FCN_QBOT_POSSESS = 20
    FCN_QBOT_POSSESS_ACK = 21


    VIEWPOINT_RGB = 0
    VIEWPOINT_DEPTH = 1
    VIEWPOINT_TRAILING = 2

    # Initialize class
    def __init__(self, qlabs, verbose=False):
        self._qlabs = qlabs
        self._verbose = verbose
        self.classID = self.ID_QBOT
        print('The class QLabsQbot will be deprecated starting 2025. Please use product specific classes (QLabsQBot2e/QLabsQBot3/QLabsQBotPlatform).')
        
        return
   
    def possess(self, camera):
        c = CommModularContainer()
        c.classID = self.ID_QBOT
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QBOT_POSSESS
        c.payload = bytearray(struct.pack(">B", camera))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()
            
        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QBOT, self.actorNumber, self.FCN_QBOT_POSSESS_ACK)
            if (c == None):
                return False
            else:
                return True
        else:
            return False            




















from qvl.qlabs import QuanserInteractiveLabs, CommModularContainer
from quanser.common import GenericError
import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsQBotHopper:


    ID_QBOT_DUMPING_MECHANISM = 111

    FCN_QBOT_DUMPING_MECHANISM_COMMAND = 10
    FCN_QBOT_DUMPING_MECHANISM_COMMAND_ACK = 12


    VIEWPOINT_RGB = 0
    VIEWPOINT_DEPTH = 1
    VIEWPOINT_TRAILING = 2

    # Initialize class
    def __init__(self):

       return

    def spawn(self, qlabs, actorNumber, location, rotation, configuration=0, waitForConfirmation=True):
        return qlabs.spawn(actorNumber, self.ID_QBOT_DUMPING_MECHANISM, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], 1.0, 1.0, 1.0, configuration, waitForConfirmation)

    def spawn_and_parent_with_relative_transform(self, qlabs, actorNumber, location, rotation, parentClass, parentActorNumber, parentComponent, waitForConfirmation=True):
        return qlabs.spawn_and_parent_with_relative_transform(actorNumber, self.ID_QBOT_DUMPING_MECHANISM, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], 1.0, 1.0, 1.0, 0, parentClass, parentActorNumber, parentComponent, waitForConfirmation)

    def spawn_degrees(self, qlabs, actorNumber, location, rotation, configuration=0, waitForConfirmation=True):

        return qlabs.spawn(actorNumber, self.ID_QBOT_DUMPING_MECHANISM, location[0], location[1], location[2], rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi, 1.0, 1.0, 1.0, configuration, waitForConfirmation)

    def command(self, qlabs, actorNumber, angle):
        c = CommModularContainer()
        c.classID = self.ID_QBOT_DUMPING_MECHANISM
        c.actorNumber = actorNumber
        c.actorFunction = self.FCN_QBOT_DUMPING_MECHANISM_COMMAND
        c.payload = bytearray(struct.pack(">f", angle))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        qlabs.flush_receive()

        if (qlabs.send_container(c)):
            c = qlabs.wait_for_container(self.ID_QBOT_DUMPING_MECHANISM, actorNumber, self.FCN_QBOT_DUMPING_MECHANISM_COMMAND_ACK)

            return True
        else:
            return False

    def command_degrees(self, qlabs, actorNumber, angle):
        self.command(qlabs, actorNumber, angle/180*math.pi)















from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import cv2
import numpy as np
import math
import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsQBotPlatform(QLabsActor):
    """This class is for spawning QBotPlatforms."""

    ID_QBOT_PLATFORM = 23

    FCN_QBOT_PLATFORM_COMMAND_AND_REQUEST_STATE = 10
    FCN_QBOT_PLATFORM_COMMAND_AND_REQUEST_STATE_RESPONSE = 11
    FCN_QBOT_PLATFORM_SET_TRANSFORM = 14
    FCN_QBOT_PLATFORM_SET_TRANSFORM_RESPONSE = 15
    FCN_QBOT_PLATFORM_POSSESS = 20
    FCN_QBOT_PLATFORM_POSSESS_ACK = 21
    FCN_QBOT_PLATFORM_IMAGE_REQUEST = 100
    FCN_QBOT_PLATFORM_IMAGE_RESPONSE = 101
    FCN_QBOT_PLATFORM_LIDAR_DATA_REQUEST = 120
    FCN_QBOT_PLATFORM_LIDAR_DATA_RESPONSE = 121
    

    VIEWPOINT_RGB = 0
    VIEWPOINT_DEPTH = 1
    VIEWPOINT_DOWNWARD = 2
    VIEWPOINT_TRAILING = 3
 
    CAMERA_RGB = 0
    CAMERA_DEPTH = 1
    CAMERA_DOWNWARD = 2
    

    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_QBOT_PLATFORM
       return

    def possess(self, camera):
        """
        Possess (take control of) a QBot in QLabs with the selected camera.

        :param camera: Pre-defined camera constant. See CAMERA constants for available options. Default is the trailing camera.
        :type camera: uint32
        :return:
            - **status** - `True` if possessing the camera was successful, `False` otherwise
        :rtype: boolean

        """
        c = CommModularContainer()
        c.classID = self.ID_QBOT_PLATFORM
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QBOT_PLATFORM_POSSESS
        c.payload = bytearray(struct.pack(">B", camera))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QBOT_PLATFORM, self.actorNumber, self.FCN_QBOT_PLATFORM_POSSESS_ACK)
            if (c == None):
                return False
            else:
                return True
        else:
            return False

    def command_and_request_state(self, rightWheelSpeed, leftWheelSpeed, leftLED=[1,0,0], rightLED=[1,0,0]):
        """Sets the wheel speeds and LED colors.

        :param rightWheelSpeed: Speed in m/s
        :param leftWheelSpeed: Speed in m/s
        :param leftLED: Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :param rightLED: Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        

        :type rightWheelSpeed: float
        :type leftWheelSpeed: float
        :type leftLED: float array[3]
        :type rightLED: float array[3]

        :return:
            - **status** - `True` if successful, `False` otherwise
            - **location** - world location in m
            - **forward vector** - unit scale vector
            - **up vector** - unit scale vector
            - **front bumper hit** - true if in contact with a collision object, False otherwise
            - **left bumper hit** - true if in contact with a collision object, False otherwise
            - **right bumper hit** - true if in contact with a collision object, False otherwise
            - **gyro** - turn rate in rad/s
            - **heading** - angle in rad
            - **encoder left** - in counts
            - **encoder right** - in counts

        :rtype: boolean, float array[3], float array[3], float array[3], boolean, boolean, boolean, float, float, uint32, uint32


        """
        c = CommModularContainer()
        c.classID = self.ID_QBOT_PLATFORM
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QBOT_PLATFORM_COMMAND_AND_REQUEST_STATE
        c.payload = bytearray(struct.pack(">ffffffff", rightWheelSpeed, leftWheelSpeed, leftLED[0], leftLED[1], leftLED[2], rightLED[0], rightLED[1], rightLED[2]))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        location = [0,0,0]
        forward = [0,0,0]
        up = [0,0,0]
        frontHit = False
        leftHit = False
        rightHit = False
        gyro = 0
        heading = 0
        encoderLeft = 0
        encoderRight = 0


        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QBOT_PLATFORM, self.actorNumber, self.FCN_QBOT_PLATFORM_COMMAND_AND_REQUEST_STATE_RESPONSE)

            if (c == None):
                return False, location, forward, up, frontHit, leftHit, rightHit, gyro, heading, encoderLeft, encoderRight

            if len(c.payload) == 55:
                location[0], location[1], location[2], forward[0], forward[1], forward[2], up[0], up[1], up[2], frontHit, leftHit, rightHit, gyro, heading, encoderLeft, encoderRight, = struct.unpack(">fffffffff???ffII", c.payload[0:55])
                return True, location, forward, up, frontHit, leftHit, rightHit, gyro, heading, encoderLeft, encoderRight
            else:
                return False, location, forward, up, frontHit, leftHit, rightHit, gyro, heading, encoderLeft, encoderRight
        else:
            return False, location, forward, up, frontHit, leftHit, rightHit, gyro, heading, encoderLeft, encoderRight
               
    def set_transform(self, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], leftLED=[1,0,0], rightLED=[1,0,0], enableDynamics=True, waitForConfirmation=True):
        """Sets the transform, LED colors, and enabling/disabling of physics dynamics

        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used.
        :param leftLED: (Optional) Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :param rightLED: (Optional) Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :param enableDynamics: (default True) Enables or disables gravity for set transform requests.
        :param waitForConfirmation: (Optional) Wait for confirmation before proceeding. This makes the method a blocking operation. NOTE: Return data will only be valid if waitForConfirmation is True.
        
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type leftLED: float array[3]
        :type rightLED: float array[3
        :type enableDynamics: boolean
        :type waitForConfirmation: boolean

        :return:
            - **status** - True if successful or False otherwise
            - **location** - world location in m
            - **forward vector** - unit scale vector
            - **up vector** - unit scale vector

        :rtype: boolean, float array[3], float array[3], float array[3]


        """
        c = CommModularContainer()
        c.classID = self.ID_QBOT_PLATFORM
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QBOT_PLATFORM_SET_TRANSFORM
        c.payload = bytearray(struct.pack(">fffffffffffffffB", location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], scale[0], scale[1], scale[2], leftLED[0], leftLED[1], leftLED[2], rightLED[0], rightLED[1], rightLED[2], enableDynamics))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        location = [0,0,0]
        forward = [0,0,0]
        up = [0,0,0]
        

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_QBOT_PLATFORM, self.actorNumber, self.FCN_QBOT_PLATFORM_SET_TRANSFORM_RESPONSE)

                if (c == None):
                    return False, location, forward, up

                if len(c.payload) == 36:

                    location[0], location[1], location[2], forward[0], forward[1], forward[2], up[0], up[1], up[2], = struct.unpack(">fffffffff", c.payload[0:36])
                    return True, location, forward, up
                else:
                    return False, location, forward, up
            else:
                return True, location, forward, up
        else:
            return False, location, forward, up

    def set_transform_degrees(self, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], leftLED=[1,0,0], rightLED=[1,0,0], enableDynamics=True, waitForConfirmation=True):
        """Sets the transform, LED colors, and enabling/disabling of physics dynamics

        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in degrees
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used.
        :param leftLED: (Optional) Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :param rightLED: (Optional) Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :param enableDynamics: (default True) Enables or disables gravity for set transform requests.
        :param waitForConfirmation: (Optional) Wait for confirmation before proceeding. This makes the method a blocking operation. NOTE: Return data will only be valid if waitForConfirmation is True.
        
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type leftLED: float array[3]
        :type rightLED: float array[3
        :type enableDynamics: boolean
        :type waitForConfirmation: boolean

        :return:
            - **status** - True if successful or False otherwise
            - **location** - world location in m
            - **forward vector** - unit scale vector
            - **up vector** - unit scale vector

        :rtype: boolean, float array[3], float array[3], float array[3]


        """
        
        
        success, location, forward, up = self.set_transform(location, [rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi], scale, leftLED, rightLED, enableDynamics, waitForConfirmation)

        return success, location, forward, up
        
    def get_image(self, camera):
        """
        Request a JPG image from the QBot camera.

        :param camera: Camera number to view from.
        
        :type camera: byte

        :return:
            - **status** - `True` and image data if successful, `False` and empty otherwise
            - **imageData** - Image in a JPG format
        :rtype: boolean, byte array with jpg data

        """

        if (not self._is_actor_number_valid()):
            return False, None

        c = CommModularContainer()
        c.classID = self.ID_QBOT_PLATFORM
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QBOT_PLATFORM_IMAGE_REQUEST
        c.payload = bytearray(struct.pack(">B", camera))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QBOT_PLATFORM, self.actorNumber, self.FCN_QBOT_PLATFORM_IMAGE_RESPONSE)

            if (c == None):
                return False, None


            imageData = cv2.imdecode(np.frombuffer(bytearray(c.payload[8:len(c.payload)]), dtype=np.uint8, count=-1, offset=0), 1)


            return True, imageData
        else:
            return False, None

    def get_lidar(self, samplePoints=400):
        """
        Request LIDAR data from a QBotPlatform.

        :param samplePoints: (Optional) Change the number of points per revolution of the LIDAR.
        :type samplePoints: uint32
        :return: `True`, angles in radians, and distances in m if successful, `False`, none, and none otherwise
        :rtype: boolean, float array, float array

        """

        if (not self._is_actor_number_valid()):
            if (self._verbose):
                print('actorNumber object variable None. Use a spawn function to assign an actor or manually assign the actorNumber variable.')
            return False, None, None
            

        LIDAR_SAMPLES = 4096
        LIDAR_RANGE = 8

        # The LIDAR is simulated by using 4 orthogonal virtual cameras that are 1 pixel high. The
        # lens distortion of these cameras must be removed to accurately calculate the XY position
        # of the depth samples.
        quarter_angle = np.linspace(0, 45, int(LIDAR_SAMPLES/8))
        lens_curve = -0.0077*quarter_angle*quarter_angle + 1.3506*quarter_angle
        lens_curve_rad = lens_curve/180*np.pi

        angles = np.concatenate((np.pi*4/2-1*np.flip(lens_curve_rad), \
                                 lens_curve_rad, \
                                 (np.pi/2 - 1*np.flip(lens_curve_rad)), \
                                 (np.pi/2 + lens_curve_rad), \
                                 (np.pi - 1*np.flip(lens_curve_rad)), \
                                 (np.pi + lens_curve_rad), \
                                 (np.pi*3/2 - 1*np.flip(lens_curve_rad)), \
                                 (np.pi*3/2 + lens_curve_rad)))



        c = CommModularContainer()
        c.classID = self.ID_QBOT_PLATFORM
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QBOT_PLATFORM_LIDAR_DATA_REQUEST
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QBOT_PLATFORM, self.actorNumber, self.FCN_QBOT_PLATFORM_LIDAR_DATA_RESPONSE)

            if (c == None):
                if (self._verbose):
                    print('Failed to receive return container.')
                return False, None, None

            if ((len(c.payload)-4)/2 != LIDAR_SAMPLES):
                
                if (self._verbose):
                    print("Received {} bytes, expected {}".format(len(c.payload), LIDAR_SAMPLES*2))

                return False, None, None

            distance = np.linspace(0,0,LIDAR_SAMPLES)

            for count in range(LIDAR_SAMPLES-1):
                # clamp any value at 65535 to 0
                raw_value = ((c.payload[4+count*2] * 256 + c.payload[5+count*2] )) %65535

                # scale to LIDAR range
                distance[count] = (raw_value/65535)*LIDAR_RANGE


            # Resample the data using a linear radial distribution to the desired number of points
            # and realign the first index to be 0 (forward)
            sampled_angles = np.linspace(0,2*np.pi, num=samplePoints, endpoint=False)
            sampled_distance = np.linspace(0,0, samplePoints)

            index_raw = 512
            for count in range(samplePoints):
                while (angles[index_raw] < sampled_angles[count]):
                    index_raw = (index_raw + 1) % 4096


                if index_raw != 0:
                    if (angles[index_raw]-angles[index_raw-1]) == 0:
                        sampled_distance[count] = distance[index_raw]
                    else:
                        sampled_distance[count] = (distance[index_raw]-distance[index_raw-1])*(sampled_angles[count]-angles[index_raw-1])/(angles[index_raw]-angles[index_raw-1]) + distance[index_raw-1]


                else:
                    sampled_distance[count] = distance[index_raw]


            return True, sampled_angles, sampled_distance
        else:
            if (self._verbose):
                print('Communications request for LIDAR data failed.')
            return False, None, None
        





















        from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import numpy as np
import math
import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsQBotPlatformFlooring(QLabsActor):
    """ This class is for spawning QBot Platform floor mats."""

    ID_FLOORING = 10091
    """Class ID"""

    FLOORING_QBOT_PLATFORM_0 = 0
    """See configurations"""
    FLOORING_QBOT_PLATFORM_1 = 1
    """See configurations"""
    FLOORING_QBOT_PLATFORM_2 = 2
    """See configurations"""
    FLOORING_QBOT_PLATFORM_3 = 3
    """See configurations"""
    FLOORING_QBOT_PLATFORM_4 = 4
    """See configurations"""
    FLOORING_QBOT_PLATFORM_5 = 5
    """See configurations"""


    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_FLOORING
       return


















from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

from quanser.common import GenericError
import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsQBot2e(QLabsActor):


    ID_QBOT2e = 20

    FCN_QBOT_COMMAND_AND_REQUEST_STATE = 10
    FCN_QBOT_COMMAND_AND_REQUEST_STATE_RESPONSE = 11
    FCN_QBOT_POSSESS = 20
    FCN_QBOT_POSSESS_ACK = 21


    VIEWPOINT_RGB = 0
    VIEWPOINT_DEPTH = 1
    VIEWPOINT_TRAILING = 2

    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_QBOT2e
       return


    def possess(self, qlabs, actorNumber, camera):
        c = CommModularContainer()
        c.classID = self.ID_QBOT2e
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QBOT_POSSESS
        c.payload = bytearray(struct.pack(">B", camera))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QBOT2e, self.actorNumber, self.FCN_QBOT_POSSESS_ACK)
            if (c == None):
                return False
            else:
                return True
        else:
            return False

    def command_and_request_state(self, rightWheelSpeed, leftWheelSpeed):
        c = CommModularContainer()
        c.classID = self.ID_QBOT2e
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QBOT_COMMAND_AND_REQUEST_STATE
        c.payload = bytearray(struct.pack(">ff", rightWheelSpeed, leftWheelSpeed))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QBOT2e, self.actorNumber, self.FCN_QBOT_COMMAND_AND_REQUEST_STATE_RESPONSE)

            return True
        else:
            return False





















from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import cv2
import numpy as np
import math
import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsQBot3(QLabsActor):


    ID_QBOT3 = 22

    FCN_QBOT3_COMMAND_AND_REQUEST_STATE = 10
    FCN_QBOT3_COMMAND_AND_REQUEST_STATE_RESPONSE = 11
    FCN_QBOT3_POSSESS = 20
    FCN_QBOT3_POSSESS_ACK = 21
    FCN_QBOT3_RGB_REQUEST = 100
    FCN_QBOT3_RGB_RESPONSE = 101
    FCN_QBOT3_DEPTH_REQUEST = 110
    FCN_QBOT3_DEPTH_RESPONSE = 111


    VIEWPOINT_RGB = 0
    VIEWPOINT_DEPTH = 1
    VIEWPOINT_TRAILING = 2

    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_QBOT3
       return

    def possess(self, camera):
        """
        Possess (take control of) a QBot in QLabs with the selected camera.

        :param camera: Pre-defined camera constant. See CAMERA constants for available options. Default is the trailing camera.
        :type camera: uint32
        :return:
            - **status** - `True` if possessing the camera was successful, `False` otherwise
        :rtype: boolean

        """
        c = CommModularContainer()
        c.classID = self.ID_QBOT3
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QBOT3_POSSESS
        c.payload = bytearray(struct.pack(">B", camera))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QBOT3, self.actorNumber, self.FCN_QBOT3_POSSESS_ACK)
            if (c == None):
                return False
            else:
                return True
        else:
            return False

    def command_and_request_state(self, rightWheelSpeed, leftWheelSpeed):
        """Sets the velocity, turn angle in radians, and other car properties.

        :param forward: Speed in m/s of a full-scale car. Multiply physical QCar speeds by 10 to get full scale speeds.
        :param turn: Turn angle in radians. Positive values turn right.

        :type actorNumber: float
        :type turn: float

        :return:
            - **status** - `True` if successful, `False` otherwise
            - **location** - world location in m
            - **forward vector** - unit scale vector
            - **up vector** - unit scale vector
            - **front bumper hit** - true if in contact with a collision object, False otherwise
            - **left bumper hit** - true if in contact with a collision object, False otherwise
            - **right bumper hit** - true if in contact with a collision object, False otherwise
            - **gyro** - turn rate in rad/s
            - **heading** - angle in rad
            - **encoder left** - in counts
            - **encoder right** - in counts

        :rtype: boolean, float array[3], float array[3], float array[3], boolean, boolean, boolean, float, float, uint32, uint32


        """
        c = CommModularContainer()
        c.classID = self.ID_QBOT3
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QBOT3_COMMAND_AND_REQUEST_STATE
        c.payload = bytearray(struct.pack(">ff", rightWheelSpeed, leftWheelSpeed))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        location = [0,0,0]
        forward = [0,0,0]
        up = [0,0,0]
        frontHit = False
        leftHit = False
        rightHit = False
        gyro = 0
        heading = 0
        encoderLeft = 0
        encoderRight = 0


        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QBOT3, self.actorNumber, self.FCN_QBOT3_COMMAND_AND_REQUEST_STATE_RESPONSE)

            if (c == None):
                return False, location, forward, up, frontHit, leftHit, rightHit, gyro, heading, encoderLeft, encoderRight

            if len(c.payload) == 55:
                location[0], location[1], location[2], forward[0], forward[1], forward[2], up[0], up[1], up[2], frontHit, leftHit, rightHit, gyro, heading, encoderLeft, encoderRight, = struct.unpack(">fffffffff???ffII", c.payload[0:55])
                return True, location, forward, up, frontHit, leftHit, rightHit, gyro, heading, encoderLeft, encoderRight
            else:
                return False, location, forward, up, frontHit, leftHit, rightHit, gyro, heading, encoderLeft, encoderRight
        else:
            return False, location, forward, up, frontHit, leftHit, rightHit, gyro, heading, encoderLeft, encoderRight

    def get_image_rgb(self):
        """
        Request a JPG image from the QBot camera.

        :return:
            - **status** - `True` and image data if successful, `False` and empty otherwise
            - **imageData** - Image in a JPG format
        :rtype: boolean, byte array with jpg data

        """

        if (not self._is_actor_number_valid()):
            return False, None

        c = CommModularContainer()
        c.classID = self.ID_QBOT3
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QBOT3_RGB_REQUEST
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QBOT3, self.actorNumber, self.FCN_QBOT3_RGB_RESPONSE)

            if (c == None):
                return False, None


            jpg_buffer = cv2.imdecode(np.frombuffer(bytearray(c.payload[4:len(c.payload)]), dtype=np.uint8, count=-1, offset=0), 1)


            return True, jpg_buffer
        else:
            return False, None

    def get_image_depth(self):
        """
        Request a JPG image from the QBot camera.

        :return:
            - **status** - `True` and image data if successful, `False` and empty otherwise
            - **imageData** - Image in a JPG format
        :rtype: boolean, byte array with jpg data

        """

        if (not self._is_actor_number_valid()):
            return False, None

        c = CommModularContainer()
        c.classID = self.ID_QBOT3
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QBOT3_DEPTH_REQUEST
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QBOT3, self.actorNumber, self.FCN_QBOT3_DEPTH_RESPONSE)

            if (c == None):
                return False, None


            jpg_buffer = cv2.imdecode(np.frombuffer(bytearray(c.payload[4:len(c.payload)]), dtype=np.uint8, count=-1, offset=0), 1)


            return True, jpg_buffer
        else:
            return False, None
        





















        from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import math
import struct
import cv2
import numpy as np


######################### MODULAR CONTAINER CLASS #########################

class QLabsQCar(QLabsActor):
    """This class is for spawning QCars."""


    ID_QCAR = 160
    """ Class ID """
    FCN_QCAR_SET_VELOCITY_AND_REQUEST_STATE = 10
    FCN_QCAR_VELOCITY_STATE_RESPONSE = 11
    FCN_QCAR_SET_TRANSFORM_AND_REQUEST_STATE = 12
    FCN_QCAR_TRANSFORM_STATE_RESPONSE = 13
    FCN_QCAR_POSSESS = 20
    FCN_QCAR_POSSESS_ACK = 21
    FCN_QCAR_GHOST_MODE = 22
    FCN_QCAR_GHOST_MODE_ACK = 23
    FCN_QCAR_CAMERA_DATA_REQUEST = 100
    FCN_QCAR_CAMERA_DATA_RESPONSE = 101
    FCN_QCAR_LIDAR_DATA_REQUEST = 110
    FCN_QCAR_LIDAR_DATA_RESPONSE = 111


    CAMERA_CSI_RIGHT = 0
    #Image capture resolution: 820x410
    CAMERA_CSI_BACK = 1
    #Image capture resolution: 820x410
    CAMERA_CSI_LEFT = 2
    #Image capture resolution: 820x410
    CAMERA_CSI_FRONT = 3
    #Image capture resolution: 820x410
    CAMERA_RGB = 4
    #Image capture resolution: 640x480
    CAMERA_DEPTH = 5
    #Image capture resolution: 640x480
    CAMERA_OVERHEAD = 6
    CAMERA_TRAILING = 7
    #Note: The mouse scroll wheel can be used to zoom in and out in this mode. """

    _sensor_scaling = 1


    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_QCAR
       return

    def spawn_id(self, actorNumber, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, waitForConfirmation=True):
        """Spawns a new QCar actor.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type actorNumber: uint32
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 2 actor number not available or already in use, 3 unknown error, -1 communications error
        :rtype: int32

        """

        self._sensor_scaling = scale[0]
        return super().spawn_id(actorNumber, location, rotation, scale, configuration, waitForConfirmation)
        
    def spawn_id_degrees(self, actorNumber, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, waitForConfirmation=True):
        """Spawns a new QCar actor.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type actorNumber: uint32
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 2 actor number not available or already in use, 3 unknown error, -1 communications error
        :rtype: int32

        """
        
        self._sensor_scaling = scale[0]
        return super().spawn_id_degrees(actorNumber, location, rotation, scale, configuration, waitForConfirmation)
        
    def spawn(self, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, waitForConfirmation=True):
        """Spawns a new QCar actor with the next available actor number within this class.

        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred. Note that if this is False, the returned actor number will be invalid.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 3 unknown error, -1 communications error.
            - **actorNumber** - An actor number to use for future references.
        :rtype: int32, int32

        """   

        self._sensor_scaling = scale[0]
        return super().spawn(location, rotation, scale, configuration, waitForConfirmation)
               
    def spawn_degrees(self, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, waitForConfirmation=True):
        """Spawns a new QCar actor with the next available actor number within this class.

        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in degrees
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred. Note that if this is False, the returned actor number will be invalid.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 3 unknown error, -1 communications error.
            - **actorNumber** - An actor number to use for future references.
        :rtype: int32, int32

        """
        self._sensor_scaling = scale[0]
        return super().spawn_degrees(location, rotation, scale, configuration, waitForConfirmation)
        
    def spawn_id_and_parent_with_relative_transform(self, actorNumber, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, parentClassID=0, parentActorNumber=0, parentComponent=0, waitForConfirmation=True):
        """Spawns a new QCar actor relative to an existing actor and creates a kinematic relationship.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param parentClassID: See the ID variables in the respective library classes for the class identifier
        :param parentActorNumber: User defined unique identifier for the class actor in QLabs
        :param parentComponent: `0` for the origin of the parent actor, see the parent class for additional reference frame options
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type actorNumber: uint32
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type parentClassID: uint32
        :type parentActorNumber: uint32
        :type parentComponent: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 2 actor number not available or already in use, 3 cannot find the parent actor, 4 unknown error, -1 communications error
        :rtype: int32

        """

        self._sensor_scaling = scale[0]
        return super().spawn_id_and_parent_with_relative_transform(actorNumber, location, rotation, scale, configuration, parentClassID, parentActorNumber, parentComponent, waitForConfirmation)

    def spawn_id_and_parent_with_relative_transform_degrees(self, actorNumber, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, parentClassID=0, parentActorNumber=0, parentComponent=0, waitForConfirmation=True):
        """Spawns a new QCar actor relative to an existing actor and creates a kinematic relationship.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in degrees
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param parentClassID: See the ID variables in the respective library classes for the class identifier
        :param parentActorNumber: User defined unique identifier for the class actor in QLabs
        :param parentComponent: `0` for the origin of the parent actor, see the parent class for additional reference frame options
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type actorNumber: uint32
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type parentClassID: uint32
        :type parentActorNumber: uint32
        :type parentComponent: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 2 actor number not available or already in use, 3 cannot find the parent actor, 4 unknown error, -1 communications error
        :rtype: int32

        """

        self._sensor_scaling = scale[0]
        return super().spawn_id_and_parent_with_relative_transform_degrees(actorNumber, location, rotation, scale, configuration, parentClassID, parentActorNumber, parentComponent, waitForConfirmation)

    def set_transform_and_request_state(self, location, rotation, enableDynamics, headlights, leftTurnSignal, rightTurnSignal, brakeSignal, reverseSignal, waitForConfirmation=True):
        """Sets the location, rotation, and other car properties. Note that setting the location ignores collisions so ensure that the location is free of obstacles that may trap the actor if it is subsequently used in a dynamic mode. This transform can also be used to "playback" previously recorded position data without the need for a full dynamic model.

        :param location: An array of floats for x, y and z coordinates in full-scale units. Multiply physical QCar locations by 10 to get full scale locations.
        :param rotation: An array of floats for the roll, pitch, and yaw in radians
        :param enableDynamics: (default True) Enables or disables gravity for set transform requests.
        :param headlights: Enable the headlights
        :param leftTurnSignal: Enable the left turn signal
        :param rightTurnSignal: Enable the right turn signal
        :param brakeSignal: Enable the brake lights (does not affect the motion of the vehicle)
        :param reverseSignal: Play a honking sound
        :param waitForConfirmation: (Optional) Wait for confirmation before proceeding. This makes the method a blocking operation. NOTE: Return data will only be valid if waitForConfirmation is True.
        :type location: float array[3]
        :type rotation: float array[3]
        :type enableDynamics: boolean
        :type headlights: boolean
        :type leftTurnSignal: boolean
        :type rightTurnSignal: boolean
        :type brakeSignal: boolean
        :type reverseSignal: boolean
        :type waitForConfirmation: boolean
        :return:
            - **status** - True if successful or False otherwise
            - **location** - in full scale
            - **rotation** - in radians
            - **forward vector** - unit scale vector
            - **up vector** - unit scale vector
            - **front bumper hit** - True if in contact with a collision object, False otherwise
            - **rear bumper hit** - True if in contact with a collision object, False otherwise
        :rtype: boolean, float array[3], float array[3], float array[3], float array[3], boolean, boolean

        """
        if (not self._is_actor_number_valid()):
            return False, [0,0,0], [0,0,0], [0,0,0], [0,0,0], False, False

        c = CommModularContainer()
        c.classID = self.ID_QCAR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QCAR_SET_TRANSFORM_AND_REQUEST_STATE
        c.payload = bytearray(struct.pack(">ffffffBBBBBB", location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], enableDynamics, headlights, leftTurnSignal, rightTurnSignal, brakeSignal, reverseSignal))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)


        location = [0,0,0]
        rotation = [0,0,0]
        forward_vector = [0,0,0]
        up_vector = [0,0,0]
        frontHit = False
        rearHit = False

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_QCAR, self.actorNumber, self.FCN_QCAR_TRANSFORM_STATE_RESPONSE)

                if (c == None):
                    return False, location, rotation, forward_vector, up_vector, frontHit, rearHit

                if len(c.payload) == 50:

                    location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], forward_vector[0], forward_vector[1], forward_vector[2], up_vector[0], up_vector[1], up_vector[2], frontHit, rearHit, = struct.unpack(">ffffffffffff??", c.payload[0:50])
                    return True, location, rotation, forward_vector, up_vector, frontHit, rearHit
                else:
                    return False, location, rotation, forward_vector, up_vector, frontHit, rearHit
            else:
                return True, location, rotation, forward_vector, up_vector, frontHit, rearHit
        else:
            return False, location, rotation, forward_vector, up_vector, frontHit, rearHit

    def set_transform_and_request_state_degrees(self, location, rotation, enableDynamics, headlights, leftTurnSignal, rightTurnSignal, brakeSignal, reverseSignal, waitForConfirmation=True):
        """Sets the location, rotation, and other car properties. Note that setting the location ignores collisions so ensure that the location is free of obstacles that may trap the actor if it is subsequently used in a dynamic mode. This transform can also be used to "playback" previously recorded position data without the need for a full dynamic model.

        :param location: An array of floats for x, y and z coordinates in full-scale units. Multiply physical QCar locations by 10 to get full scale locations.
        :param rotation: An array of floats for the roll, pitch, and yaw in degrees.
        :param enableDynamics: (default True) Enables or disables gravity for set transform requests.
        :param headlights: Enable the headlights.
        :param leftTurnSignal: Enable the left turn signal.
        :param rightTurnSignal: Enable the right turn signal.
        :param brakeSignal: Enable the brake lights (does not affect the motion of the vehicle).
        :param reverseSignal: Enable the reverse lights.
        :param waitForConfirmation: (Optional) Wait for confirmation before proceeding. This makes the method a blocking operation. NOTE: Return data will only be valid if waitForConfirmation is True.
        :type location: float array[3]
        :type rotation: float array[3]
        :type enableDynamics: boolean
        :type headlights: boolean
        :type leftTurnSignal: boolean
        :type rightTurnSignal: boolean
        :type brakeSignal: boolean
        :type reverseSignal: boolean
        :type waitForConfirmation: boolean
        :return:
            - **status** - True if successful or False otherwise
            - **location** - in full scale
            - **rotation** - in radians
            - **forward vector** - unit scale vector
            - **up vector** - unit scale vector
            - **front bumper hit** - True if in contact with a collision object, False otherwise
            - **rear bumper hit** - True if in contact with a collision object, False otherwise
        :rtype: boolean, float array[3], float array[3], float array[3], float array[3], boolean, boolean

        """
        success, location, rotation, forward_vector, up_vector, frontHit, rearHit = self.set_transform_and_request_state(location, [rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi], enableDynamics, headlights, leftTurnSignal, rightTurnSignal, brakeSignal, reverseSignal, waitForConfirmation)
        rotation_deg = [rotation[0]/math.pi*180, rotation[1]/math.pi*180, rotation[2]/math.pi*180]

        return success, location, rotation_deg, forward_vector, up_vector, frontHit, rearHit

    def set_velocity_and_request_state(self, forward, turn, headlights, leftTurnSignal, rightTurnSignal, brakeSignal, reverseSignal):
        """Sets the velocity, turn angle in radians, and other car properties.

        :param forward: Speed in m/s of a full-scale car. Multiply physical QCar speeds by 10 to get full scale speeds.
        :param turn: Turn angle in radians. Positive values turn right.
        :param headlights: Enable the headlights.
        :param leftTurnSignal: Enable the left turn signal.
        :param rightTurnSignal: Enable the right turn signal.
        :param brakeSignal: Enable the brake lights (does not affect the motion of the vehicle).
        :param reverseSignal: Enable the reverse lights.
        :type forward: float
        :type turn: float
        :type headlights: boolean
        :type leftTurnSignal: boolean
        :type rightTurnSignal: boolean
        :type brakeSignal: boolean
        :type reverseSignal: boolean
        :return:
            - **status** - True if successful, False otherwise
            - **location**
            - **rotation** - in radians
            - **front bumper hit** - True if in contact with a collision object, False otherwise
            - **rear bumper hit** - True if in contact with a collision object, False otherwise
        :rtype: boolean, float array[3], float array[3], boolean, boolean


        """

        if (not self._is_actor_number_valid()):
            return False, [0,0,0], [0,0,0], False, False

        c = CommModularContainer()
        c.classID = self.ID_QCAR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QCAR_SET_VELOCITY_AND_REQUEST_STATE
        c.payload = bytearray(struct.pack(">ffBBBBB", forward, turn, headlights, leftTurnSignal, rightTurnSignal, brakeSignal, reverseSignal))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        location = [0,0,0]
        rotation = [0,0,0]
        frontHit = False
        rearHit = False


        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QCAR, self.actorNumber, self.FCN_QCAR_VELOCITY_STATE_RESPONSE)

            if (c == None):
                return False, location, rotation, frontHit, rearHit

            if len(c.payload) == 26:
                location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], frontHit, rearHit, = struct.unpack(">ffffff??", c.payload[0:26])
                return True, location, rotation, frontHit, rearHit
            else:
                return False, location, rotation, frontHit, rearHit
        else:
            return False, location, rotation, frontHit, rearHit

    def set_velocity_and_request_state_degrees(self, forward, turn, headlights, leftTurnSignal, rightTurnSignal, brakeSignal, reverseSignal):
        """Sets the velocity, turn angle in degrees, and other car properties.

        :param forward: Speed in m/s of a full-scale car. Multiply physical QCar speeds by 10 to get full scale speeds.
        :param turn: Turn angle in degrees. Positive values turn right.
        :param headlights: Enable the headlights.
        :param leftTurnSignal: Enable the left turn signal.
        :param rightTurnSignal: Enable the right turn signal.
        :param brakeSignal: Enable the brake lights (does not affect the motion of the vehicle).
        :param reverseSignal: Enable the reverse lights.
        :type forward: float
        :type turn: float
        :type headlights: boolean
        :type leftTurnSignal: boolean
        :type rightTurnSignal: boolean
        :type brakeSignal: boolean
        :type reverseSignal: boolean
        :return:
            - **status** - `True` if successful, `False` otherwise
            - **location**
            - **rotation** - in radians
            - **front bumper hit** - `True` if in contact with a collision object, `False` otherwise
            - **rear bumper hit** - `True` if in contact with a collision object, `False` otherwise
        :rtype: boolean, float array[3], float array[3], boolean, boolean


        """
        success, location, rotation, frontHit, rearHit = self.set_velocity_and_request_state(forward, turn/180*math.pi, headlights, leftTurnSignal, rightTurnSignal, brakeSignal, reverseSignal)

        rotation_deg = [rotation[0]/math.pi*180, rotation[1]/math.pi*180, rotation[2]/math.pi*180]
        return success, location, rotation_deg, frontHit, rearHit

    def possess(self, camera=CAMERA_TRAILING):
        """
        Possess (take control of) a QCar in QLabs with the selected camera.

        :param camera: Pre-defined camera constant. See CAMERA constants for available options. Default is the trailing camera.
        :type camera: uint32
        :return:
            - **status** - `True` if possessing the camera was successful, `False` otherwise
        :rtype: boolean

        """

        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_QCAR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QCAR_POSSESS
        c.payload = bytearray(struct.pack(">B", camera))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QCAR, self.actorNumber, self.FCN_QCAR_POSSESS_ACK)
            if (c == None):
                return False
            else:
                return True
        else:
            return False

    def ghost_mode(self, enable=True, color=[0,1,0]):
        """
        Ghost mode changes the selected QCar actor into a transparent colored version. This can be useful as a reference actor or indicating a change in state.

        :param enable: Set the QCar to the defined transparent color, otherwise revert to the solid color scheme.
        :param color: Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :type camera: uint32
        :type enable: boolean
        :type color: float array[3]
        :return:
            - **status** - `True` if successful, `False` otherwise
        :rtype: boolean

        """

        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_QCAR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QCAR_GHOST_MODE
        c.payload = bytearray(struct.pack(">Bfff", enable, color[0], color[1], color[2]))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QCAR, self.actorNumber, self.FCN_QCAR_GHOST_MODE_ACK)
            if (c == None):
                return False
            else:
                return True
        else:
            return False

    def get_image(self, camera):
        """
        Request a JPG image from one of the QCar cameras.

        :param camera: Pre-defined camera constant. See CAMERA constants for available options. Trailing and Overhead cameras cannot be selected.
        :type camera: uint32
        :return:
            - **status** - `True` and image data if successful, `False` and empty otherwise
            - **imageData** - Image in a JPG format
        :rtype: boolean, byte array with jpg data

        """

        if (not self._is_actor_number_valid()):
            return False, None

        c = CommModularContainer()
        c.classID = self.ID_QCAR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QCAR_CAMERA_DATA_REQUEST
        c.payload = bytearray(struct.pack(">I", camera))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QCAR, self.actorNumber, self.FCN_QCAR_CAMERA_DATA_RESPONSE)

            if (c == None):
                return False, None


            jpg_buffer = cv2.imdecode(np.frombuffer(bytearray(c.payload[8:len(c.payload)]), dtype=np.uint8, count=-1, offset=0), 1)


            return True, jpg_buffer
        else:
            return False, None

    def get_lidar(self, samplePoints=400):
        """
        Request LIDAR data from a QCar.

        :param samplePoints: (Optional) Change the number of points per revolution of the LIDAR.
        :type samplePoints: uint32
        :return: `True`, angles in radians, and distances in m if successful, `False`, none, and none otherwise
        :rtype: boolean, float array, float array

        """

        if (not self._is_actor_number_valid()):
            return False, None, None

        LIDAR_SAMPLES = 4096
        LIDAR_RANGE = 80*self._sensor_scaling

        # The LIDAR is simulated by using 4 orthogonal virtual cameras that are 1 pixel high. The
        # lens distortion of these cameras must be removed to accurately calculate the XY position
        # of the depth samples.
        quarter_angle = np.linspace(0, 45, int(LIDAR_SAMPLES/8))
        lens_curve = -0.0077*quarter_angle*quarter_angle + 1.3506*quarter_angle
        lens_curve_rad = lens_curve/180*np.pi

        angles = np.concatenate((np.pi*4/2-1*np.flip(lens_curve_rad), \
                                 lens_curve_rad, \
                                 (np.pi/2 - 1*np.flip(lens_curve_rad)), \
                                 (np.pi/2 + lens_curve_rad), \
                                 (np.pi - 1*np.flip(lens_curve_rad)), \
                                 (np.pi + lens_curve_rad), \
                                 (np.pi*3/2 - 1*np.flip(lens_curve_rad)), \
                                 (np.pi*3/2 + lens_curve_rad)))



        c = CommModularContainer()
        c.classID = self.ID_QCAR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QCAR_LIDAR_DATA_REQUEST
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QCAR, self.actorNumber, self.FCN_QCAR_LIDAR_DATA_RESPONSE)

            if (c == None):
                return False, None, None

            if ((len(c.payload)-4)/2 != LIDAR_SAMPLES):
                #print("Received {} bytes, expected {}".format(len(c.payload), LIDAR_SAMPLES*2))
                return False, None, None

            distance = np.linspace(0,0,LIDAR_SAMPLES)

            for count in range(LIDAR_SAMPLES-1):
                # clamp any value at 65535 to 0
                raw_value = ((c.payload[4+count*2] * 256 + c.payload[5+count*2] )) %65535

                # scale to LIDAR range
                distance[count] = (raw_value/65535)*LIDAR_RANGE


            # Resample the data using a linear radial distribution to the desired number of points
            # and realign the first index to be 0 (forward)
            sampled_angles = np.linspace(0,2*np.pi, num=samplePoints, endpoint=False)
            sampled_distance = np.linspace(0,0, samplePoints)

            index_raw = 512
            for count in range(samplePoints):
                while (angles[index_raw] < sampled_angles[count]):
                    index_raw = (index_raw + 1) % 4096


                if index_raw != 0:
                    if (angles[index_raw]-angles[index_raw-1]) == 0:
                        sampled_distance[count] = distance[index_raw]
                    else:
                        sampled_distance[count] = (distance[index_raw]-distance[index_raw-1])*(sampled_angles[count]-angles[index_raw-1])/(angles[index_raw]-angles[index_raw-1]) + distance[index_raw-1]


                else:
                    sampled_distance[count] = distance[index_raw]


            return True, sampled_angles, sampled_distance
        else:
            return False, None, None






























from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import numpy as np
import math
import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsQCarFlooring(QLabsActor):
    """ This class is for spawning qcar floor maps."""

    ID_FLOORING = 10090
    """Class ID"""

    FLOORING_QCAR_MAP_LARGE = 0
    FLOORING_QCAR_MAP_SMALL = 1


    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_FLOORING
       return
    























    from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import math
import struct
import cv2
import numpy as np


######################### MODULAR CONTAINER CLASS #########################

class QLabsQCar2(QLabsActor):
    """This class is for spawning QCars."""


    ID_QCAR = 161
    """ Class ID """
    FCN_QCAR_SET_VELOCITY_AND_REQUEST_STATE = 10
    FCN_QCAR_VELOCITY_STATE_RESPONSE = 11
    FCN_QCAR_SET_TRANSFORM_AND_REQUEST_STATE = 12
    FCN_QCAR_TRANSFORM_STATE_RESPONSE = 13
    FCN_QCAR_POSSESS = 20
    FCN_QCAR_POSSESS_ACK = 21
    FCN_QCAR_GHOST_MODE = 22
    FCN_QCAR_GHOST_MODE_ACK = 23
    FCN_QCAR_SET_LED_STRIP_UNIFORM = 30
    FCN_QCAR_SET_LED_STRIP_UNIFORM_ACK = 31
    FCN_QCAR_SET_LED_STRIP_INDIVIDUAL = 32
    FCN_QCAR_SET_LED_STRIP_INDIVIDUAL_ACK = 33
    FCN_QCAR_CAMERA_DATA_REQUEST = 100
    FCN_QCAR_CAMERA_DATA_RESPONSE = 101
    FCN_QCAR_LIDAR_DATA_REQUEST = 110
    FCN_QCAR_LIDAR_DATA_RESPONSE = 111


    CAMERA_CSI_RIGHT = 0
    #Image capture resolution: 820x410
    CAMERA_CSI_BACK = 1
    #Image capture resolution: 820x410
    CAMERA_CSI_LEFT = 2
    #Image capture resolution: 820x410
    CAMERA_CSI_FRONT = 3
    #Image capture resolution: 820x410
    CAMERA_RGB = 4
    #Image capture resolution: 640x480
    CAMERA_DEPTH = 5
    #Image capture resolution: 640x480
    CAMERA_OVERHEAD = 6
    CAMERA_TRAILING = 7
    #Note: The mouse scroll wheel can be used to zoom in and out in this mode. """

    _sensor_scaling = 1


    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_QCAR
       return

    def spawn_id(self, actorNumber, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, waitForConfirmation=True):
        """Spawns a new QCar actor.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type actorNumber: uint32
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 2 actor number not available or already in use, 3 unknown error, -1 communications error
        :rtype: int32

        """

        self._sensor_scaling = scale[0]
        return super().spawn_id(actorNumber, location, rotation, scale, configuration, waitForConfirmation)
        
    def spawn_id_degrees(self, actorNumber, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, waitForConfirmation=True):
        """Spawns a new QCar actor.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type actorNumber: uint32
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 2 actor number not available or already in use, 3 unknown error, -1 communications error
        :rtype: int32

        """
        
        self._sensor_scaling = scale[0]
        return super().spawn_id_degrees(actorNumber, location, rotation, scale, configuration, waitForConfirmation)
        
    def spawn(self, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, waitForConfirmation=True):
        """Spawns a new QCar actor with the next available actor number within this class.

        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred. Note that if this is False, the returned actor number will be invalid.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 3 unknown error, -1 communications error.
            - **actorNumber** - An actor number to use for future references.
        :rtype: int32, int32

        """   

        self._sensor_scaling = scale[0]
        return super().spawn(location, rotation, scale, configuration, waitForConfirmation)
               
    def spawn_degrees(self, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, waitForConfirmation=True):
        """Spawns a new QCar actor with the next available actor number within this class.

        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in degrees
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred. Note that if this is False, the returned actor number will be invalid.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 3 unknown error, -1 communications error.
            - **actorNumber** - An actor number to use for future references.
        :rtype: int32, int32

        """
        self._sensor_scaling = scale[0]
        return super().spawn_degrees(location, rotation, scale, configuration, waitForConfirmation)
        
    def spawn_id_and_parent_with_relative_transform(self, actorNumber, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, parentClassID=0, parentActorNumber=0, parentComponent=0, waitForConfirmation=True):
        """Spawns a new QCar actor relative to an existing actor and creates a kinematic relationship.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in radians
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param parentClassID: See the ID variables in the respective library classes for the class identifier
        :param parentActorNumber: User defined unique identifier for the class actor in QLabs
        :param parentComponent: `0` for the origin of the parent actor, see the parent class for additional reference frame options
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type actorNumber: uint32
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type parentClassID: uint32
        :type parentActorNumber: uint32
        :type parentComponent: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 2 actor number not available or already in use, 3 cannot find the parent actor, 4 unknown error, -1 communications error
        :rtype: int32

        """

        self._sensor_scaling = scale[0]
        return super().spawn_id_and_parent_with_relative_transform(actorNumber, location, rotation, scale, configuration, parentClassID, parentActorNumber, parentComponent, waitForConfirmation)

    def spawn_id_and_parent_with_relative_transform_degrees(self, actorNumber, location=[0,0,0], rotation=[0,0,0], scale=[1,1,1], configuration=0, parentClassID=0, parentActorNumber=0, parentComponent=0, waitForConfirmation=True):
        """Spawns a new QCar actor relative to an existing actor and creates a kinematic relationship.

        :param actorNumber: User defined unique identifier for the class actor in QLabs
        :param location: (Optional) An array of floats for x, y and z coordinates
        :param rotation: (Optional) An array of floats for the roll, pitch, and yaw in degrees
        :param scale: (Optional) An array of floats for the scale in the x, y, and z directions. Scale values of 0.0 should not be used and only uniform scaling is recommended. Sensor scaling will be based on scale[0].
        :param configuration: (Optional) Spawn configuration. See class library for configuration options.
        :param parentClassID: See the ID variables in the respective library classes for the class identifier
        :param parentActorNumber: User defined unique identifier for the class actor in QLabs
        :param parentComponent: `0` for the origin of the parent actor, see the parent class for additional reference frame options
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type actorNumber: uint32
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type parentClassID: uint32
        :type parentActorNumber: uint32
        :type parentComponent: uint32
        :type waitForConfirmation: boolean
        :return:
            - **status** - 0 if successful, 1 class not available, 2 actor number not available or already in use, 3 cannot find the parent actor, 4 unknown error, -1 communications error
        :rtype: int32

        """

        self._sensor_scaling = scale[0]
        return super().spawn_id_and_parent_with_relative_transform_degrees(actorNumber, location, rotation, scale, configuration, parentClassID, parentActorNumber, parentComponent, waitForConfirmation)

    def set_transform_and_request_state(self, location, rotation, enableDynamics, headlights, leftTurnSignal, rightTurnSignal, brakeSignal, reverseSignal, waitForConfirmation=True):
        """Sets the location, rotation, and other car properties. Note that setting the location ignores collisions so ensure that the location is free of obstacles that may trap the actor if it is subsequently used in a dynamic mode. This transform can also be used to "playback" previously recorded position data without the need for a full dynamic model.

        :param location: An array of floats for x, y and z coordinates in full-scale units. Multiply physical QCar locations by 10 to get full scale locations.
        :param rotation: An array of floats for the roll, pitch, and yaw in radians
        :param enableDynamics: Enables or disables gravity for set transform requests.
        :param headlights: If the type is a boolean set all the headlights. If the type is an int, it will be treated as a bit mask for individual light control (b0: left outside, b1: left middle, b2: left inside, b3: right outside, b4: right middle, b5: right inside).
        :param leftTurnSignal: If the type is a boolean set all the left turn signals. If the type is an int, it will be treated as a bit mask for individual light control (b0: front, b1: rear).
        :param rightTurnSignal: If the type is a boolean set all the right turn signals. If the type is an int, it will be treated as a bit mask for individual light control (b0: front, b1: rear).
        :param brakeSignal: This does not affect the motion of the vehicle. If the type is a boolean set all the brake lights. If the type is an int, it will be treated as a bit mask for individual light control (b0: left outside, b1: left inside, b2: right outside, b3: right inside).
        :param reverseSignal: If the type is a boolean set all the reverse lights. If the type is an int, it will be treated as a bit mask for individual light control (b0: left, b1: right).
        :param waitForConfirmation: (Optional) Wait for confirmation before proceeding. This makes the method a blocking operation. NOTE: Return data will only be valid if waitForConfirmation is True.
        :type location: float array[3]
        :type rotation: float array[3]
        :type enableDynamics: boolean/int
        :type headlights: boolean/int
        :type leftTurnSignal: boolean/int
        :type rightTurnSignal: boolean/int
        :type brakeSignal: boolean/int
        :type reverseSignal: boolean/int
        :type waitForConfirmation: boolean
        :return:
            - **status** - True if successful or False otherwise
            - **location** - in full scale
            - **rotation** - in radians
            - **forward vector** - unit scale vector
            - **up vector** - unit scale vector
            - **front bumper hit** - True if in contact with a collision object, False otherwise
            - **rear bumper hit** - True if in contact with a collision object, False otherwise
        :rtype: boolean, float array[3], float array[3], float array[3], float array[3], boolean, boolean

        """
        if (not self._is_actor_number_valid()):
            return False, [0,0,0], [0,0,0], [0,0,0], [0,0,0], False, False

        if type(headlights) == bool:
            if (headlights):
                headlights = 0xFF

        if type(leftTurnSignal) == bool:
            if (leftTurnSignal):
                leftTurnSignal = 0xFF

        if type(rightTurnSignal) == bool:
            if (rightTurnSignal):
                rightTurnSignal = 0xFF

        if type(brakeSignal) == bool:
            if (brakeSignal):
                brakeSignal = 0xFF

        if type(reverseSignal) == bool:
            if (reverseSignal):
                reverseSignal = 0xFF

        c = CommModularContainer()
        c.classID = self.ID_QCAR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QCAR_SET_TRANSFORM_AND_REQUEST_STATE
        c.payload = bytearray(struct.pack(">ffffffBBBBBB", location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], enableDynamics, headlights, leftTurnSignal, rightTurnSignal, brakeSignal, reverseSignal))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)


        location = [0,0,0]
        rotation = [0,0,0]
        forward_vector = [0,0,0]
        up_vector = [0,0,0]
        frontHit = False
        rearHit = False

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_QCAR, self.actorNumber, self.FCN_QCAR_TRANSFORM_STATE_RESPONSE)

                if (c == None):
                    return False, location, rotation, forward_vector, up_vector, frontHit, rearHit

                if len(c.payload) == 50:

                    location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], forward_vector[0], forward_vector[1], forward_vector[2], up_vector[0], up_vector[1], up_vector[2], frontHit, rearHit, = struct.unpack(">ffffffffffff??", c.payload[0:50])
                    return True, location, rotation, forward_vector, up_vector, frontHit, rearHit
                else:
                    return False, location, rotation, forward_vector, up_vector, frontHit, rearHit
            else:
                return True, location, rotation, forward_vector, up_vector, frontHit, rearHit
        else:
            return False, location, rotation, forward_vector, up_vector, frontHit, rearHit

    def set_transform_and_request_state_degrees(self, location, rotation, enableDynamics, headlights, leftTurnSignal, rightTurnSignal, brakeSignal, reverseSignal, waitForConfirmation=True):
        """Sets the location, rotation, and other car properties. Note that setting the location ignores collisions so ensure that the location is free of obstacles that may trap the actor if it is subsequently used in a dynamic mode. This transform can also be used to "playback" previously recorded position data without the need for a full dynamic model.

        :param location: An array of floats for x, y and z coordinates in full-scale units. Multiply physical QCar locations by 10 to get full scale locations.
        :param rotation: An array of floats for the roll, pitch, and yaw in degrees
        :param enableDynamics: (default True) Enables or disables gravity for set transform requests.
        :param headlights: If the type is a boolean set all the headlights. If the type is an int, it will be treated as a bit mask for individual light control (b0: left outside, b1: left middle, b2: left inside, b3: right outside, b4: right middle, b5: right inside).
        :param leftTurnSignal: If the type is a boolean set all the left turn signals. If the type is an int, it will be treated as a bit mask for individual light control (b0: front, b1: rear).
        :param rightTurnSignal: If the type is a boolean set all the right turn signals. If the type is an int, it will be treated as a bit mask for individual light control (b0: front, b1: rear).
        :param brakeSignal: This does not affect the motion of the vehicle. If the type is a boolean set all the brake lights. If the type is an int, it will be treated as a bit mask for individual light control (b0: left outside, b1: left inside, b2: right outside, b3: right inside).
        :param reverseSignal: If the type is a boolean set all the reverse lights. If the type is an int, it will be treated as a bit mask for individual light control (b0: left, b1: right).
        :param waitForConfirmation: (Optional) Wait for confirmation before proceeding. This makes the method a blocking operation. NOTE: Return data will only be valid if waitForConfirmation is True.
        :type location: float array[3]
        :type rotation: float array[3]
        :type enableDynamics: boolean
        :type headlights: boolean/int
        :type leftTurnSignal: boolean/int
        :type rightTurnSignal: boolean/int
        :type brakeSignal: boolean/int
        :type reverseSignal: boolean/int
        :type waitForConfirmation: boolean
        :return:
            - **status** - True if successful or False otherwise
            - **location** - in full scale
            - **rotation** - in radians
            - **forward vector** - unit scale vector
            - **up vector** - unit scale vector
            - **front bumper hit** - True if in contact with a collision object, False otherwise
            - **rear bumper hit** - True if in contact with a collision object, False otherwise
        :rtype: boolean, float array[3], float array[3], float array[3], float array[3], boolean, boolean

        """
        success, location, rotation, forward_vector, up_vector, frontHit, rearHit = self.set_transform_and_request_state(location, [rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi], enableDynamics, headlights, leftTurnSignal, rightTurnSignal, brakeSignal, reverseSignal, waitForConfirmation)
        rotation_deg = [rotation[0]/math.pi*180, rotation[1]/math.pi*180, rotation[2]/math.pi*180]

        return success, location, rotation_deg, forward_vector, up_vector, frontHit, rearHit

    def set_velocity_and_request_state(self, forward, turn, headlights, leftTurnSignal, rightTurnSignal, brakeSignal, reverseSignal):
        """Sets the velocity, turn angle in radians, and other car properties.

        :param forward: Speed in m/s of a full-scale car. Multiply physical QCar speeds by 10 to get full scale speeds.
        :param turn: Turn angle in radians. Positive values turn right.
        :param headlights: If the type is a boolean set all the headlights. If the type is an int, it will be treated as a bit mask for individual light control (b0: left outside, b1: left middle, b2: left inside, b3: right outside, b4: right middle, b5: right inside).
        :param leftTurnSignal: If the type is a boolean set all the left turn signals. If the type is an int, it will be treated as a bit mask for individual light control (b0: front, b1: rear).
        :param rightTurnSignal: If the type is a boolean set all the right turn signals. If the type is an int, it will be treated as a bit mask for individual light control (b0: front, b1: rear).
        :param brakeSignal: This does not affect the motion of the vehicle. If the type is a boolean set all the brake lights. If the type is an int, it will be treated as a bit mask for individual light control (b0: left outside, b1: left inside, b2: right outside, b3: right inside).
        :param reverseSignal: If the type is a boolean set all the reverse lights. If the type is an int, it will be treated as a bit mask for individual light control (b0: left, b1: right).
        :type forward: float
        :type turn: float
        :type headlights: boolean/int
        :type leftTurnSignal: boolean/int
        :type rightTurnSignal: boolean/int
        :type brakeSignal: boolean/int
        :type reverseSignal: boolean/int
        :return:
            - **status** - True if successful, False otherwise
            - **location**
            - **rotation** - in radians
            - **front bumper hit** - True if in contact with a collision object, False otherwise
            - **rear bumper hit** - True if in contact with a collision object, False otherwise
        :rtype: boolean, float array[3], float array[3], boolean, boolean


        """

        if (not self._is_actor_number_valid()):
            return False, [0,0,0], [0,0,0], False, False

        if type(headlights) == bool:
            if (headlights):
                headlights = 0xFF

        if type(leftTurnSignal) == bool:
            if (leftTurnSignal):
                leftTurnSignal = 0xFF

        if type(rightTurnSignal) == bool:
            if (rightTurnSignal):
                rightTurnSignal = 0xFF

        if type(brakeSignal) == bool:
            if (brakeSignal):
                brakeSignal = 0xFF

        if type(reverseSignal) == bool:
            if (reverseSignal):
                reverseSignal = 0xFF

        c = CommModularContainer()
        c.classID = self.ID_QCAR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QCAR_SET_VELOCITY_AND_REQUEST_STATE
        c.payload = bytearray(struct.pack(">ffBBBBB", forward, turn, headlights, leftTurnSignal, rightTurnSignal, brakeSignal, reverseSignal))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        location = [0,0,0]
        rotation = [0,0,0]
        frontHit = False
        rearHit = False


        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QCAR, self.actorNumber, self.FCN_QCAR_VELOCITY_STATE_RESPONSE)

            if (c == None):
                return False, location, rotation, frontHit, rearHit

            if len(c.payload) == 26:
                location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], frontHit, rearHit, = struct.unpack(">ffffff??", c.payload[0:26])
                return True, location, rotation, frontHit, rearHit
            else:
                return False, location, rotation, frontHit, rearHit
        else:
            return False, location, rotation, frontHit, rearHit

    def set_velocity_and_request_state_degrees(self, forward, turn, headlights, leftTurnSignal, rightTurnSignal, brakeSignal, reverseSignal):
        """Sets the velocity, turn angle in degrees, and other car properties.

        :param forward: Speed in m/s of a full-scale car. Multiply physical QCar speeds by 10 to get full scale speeds.
        :param turn: Turn angle in degrees. Positive values turn right.
        :param headlights: Enable the headlights
        :param leftTurnSignal: Enable the left turn signal
        :param rightTurnSignal: Enable the right turn signal
        :param brakeSignal: Enable the brake lights (does not affect the motion of the vehicle)
        :param reverseSignal: Play a honking sound
        :type turn: float
        :type enableDynamics: boolean
        :type headlights: boolean
        :type leftTurnSignal: boolean
        :type rightTurnSignal: boolean
        :type brakeSignal: boolean
        :type reverseSignal: boolean
        :return:
            - **status** - `True` if successful, `False` otherwise
            - **location**
            - **rotation** - in radians
            - **front bumper hit** - `True` if in contact with a collision object, `False` otherwise
            - **rear bumper hit** - `True` if in contact with a collision object, `False` otherwise
        :rtype: boolean, float array[3], float array[3], boolean, boolean


        """
        success, location, rotation, frontHit, rearHit = self.set_velocity_and_request_state(forward, turn/180*math.pi, headlights, leftTurnSignal, rightTurnSignal, brakeSignal, reverseSignal)

        rotation_deg = [rotation[0]/math.pi*180, rotation[1]/math.pi*180, rotation[2]/math.pi*180]
        return success, location, rotation_deg, frontHit, rearHit

    def possess(self, camera=CAMERA_TRAILING):
        """
        Possess (take control of) a QCar in QLabs with the selected camera.

        :param camera: Pre-defined camera constant. See CAMERA constants for available options. Default is the trailing camera.
        :type camera: uint32
        :return:
            - **status** - `True` if possessing the camera was successful, `False` otherwise
        :rtype: boolean

        """

        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_QCAR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QCAR_POSSESS
        c.payload = bytearray(struct.pack(">B", camera))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QCAR, self.actorNumber, self.FCN_QCAR_POSSESS_ACK)
            if (c == None):
                return False
            else:
                return True
        else:
            return False

    def ghost_mode(self, enable=True, color=[0,1,0]):
        """
        Ghost mode changes the selected QCar actor into a transparent colored version. This can be useful as a reference actor or indicating a change in state.

        :param enable: Set the QCar to the defined transparent color, otherwise revert to the solid color scheme.
        :param color: Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :type camera: uint32
        :type enable: boolean
        :type color: float array[3]
        :return:
            - **status** - `True` if successful, `False` otherwise
        :rtype: boolean

        """

        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_QCAR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QCAR_GHOST_MODE
        c.payload = bytearray(struct.pack(">Bfff", enable, color[0], color[1], color[2]))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QCAR, self.actorNumber, self.FCN_QCAR_GHOST_MODE_ACK)
            if (c == None):
                return False
            else:
                return True
        else:
            return False

    def get_image(self, camera):
        """
        Request a JPG image from one of the QCar cameras.

        :param camera: Pre-defined camera constant. See CAMERA constants for available options. Trailing and Overhead cameras cannot be selected.
        :type camera: uint32
        :return:
            - **status** - `True` and image data if successful, `False` and empty otherwise
            - **imageData** - Image in a JPG format
        :rtype: boolean, byte array with jpg data

        """

        if (not self._is_actor_number_valid()):
            return False, None

        c = CommModularContainer()
        c.classID = self.ID_QCAR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QCAR_CAMERA_DATA_REQUEST
        c.payload = bytearray(struct.pack(">I", camera))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QCAR, self.actorNumber, self.FCN_QCAR_CAMERA_DATA_RESPONSE)

            if (c == None):
                return False, None


            jpg_buffer = cv2.imdecode(np.frombuffer(bytearray(c.payload[8:len(c.payload)]), dtype=np.uint8, count=-1, offset=0), 1)


            return True, jpg_buffer
        else:
            return False, None

    def get_lidar(self, samplePoints=400):
        """
        Request LIDAR data from a QCar.

        :param samplePoints: (Optional) Change the number of points per revolution of the LIDAR.
        :type samplePoints: uint32
        :return: `True`, angles in radians, and distances in m if successful, `False`, none, and none otherwise
        :rtype: boolean, float array, float array

        """

        if (not self._is_actor_number_valid()):
            return False, None, None

        LIDAR_SAMPLES = 4096
        LIDAR_RANGE = 80*self._sensor_scaling

        # The LIDAR is simulated by using 4 orthogonal virtual cameras that are 1 pixel high. The
        # lens distortion of these cameras must be removed to accurately calculate the XY position
        # of the depth samples.
        quarter_angle = np.linspace(0, 45, int(LIDAR_SAMPLES/8))
        lens_curve = -0.0077*quarter_angle*quarter_angle + 1.3506*quarter_angle
        lens_curve_rad = lens_curve/180*np.pi

        angles = np.concatenate((np.pi*4/2-1*np.flip(lens_curve_rad), \
                                 lens_curve_rad, \
                                 (np.pi/2 - 1*np.flip(lens_curve_rad)), \
                                 (np.pi/2 + lens_curve_rad), \
                                 (np.pi - 1*np.flip(lens_curve_rad)), \
                                 (np.pi + lens_curve_rad), \
                                 (np.pi*3/2 - 1*np.flip(lens_curve_rad)), \
                                 (np.pi*3/2 + lens_curve_rad)))



        c = CommModularContainer()
        c.classID = self.ID_QCAR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QCAR_LIDAR_DATA_REQUEST
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QCAR, self.actorNumber, self.FCN_QCAR_LIDAR_DATA_RESPONSE)

            if (c == None):
                return False, None, None

            if ((len(c.payload)-4)/2 != LIDAR_SAMPLES):
                #print("Received {} bytes, expected {}".format(len(c.payload), LIDAR_SAMPLES*2))
                return False, None, None

            distance = np.linspace(0,0,LIDAR_SAMPLES)

            for count in range(LIDAR_SAMPLES-1):
                # clamp any value at 65535 to 0
                raw_value = ((c.payload[4+count*2] * 256 + c.payload[5+count*2] )) %65535

                # scale to LIDAR range
                distance[count] = (raw_value/65535)*LIDAR_RANGE


            # Resample the data using a linear radial distribution to the desired number of points
            # and realign the first index to be 0 (forward)
            sampled_angles = np.linspace(0,2*np.pi, num=samplePoints, endpoint=False)
            sampled_distance = np.linspace(0,0, samplePoints)

            index_raw = 512
            for count in range(samplePoints):
                while (angles[index_raw] < sampled_angles[count]):
                    index_raw = (index_raw + 1) % 4096


                if index_raw != 0:
                    if (angles[index_raw]-angles[index_raw-1]) == 0:
                        sampled_distance[count] = distance[index_raw]
                    else:
                        sampled_distance[count] = (distance[index_raw]-distance[index_raw-1])*(sampled_angles[count]-angles[index_raw-1])/(angles[index_raw]-angles[index_raw-1]) + distance[index_raw-1]


                else:
                    sampled_distance[count] = distance[index_raw]


            return True, sampled_angles, sampled_distance
        else:
            return False, None, None


    def set_led_strip_uniform(self, color=[0, 0, 0], waitForConfirmation=True):
        """Sets the entire LED color strip to the RGB value specified.

        :param color: Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale. Values greater than 1 can be used to enhance the glow or bloom effect (try 50).
        :type color: float array[3]
        :return:
            - **status** - `True` if successful, `False` otherwise
        :rtype: boolean                   


        """

        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_QCAR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QCAR_SET_LED_STRIP_UNIFORM
        c.payload = bytearray(struct.pack(">fff", color[0], color[1], color[2]))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_QCAR, self.actorNumber, self.FCN_QCAR_SET_LED_STRIP_INDIVIDUAL_ACK)

                if (c == None):
                    return False


            return True
        else:
            return False
        

    def set_led_strip_individual(self, color, waitForConfirmation=True):
        """Sets the entire LED color strip to the RGB value specified. Note that specifying individual LED's has a slight impact on performance versus set_led_strip_uniform.

        :param color: A 2D array Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale by 33 rows. Values greater than 1 can be used to enhance the glow or bloom effect (try 50).
        :type color: float array[3][33]
        :return:
            - **status** - `True` if successful, `False` otherwise
        :rtype: boolean                   


        """

        if (not self._is_actor_number_valid()):
            return False
        
        if (len(color) != 33):
            if self._verbose == True:
                print("The color array was of length {} instead of the expected length of 33.".format(len(color)))

            return False
        
        if (len(color[0]) != 3):
            if self._verbose == True:
                print("Each row of the color array should be 3 elements (received {}).".format(len(color[0])))

            return False

        c = CommModularContainer()
        c.classID = self.ID_QCAR
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QCAR_SET_LED_STRIP_INDIVIDUAL
        c.payload = bytearray()

        for LED in color:
            c.payload = c.payload + bytearray(struct.pack(">fff", LED[0], LED[1], LED[2]))        

        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_QCAR, self.actorNumber, self.FCN_QCAR_SET_LED_STRIP_INDIVIDUAL_ACK)

                if (c == None):
                    return False

            return True
        else:
            return False        
        



















from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import cv2
import numpy as np
import math
import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsQDrone2(QLabsActor):
    """This class is for spawning a QDrone 2."""

    ID_QDRONE2 = 231

    FCN_QDRONE2_COMMAND_VELOCITY_AND_REQUEST_STATE = 10
    FCN_QDRONE2_COMMAND_VELOCITY_AND_REQUEST_STATE_RESPONSE = 11
    FCN_QDRONE2_SET_WORLD_TRANSFORM = 12
    FCN_QDRONE2_SET_WORLD_TRANSFORM_ACK = 13
    FCN_QDRONE2_POSSESS = 20
    FCN_QDRONE2_POSSESS_ACK = 21
    FCN_QDRONE2_IMAGE_REQUEST = 100
    FCN_QDRONE2_IMAGE_RESPONSE = 101
    FCN_QDRONE2_SET_CAMERA_RESOLUTION = 102
    FCN_QDRONE2_SET_CAMERA_RESOLUTION_RESPONSE = 103
    

    VIEWPOINT_CSI_LEFT = 0
    VIEWPOINT_CSI_BACK = 1
    VIEWPOINT_CSI_RIGHT = 2
    VIEWPOINT_RGB = 3
    VIEWPOINT_DEPTH = 4
    VIEWPOINT_DOWNWARD = 5
    VIEWPOINT_OPTICAL_FLOW = 6
    VIEWPOINT_OVERHEAD = 7
    VIEWPOINT_TRAILING = 8
 
    CAMERA_CSI_LEFT = 0
    CAMERA_CSI_BACK = 1
    CAMERA_CSI_RIGHT = 2
    CAMERA_RGB = 3
    CAMERA_DEPTH = 4
    CAMERA_DOWNWARD = 5
    CAMERA_OPTICAL_FLOW = 6
    

    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_QDRONE2
       return

    def possess(self, camera=VIEWPOINT_TRAILING):
        """
        Possess (take control of) a QDrone in QLabs with the selected camera.

        :param camera: Pre-defined camera constant. See CAMERA constants for available options. Default is the trailing camera.
        :type camera: uint32
        :return:
            - **status** - `True` if possessing the camera was successful, `False` otherwise
        :rtype: boolean

        """
        
        if (not self._is_actor_number_valid()):
            return False
        
        c = CommModularContainer()
        c.classID = self.ID_QDRONE2
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QDRONE2_POSSESS
        c.payload = bytearray(struct.pack(">I", camera))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QDRONE2, self.actorNumber, self.FCN_QDRONE2_POSSESS_ACK)
            if (c == None):
                if self._verbose:
                    print("QDrone 2 possess: No data returned from QLabs possibly due to a communcations timeout.")
                return False
            else:
                return True
        else:
            if self._verbose:
                print("QDrone 2 possess: Communications failure.")
            return False

    def command_velocity_and_request_state(self, motorsEnabled=False, velocity=[0,0,0], orientation=[0,0,0]):
        """Sets the velocity, turn angle in radians, and other properties.

        :param motorsEnabled: Enable the motors. Disabled by default immediately after spawning.
        :param velocity: The linear velocity in m/s in the body frame.
        :param orientation: The orientation in radians expressed in roll-pitch-yaw Euler angles.
        
        :type motorsEnabled: boolean
        :type velocity: float array[3]
        :type orientation: float array[3]

        :return:
            - **status** - `True` if successful, `False` otherwise. Other returned values are invalid if status is `False`.
            - **location** - World location in m
            - **orientation** - World orientation in radians (roll, pitch, yaw)
            - **quaternion** - World orientation in a quaternion vector
            - **velocity** - World linear velocity in m/s
            - **TOF distance** - Time of flight distance sensor. Returns 0 when outside the range of the sensor (too close or too far).
            - **collision** - The QDrone is currently colliding with another object or the environment.
            - **collision location** - Body frame location. Invalid if collision is False.
            - **collision force vector** - The vector along which the collision force is occuring. Invalid if collision is False.
            
        :rtype: boolean, float array[3], float array[3], float array[4], float array[3], float, boolean, float array[3], float array[3]


        """
        if (not self._is_actor_number_valid()):
            return False, [0,0,0], [0,0,0], [0,0,0,0], [0,0,0], 0.0, False, [0,0,0], [0,0,0]
        
        c = CommModularContainer()
        c.classID = self.ID_QDRONE2
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QDRONE2_COMMAND_VELOCITY_AND_REQUEST_STATE
        c.payload = bytearray(struct.pack(">ffffffB", velocity[0], velocity[1], velocity[2], orientation[0], orientation[1], orientation[2], motorsEnabled))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)
        
        location = [0,0,0]
        orientation = [0,0,0]
        quaternion = [0,0,0,0]
        velocity = [0,0,0]
        TOFDistance = 0
        collision = False
        collisionLocation = [0,0,0]
        collisionForce = [0,0,0]  


        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QDRONE2, self.actorNumber, self.FCN_QDRONE2_COMMAND_VELOCITY_AND_REQUEST_STATE_RESPONSE)

            if (c == None):
                if self._verbose:
                    print("QDrone 2 command_velocity_and_request_state: Received no data from QLabs possibly due to a communcations timeout.")
                return False, location, orientation, quaternion, velocity, TOFDistance, collision, collisionLocation, collisionForce

            if len(c.payload) == 81:
                location[0], location[1], location[2], orientation[0], orientation[1], orientation[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3], velocity[0], velocity[1], velocity[2], TOFDistance, collision, collisionLocation[0], collisionLocation[1], collisionLocation[2], collisionForce[0], collisionForce[1], collisionForce[2], = struct.unpack(">ffffffffffffff?ffffff", c.payload[0:81])
                return True, location, orientation, quaternion, velocity, TOFDistance, collision, collisionLocation, collisionForce
            else:
                if self._verbose:
                    print("QDrone 2 command_velocity_and_request_state: Response packet was not the expected length ({} bytes).".format(len(c.payload)))
                return False, location, orientation, quaternion, velocity, TOFDistance, collision, collisionLocation, collisionForce

        else:
            if self._verbose:
                print("QDrone 2 command_velocity_and_request_state: Communications failure.")
            return False, location, orientation, quaternion, velocity, TOFDistance, collision, collisionLocation, collisionForce

    def command_velocity_and_request_state_degrees(self, motorsEnabled=False, velocity=[0,0,0], orientation=[0,0,0]):
        """Sets the velocity, turn angle in radians, and other properties.

        :param motorsEnabled: Enable the motors. Disabled by default immediately after spawning.
        :param velocity: The linear velocity in m/s in the body frame.
        :param orientation: The orientation in degrees expressed in roll-pitch-yaw Euler angles.
        
        :type motorsEnabled: boolean
        :type velocity: float array[3]
        :type orientation: float array[3]

        :return:
            - **status** - `True` if successful, `False` otherwise. Other returned values are invalid if status is `False`.
            - **location** - World location in m
            - **orientation** - World orientation in degrees (roll, pitch, yaw)
            - **quaternion** - World orientation in a quaternion vector
            - **velocity** - World linear velocity in m/s
            - **TOF distance** - Time of flight distance sensor. Returns 0 when outside the range of the sensor (too close or too far).
            - **collision** - The QDrone is currently colliding with another object or the environment.
            - **collision location** - Body frame location. Invalid if collision is False.
            - **collision force vector** - The vector along which the collision force is occuring. Invalid if collision is False.
            
        :rtype: boolean, float array[3], float array[3], float array[4], float array[3], float, boolean, float array[3], float array[3]


        """   

        success, location, orientation_r, quaternion, velocity, TOFDistance, collision, collisionLocation, collisionForce = command_velocity_and_request_state(self, motorsEnabled, velocity, [orientation[0]/180*math.pi, orientation[1]/180*math.pi, orientation[2]/180*math.pi])
        
        orientation_d = [orientation_r[0]/math.pi*180, orientation_r[1]/math.pi*180, orientation_r[2]/math.pi*180]
        
        return success, location, orientation_d, quaternion, velocity, TOFDistance, collision, collisionLocation, collisionForce

    def set_transform_and_dynamics(self, location, rotation, enableDynamics, waitForConfirmation=True):
        """Sets the location, rotation, and other properties. Note that setting the location ignores collisions so ensure that the location is free of obstacles that may trap the actor if it is subsequently used in a dynamic mode. This transform can also be used to "playback" previously recorded position data without the need for a full dynamic model.

        :param location: An array of floats for x, y and z coordinates in full-scale units. Multiply physical QCar locations by 10 to get full scale locations.
        :param rotation: An array of floats for the roll, pitch, and yaw in radians
        :param enableDynamics: Enables or disables dynamics. The velocity commands will have no effect when the dynamics are disabled.
        :param waitForConfirmation: (Optional) Wait for confirmation before proceeding. This makes the method a blocking operation.
        :type location: float array[3]
        :type rotation: float array[3]
        :type enableDynamics: boolean
        :type waitForConfirmation: boolean
        :return:
            - **status** - True if successful or False otherwise
        :rtype: boolean

        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_QDRONE2
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QDRONE2_SET_WORLD_TRANSFORM
        c.payload = bytearray(struct.pack(">ffffffB", location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], enableDynamics))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)


        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_QDRONE2, self.actorNumber, self.FCN_QDRONE2_SET_WORLD_TRANSFORM_ACK)

                if (c == None):
                    return False

                
            return True
        else:
            return False

    def get_image(self, camera):
        """
        Request a JPG image from the QDrone camera.

        :param camera: Camera number to view from. Use the CAMERA constants.
        
        :type camera: int32

        :return:
            - **status** - `True` and image data if successful, `False` and empty otherwise
            - **cameraNumber** - Camera number of the image. A value of -1 indicates an invalid camera was selected.
            - **imageData** - Image in a JPG format
        :rtype: boolean, int32, byte array with jpg data

        """

        if (not self._is_actor_number_valid()):
            return False, -1, None

        c = CommModularContainer()
        c.classID = self.ID_QDRONE2
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_QDRONE2_IMAGE_REQUEST
        c.payload = bytearray(struct.pack(">I", camera))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_QDRONE2, self.actorNumber, self.FCN_QDRONE2_IMAGE_RESPONSE)

            if (c == None):
                if self._verbose:
                    print("QDrone 2 get_image: No data returned from QLabs possibly due to a communcations timeout.")
                return False, -1, None

            if len(c.payload) >= 8:
                camera_number, image_size, = struct.unpack(">II", c.payload[0:8])

                if (camera_number >= 0) and (len(c.payload) > 8):
                    imageData = cv2.imdecode(np.frombuffer(bytearray(c.payload[8:len(c.payload)]), dtype=np.uint8, count=-1, offset=0), 1)
                    
                    if (imageData is None) and self._verbose:
                        print("QDrone 2 get_image: Error decoding image data.")   
                    return True, camera_number, imageData
                else:
                    if self._verbose:
                        if (camera_number >= 0):
                            print("QDrone 2 get_image: Camera number was invalid.")
                        else:
                            print("Camera number was valid, but no image data was returned. Check that the image size is valid.")
                    return False, camera_number, None                    
            else:
                if self._verbose:
                    print("QDrone 2 get_image: Returned data was not in the expected format.")
                return False, -1, None
        else:
            if self._verbose:
                print("QDrone 2 get_image: Communications failure.")
            return False, -1, None

    def set_image_capture_resolution(self, width=640, height=480):
        """Change the default width and height of image resolution for capture

        :param width: Must be an even number. Default 640
        :param height: Must be an even number. Default 480
        :type width: uint32
        :type height: uint32
        :return: `True` if spawn was successful, `False` otherwise
        :rtype: boolean
        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_FREE_CAMERA
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_FREE_CAMERA_SET_IMAGE_RESOLUTION
        c.payload = bytearray(struct.pack(">II", width, height))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_FREE_CAMERA, self.actorNumber, self.FCN_FREE_CAMERA_SET_IMAGE_RESOLUTION_RESPONSE)
            if (c == None):
                return False
            else:
                return True
        else:
            return False
        






















        from quanser.communications import Stream, StreamError, PollFlag
from quanser.common import ErrorCode
try:
    from quanser.common import Timeout
except:
    from quanser.communications import Timeout

import struct
import os
import platform
import time


######################### MODULAR CONTAINER CLASS #########################

class CommModularContainer:

    """The CommModularContainer is a collection of data used to communicate with actors. Multiple containers can be packaged into a single packet."""

    # Define class-level variables
    containerSize = 0
    """The size of the packet in bytes. Container size (uint32: 4 bytes) + class ID (uint32: 4 bytes) + actor number (uint32: 4 bytes) + actor function (1 byte) + payload (varies per function)"""
    classID = 0
    """See the class ID variables in the respective library classes."""
    actorNumber = 0
    """An identifier that should be unique for each actor of a given class. """
    actorFunction = 0
    """See the FCN constants in the respective library classes."""
    payload = bytearray()
    """A variable sized payload depending on the actor function in use."""

    ID_GENERIC_ACTOR_SPAWNER = 135
    """The actor spawner is a special actor class that exists in open world environments that manages the spawning and destruction of dynamic actors."""
    FCN_GENERIC_ACTOR_SPAWNER_SPAWN_ID = 10
    FCN_GENERIC_ACTOR_SPAWNER_SPAWN_ID_ACK = 11
    FCN_GENERIC_ACTOR_SPAWNER_DESTROY_ACTOR = 12
    FCN_GENERIC_ACTOR_SPAWNER_DESTROY_ACTOR_ACK = 13
    FCN_GENERIC_ACTOR_SPAWNER_DESTROY_ALL_ACTORS_OF_CLASS = 24
    FCN_GENERIC_ACTOR_SPAWNER_DESTROY_ALL_ACTORS_OF_CLASS_ACK = 25
    FCN_GENERIC_ACTOR_SPAWNER_DESTROY_ALL_SPAWNED_ACTORS = 14
    FCN_GENERIC_ACTOR_SPAWNER_DESTROY_ALL_SPAWNED_ACTORS_ACK = 15
    FCN_GENERIC_ACTOR_SPAWNER_REGENERATE_CACHE_LIST = 16
    FCN_GENERIC_ACTOR_SPAWNER_REGENERATE_CACHE_LIST_ACK = 17
    FCN_GENERIC_ACTOR_SPAWNER_SPAWN = 22
    FCN_GENERIC_ACTOR_SPAWNER_SPAWN_RESPONSE = 23
    FCN_GENERIC_ACTOR_SPAWNER_SPAWN_AND_PARENT_RELATIVE = 50
    FCN_GENERIC_ACTOR_SPAWNER_SPAWN_AND_PARENT_RELATIVE_ACK = 51
    FCN_GENERIC_ACTOR_SPAWNER_PARENT_CURRENT_WORLD = 52
    FCN_GENERIC_ACTOR_SPAWNER_PARENT_CURRENT_WORLD_ACK = 53
    FCN_GENERIC_ACTOR_SPAWNER_PARENT_RELATIVE = 54
    FCN_GENERIC_ACTOR_SPAWNER_PARENT_RELATIVE_ACK = 55
    FCN_GENERIC_ACTOR_SPAWNER_PARENT_BREAK_WITH_CURRENT_WORLD = 56
    FCN_GENERIC_ACTOR_SPAWNER_PARENT_BREAK_WITH_CURRENT_WORLD_ACK = 57

    ID_UNKNOWN = 0
    """Class ID 0 is reserved as an unknown class. QLabs may respond with a container with information it does not understand due to an unknown class, if data was improperly formatted, or if communication methods were executed in the wrong order."""

    BASE_CONTAINER_SIZE = 13
    """Container size variable (4 bytes) + class ID (4 bytes) + actor number (4 bytes) + actor function (1 byte). Does not include the payload size which is variable per function."""

    # Initialize class
    def __init__(self):

       return

######################### COMMUNICATIONS #########################

class QuanserInteractiveLabs:
    """This class establishes a server connection with QLabs and manages the communications."""
    _stream = None
    
    _BUFFER_SIZE = 100000

    _readBuffer = bytearray(_BUFFER_SIZE)
    _sendBuffer = bytearray()

    _receivePacketBuffer = bytearray()
    _receivePacketSize = 0
    _receivePacketContainerIndex = 0
    _wait_for_container_timeout = 5

    _send_queue = bytearray()


    # Initialize QLabs
    def __init__(self):
        """ Constructor Method """
        pass

    def open(self, address, timeout=10):
        """Open a connection to QLabs.

        :param address: The machine name or IP address of a local or remote copy of QLabs such as "localhost", or "192.168.1.123".
        :param timeout: (Optional) Period to attempt the connection for before aborting. Default 10s.
        :type address: string
        :type timeout: float
        :return: `True` if successful, `False` otherwise
        :rtype: boolean

        """

        address = "tcpip://" + address + ":18000"

        self._stream = Stream()

        result = self._stream.connect(address, True, self._BUFFER_SIZE, self._BUFFER_SIZE)
        if (result == False):
            # Connection was not immediate so poll the stream to determine what might be wrong
            pollResult = 0

            while (((pollResult & PollFlag.CONNECT) != PollFlag.CONNECT) and (timeout > 0)):
                try:
                    pollResult = self._stream.poll(Timeout(1), PollFlag.CONNECT)
                except StreamError as e:
                    if e.error_code == -33:
                        self._stream.close()
                        return False
                    
                timeout = timeout - 1

            if pollResult & PollFlag.CONNECT == PollFlag.CONNECT:
                #print("Connection accepted")
                pass
            else:
                if (timeout == 0):
                    print("Connection timeout")
                    
                self._stream.close()
                return False

        return True

    def close(self):
        """Shutdown and close a connection to QLabs. Always close a connection when communications are finished.

        :return: No return. If an existing connection cannot be found, the function will fail silently.
        :rtype: none

        """
        try:
            self._stream.shutdown()
            self._stream.close()
        except:
            pass

    def queue_add_container(self, container):
        """Queue a single container into a buffer for future transmission

        :param container: CommModularContainer populated with the actor information.
        :type container: CommModularContainer object

        """

        self._send_queue = self._send_queue + bytearray(struct.pack(">iiiB", container.containerSize, container.classID, container.actorNumber, container.actorFunction)) + container.payload

    def queue_send(self):
        """Package the containers in the queue and transmit immediately

        :param container: CommModularContainer populated with the actor information.
        :type container: CommModularContainer object
        :return: `True` if successful and the queue will be emptied, `False` otherwise and the queue will remain intact.
        :rtype: boolean

        """

        try:
            data = bytearray(struct.pack("<iB", 1+len(self._send_queue))) + self._send_queue
            numBytes = len(data)
            bytesWritten = self._stream.send(data, numBytes)
            self._stream.flush()
            self._send_queue = bytearray()
            return True
        except:
            return False

    def queue_destroy(self):
        """The container queue is emptied of all data.

        """
        self._send_queue = bytearray()

    # Pack data and send immediately
    def send_container (self, container):
        """Package a single container into a packet and transmit immediately

        :param container: CommModularContainer populated with the actor information.
        :type container: CommModularContainer object
        :return: `True` if successful, `False` otherwise
        :rtype: boolean

        """
        result = False
        try:
            data = bytearray(struct.pack("<i", 1+container.containerSize)) + bytearray(struct.pack(">BiiiB", 123, container.containerSize, container.classID, container.actorNumber, container.actorFunction)) + container.payload
            numBytes = len(data)
            bytesWritten = self._stream.send_byte_array(data, numBytes)
            if bytesWritten > 0:
                self._stream.flush()
                result = True
        except:
            pass

        return result

    # Check if new data is available.  Returns true if a complete packet has been received.
    def receive_new_data(self):
        """Poll for new data received from QLabs through the communications framework. If you are expecting large amounts of data such as video, this should be executed frequently to avoid overflowing internal buffers. Data split over multiple packets will be automatically reassembled before returning true. This method is non-blocking.

        :return: `True` if at least one complete container has been received, `False` otherwise
        :rtype: boolean

        """
        bytesRead = self._stream.receive(self._readBuffer, self._BUFFER_SIZE) # returns -ErrorCode.WOULD_BLOCK if it would block
        newData = False

        while bytesRead > 0:
            self._receivePacketBuffer += bytearray(self._readBuffer[0:(bytesRead)])

            #while we're here, check if there are any more bytes in the receive buffer
            bytesRead = self._stream.receive(self._readBuffer, self._BUFFER_SIZE) # returns -ErrorCode.WOULD_BLOCK if it would block

        # check if we already have data in the receive buffer that was unprocessed (multiple packets in a single receive)
        if len(self._receivePacketBuffer) > 5:
            if (self._receivePacketBuffer[4] == 123):

                # packet size
                self._receivePacketSize, = struct.unpack("<I", self._receivePacketBuffer[0:4])
                # add the 4 bytes for the size to the packet size
                self._receivePacketSize = self._receivePacketSize + 4

                if len(self._receivePacketBuffer) >= self._receivePacketSize:

                    self._receivePacketContainerIndex = 5
                    newData = True

            else:
                print("Error parsing multiple packets in receive buffer.  Clearing internal buffers.")
                _receivePacketBuffer = bytearray()

        return newData

    # Parse out received containers
    def get_next_container(self):
        """If receive_new_data has returned true, use this method to receive the next container in the queue.

        :return: The data will be returned in a CommModularContainer object along with a flag to indicate if additional complete containers remain in the queue for extraction. If this method was used without checking for new data first and the queue is empty, the container will contain the default values with a class ID of ID_UNKNOWN.
        :rtype: CommModularContainer object, boolean

        """

        c = CommModularContainer()
        isMoreContainers = False

        if (self._receivePacketContainerIndex > 0):
            c.containerSize, = struct.unpack(">I", self._receivePacketBuffer[self._receivePacketContainerIndex:(self._receivePacketContainerIndex+4)])
            c.classID, = struct.unpack(">I", self._receivePacketBuffer[(self._receivePacketContainerIndex+4):(self._receivePacketContainerIndex+8)])
            c.actorNumber, = struct.unpack(">I", self._receivePacketBuffer[(self._receivePacketContainerIndex+8):(self._receivePacketContainerIndex+12)])
            c.actorFunction = self._receivePacketBuffer[self._receivePacketContainerIndex+12]
            c.payload = bytearray(self._receivePacketBuffer[(self._receivePacketContainerIndex+c.BASE_CONTAINER_SIZE):(self._receivePacketContainerIndex+c.containerSize)])

            self._receivePacketContainerIndex = self._receivePacketContainerIndex + c.containerSize

            if (self._receivePacketContainerIndex >= self._receivePacketSize):

                isMoreContainers = False

                if len(self._receivePacketBuffer) == self._receivePacketSize:
                    # The data buffer contains only the one packet.  Clear the buffer.
                    self._receivePacketBuffer = bytearray()
                else:
                    # Remove the packet from the data buffer.  There is another packet in the buffer already.
                    self._receivePacketBuffer = self._receivePacketBuffer[(self._receivePacketContainerIndex):(len(self._receivePacketBuffer))]

                self._receivePacketContainerIndex = 0

            else:
                isMoreContainers = True


        return c, isMoreContainers

    def set_wait_for_container_timeout(self, timeout):
        """By default, a method using the wait_for_container method (typically represented with the waitForComfirmation flag) will abort waiting for an acknowledgment after 5 seconds
        at which time the method will return a failed response. This time period can be adjusted with this function. Values
        less than or equal to zero will cause the methods to wait indefinitely until the expected acknowledgment is received.

        :param timeout: Timeout period in seconds
        :type timeout: float

        """


        if (timeout < 0):
            timeout = 0

        self._wait_for_container_timeout = timeout

    def wait_for_container(self, classID, actorNumber, functionNumber):
        """Continually poll and parse incoming containers until a response from specific actor with a specific function response is received.
        Containers that do not match the class, actor number, and function number are discarded. This function blocks until the appropriate packet
        is received or the timeout is reached.

        :return: The data will be returned in a CommModularContainer object.
        :rtype: CommModularContainer object

        """

        startTime = time.time()

        while(True):
            while (self.receive_new_data() == False):
                if self._wait_for_container_timeout > 0:
                    currentTime = time.time()
                    if (currentTime - startTime >= self._wait_for_container_timeout):
                        return None
                pass

            moreContainers = True

            while (moreContainers):
                c, moreContainers = self.get_next_container()

                if c.classID == classID:
                    if c.actorNumber == actorNumber:
                        if c.actorFunction == functionNumber:
                            return c

    def flush_receive(self):
        """Flush receive buffers removing all unread data. This can be used to clear receive buffers after fault conditions to ensure it contains only new data.

        :return: None
        :rtype: None

        """
        bytesRead = self._stream.receive(self._readBuffer, self._BUFFER_SIZE) # returns -ErrorCode.WOULD_BLOCK if it would block
        self._receivePacketBuffer = bytearray()
        self._receivePacketContainerIndex = 0

    def regenerate_cache_list(self):
        """Advanced function for actor indexing.

        :return: `True` if successful, `False` otherwise
        :rtype: boolean

        .. danger::

            TODO: Improve this description.
        """
        c = CommModularContainer()
        c.classID = CommModularContainer.ID_GENERIC_ACTOR_SPAWNER
        c.actorNumber = 0
        c.actorFunction = CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_REGENERATE_CACHE_LIST
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if (self.send_container(c)):
            c = self.wait_for_container(CommModularContainer.ID_GENERIC_ACTOR_SPAWNER, c.actorNumber, CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_REGENERATE_CACHE_LIST_ACK)
            if (c == None):
                return False
            else:
                return True

        else:
            return False

    def ping(self):
        """QLabs will automatically disconnect a non-responsive client connection. The ping method can be used to keep the connection alive if operations are infrequent.

        :return: `True` if successful, `False` otherwise
        :rtype: boolean
        """

        FCN_REQUEST_PING = 1
        FCN_RESPONSE_PING = 2


        c = CommModularContainer()
        c.classID = CommModularContainer.ID_GENERIC_ACTOR_SPAWNER
        c.actorNumber = 0
        c.actorFunction = FCN_REQUEST_PING
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self.flush_receive()

        if (self.send_container(c)):

            c = self.wait_for_container(CommModularContainer.ID_GENERIC_ACTOR_SPAWNER, 0, FCN_RESPONSE_PING)
            if (c == None):
                return False
            elif c.payload[0] > 0:
                return True
            else:
                return False
        else:
            return False

    def destroy_all_spawned_actors(self):
        """Find and destroy all spawned actors and widgets. This is a blocking operation.

        :return: The number of actors deleted. -1 if failed.
        :rtype: int32

        """
        actorNumber = 0
        c = CommModularContainer()

        c.classID = CommModularContainer.ID_GENERIC_ACTOR_SPAWNER
        c.actorNumber = actorNumber
        c.actorFunction = CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_DESTROY_ALL_SPAWNED_ACTORS
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if (self.send_container(c)):
            c = self.wait_for_container(CommModularContainer.ID_GENERIC_ACTOR_SPAWNER, actorNumber, CommModularContainer.FCN_GENERIC_ACTOR_SPAWNER_DESTROY_ALL_SPAWNED_ACTORS_ACK)
            if (c == None):
                return -1

            if len(c.payload) == 4:
                num_actors_destroyed, = struct.unpack(">I", c.payload[0:4])
                return num_actors_destroyed
            else:
                return -1

        else:
            return -1

    def __del__(self):
        """ Destructor Method """
        self.close()


























        import sys
import platform
import os
import math

######################### MODULAR CONTAINER CLASS #########################

class QLabsRealTime:
    """ The QLabsRealTime class is a collection of methods to start and stop pre-compiled real-time models to support open worlds in QLabs."""

    _URIPort = 17001

    # Initialize class
    def __init__(self):
       """ Constructor Method """
       return

    def start_real_time_model(self, modelName, actorNumber=0, QLabsHostName='localhost', userArguments=True, additionalArguments=""):
        """Starts pre-compiled real-time code made with QUARC or the Quanser APIs that has been designed to provide a real-time dynamic model and a virtual hardware interface. This function is for local execution only, but QLabs can still be running remotely.

        :param modelName: Filename of the model without extension.
        :param actorNumber: (Optional) The user defined identifier corresponding with a spawned actor of the same class and actor number. Only used for models with "workspace" in the model name.
        :param QLabsHostName: (Optional) The host name or IP address of the machine running QLabs. Only used for models with "workspace" in the model name.
        :param userArguments: (Optional) Enables using non-standard device unmbers and uris
        :param additionalArguments: (Optional) See QUARC documentation for additional quarc_run arguments.
        :type modelName: string
        :type actorNumber: uint32
        :type QLabsHostName: string
        :type additionalArguments: string
        :return: If the platform is supported, returns the command line used to start the model execution.
        :rtype: string

        """
        qlabs_rt_model = False
        if 'workspace' in modelName.lower() or 'spawn' in modelName.lower(): qlabs_rt_model = True

        if platform.system() == "Windows":
            if qlabs_rt_model:
                self._URIPort = 17001 + actorNumber
                if userArguments:
                    # this is a qlabs rt model, and use the QLabsHostName, _URIPort and actorNumber parameters
                    cmdString="start \"QLabs_{}_{}\" \"%QUARC_DIR%\\quarc_run\" -D -r -t tcpip://localhost:17000 \"{}.rt-win64\" -uri tcpip://localhost:{} -hostname {} -devicenum {} {}".format(modelName, actorNumber, modelName, self._URIPort, QLabsHostName, actorNumber, additionalArguments)
                else:
                    # this is a qlabs rt model, but don't use additional parameters
                    cmdString="start \"QLabs_{}_{}\" \"%QUARC_DIR%\\quarc_run\" -D -r -t tcpip://localhost:17000 \"{}.rt-win64\" {}".format(modelName, actorNumber, modelName, additionalArguments)

            else:
                # this is not a qlabs rt model, but a generic one for windows
                cmdString="start \"Generic_{}\" \"%QUARC_DIR%\\quarc_run\" -D -r -t tcpip://localhost:17000 \"{}.rt-win64\" {}".format(modelName, modelName, additionalArguments)
        elif platform.system() == "Linux":
            if platform.machine() == "armv7l":
                if qlabs_rt_model:
                    #Raspberry Pi 3, 4
                    if userArguments:
                        # this is a qlabs rt model, and use the QLabsHostName, _URIPort and actorNumber parameters
                        cmdString="quarc_run -D -r -t tcpip://localhost:17000 {}.rt-linux_pi_3 -uri tcpip://localhost:{} -hostname {} -devicenum {} {}".format(modelName, self._URIPort, QLabsHostName, actorNumber, additionalArguments)
                    else:
                        # this is a qlabs rt model, but don't use additional parameters
                        cmdString="quarc_run -D -r -t tcpip://localhost:17000 {}.rt-linux_pi_3 {}".format(modelName, additionalArguments)
                else:
                    print("This method cannot be used to deploy generic real-time models to this platform. Please refer to the QUARC command line tools documentation for more information.")
            else:
                print("This Linux machine not supported for real-time model execution")
                return
        else:
            if qlabs_rt_model:
                print("Platform not supported for real-time model execution")
            else:
                print("This method cannot be used to deploy generic real-time models to this platform. Please refer to the QUARC command line tools documentation for more information.")
            return

        os.system(cmdString)

        self._URIPort = self._URIPort + 1
        return cmdString

    def terminate_real_time_model(self, modelName, additionalArguments=''):
        """Stops a real-time model specified by name that is currently running.

        :param modelName: Filename of the model without extension.
        :param additionalArguments: (Optional) See QUARC documentation for additional quarc_run arguments.
        :type modelName: string
        :type additionalArguments: string
        :return: If the platform is supported, returns the command line used to stop the model execution.
        :rtype: string

        """
        if platform.system() == "Windows":
            cmdString="start \"QLabs_Spawn_Model\" \"%QUARC_DIR%\\quarc_run\" -q -Q -t tcpip://localhost:17000 {}.rt-win64 {}".format(modelName, additionalArguments)
        elif platform.system() == "Linux":
            if platform.machine() == "armv7l":
                cmdString="quarc_run -q -Q -t tcpip://localhost:17000 {}.rt-linux_pi_3 {}".format(modelName, additionalArguments)
            else:
                print("This Linux machine not supported for real-time model execution")
                return

        else:
            print("Platform not supported for real-time model execution")
            return

        os.system(cmdString)
        return cmdString

    def terminate_all_real_time_models(self, additionalArguments=''):
        """Stops all real-time models currently running.

        :param additionalArguments: (Optional) See QUARC documentation for additional quarc_run arguments.
        :type additionalArguments: string
        :return: If the platform is supported, returns the command line used to stop the model execution.
        :rtype: string

        """
        if platform.system() == "Windows":
            cmdString="start \"QLabs_Spawn_Model\" \"%QUARC_DIR%\\quarc_run\" -q -Q -t tcpip://localhost:17000 *.rt-win64 {}".format(additionalArguments)
        elif platform.system() == "Linux":
            if platform.machine() == "armv7l":
                cmdString="quarc_run -q -Q -t tcpip://localhost:17000 *.rt-linux_pi_3 {}".format(additionalArguments)
            else:
                print("This Linux machine not supported for real-time model execution")
                return

        else:
            print("Platform not supported for real-time model execution")
            return

        os.system(cmdString)
        return cmdString
    



















    from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import math
import struct
import cv2
import numpy as np


######################### MODULAR CONTAINER CLASS #########################

class QLabsReferenceFrame(QLabsActor):
    """ This class supports the spawning of reference frame actors in the QLabs open worlds."""

    ID_REFERENCE_FRAME = 10040
    """Class ID"""
    FCN_REFERENCE_FRAME_SET_TRANSFORM = 10
    FCN_REFERENCE_FRAME_SET_TRANSFORM_ACK = 11
    FCN_REFERENCE_FRAME_SET_ICON_SCALE = 12
    FCN_REFERENCE_FRAME_SET_ICON_SCALE_ACK = 13


    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_REFERENCE_FRAME
       return

    def set_transform(self, location, rotation, scale, waitForConfirmation=True):
        """
        Change the location, rotation, and scale of a spawned reference frame in radians

        :param location: An array of floats for x, y and z coordinates
        :param rotation: An array of floats for the roll, pitch, yaw in radians
        :param scale: An array of floats for x, y and z coordinates
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the operation has occurred.
        :type location: array[3]
        :type rotation: array[3]
        :type scale: array[3]
        :type waitForConfirmation: boolean
        :return: `True` if spawn was successful, `False` otherwise
        :rtype: boolean

        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_REFERENCE_FRAME
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_REFERENCE_FRAME_SET_TRANSFORM
        c.payload = bytearray(struct.pack(">fffffffff", location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], scale[0], scale[1], scale[2]))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_REFERENCE_FRAME, self.actorNumber, self.FCN_REFERENCE_FRAME_SET_TRANSFORM_ACK)
                if (c == None):
                    return False
                else:
                    return True
            return True
        else:
            return False

    def set_transform_degrees(self, location, rotation, scale, waitForConfirmation=True):
        """
        Change the location and rotation of a spawned reference frame in degrees

        :param location: An array of floats for x, y and z coordinates
        :param rotation: An array of floats for the roll, pitch, yaw in degrees
        :param scale: An array of floats for x, y and z coordinates
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the operation has occurred.
        :type location: array[3]
        :type rotation: array[3]
        :type scale: array[3]
        :type waitForConfirmation: boolean
        :return: `True` if spawn was successful, `False` otherwise
        :rtype: boolean

        """
        return self.set_transform(location, [rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi], scale, waitForConfirmation)

    def set_icon_scale(self, scale, waitForConfirmation=True):
        """
        Change the scale of the axis icon only (if a visible configuration was selected) relative to the actor scale. This scale will not affect any child actors.

        :param scale: An array of floats for x, y and z coordinates
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the operation has occurred.
        :type scale: array[3]
        :type waitForConfirmation: boolean
        :return: `True` if successful, `False` otherwise
        :rtype: boolean

        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_REFERENCE_FRAME
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_REFERENCE_FRAME_SET_ICON_SCALE
        c.payload = bytearray(struct.pack(">fff", scale[0], scale[1], scale[2]))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_REFERENCE_FRAME, self.actorNumber, self.FCN_REFERENCE_FRAME_SET_ICON_SCALE_ACK)
                if (c == None):
                    return False
                else:
                    return True
            return True
        else:
            return False


















from qvl.actor import QLabsActor
import math
import struct

class QLabsRoundaboutSign(QLabsActor):
    """This class is for spawning roundabout signs."""

    ID_ROUNDABOUT_SIGN = 10060
    """Class ID"""

    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_ROUNDABOUT_SIGN
       return





















from qvl.qlabs import QuanserInteractiveLabs, CommModularContainer
from quanser.common import GenericError
from qvl.actor import QLabsActor
import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsShredder(QLabsActor):


    ID_SHREDDER = 190

    RED = 0
    GREEN = 1
    BLUE = 2
    WHITE = 3

    # Initialize class
    def __init__(self, qlabs, verbose=False):
        """ Constructor Method

        :param qlabs: A QuanserInteractiveLabs object
        :param verbose: (Optional) Print error information to the console.
        :type qlabs: object
        :type verbose: boolean
        """
        
        self._qlabs = qlabs
        self._verbose = verbose
        self.classID = self.ID_SHREDDER
        return


















from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import math
import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsSplineLine(QLabsActor):


    ID_SPLINE_LINE = 180
    """Class ID"""

    FCN_SPLINE_LINE_SET_POINTS = 12
    FCN_SPLINE_LINE_SET_POINTS_ACK = 13

    LINEAR = 0
    """See configurations"""
    CURVE = 1
    """See configurations"""
    CONSTANT = 2
    """See configurations"""
    CLAMPED_CURVE = 3
    """See configurations"""


    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_SPLINE_LINE
       return

    def set_points(self, color, pointList, alignEndPointTangents=False, waitForConfirmation=True):
        """After spawning the origin of the spline actor, this method is used to create the individual points. At least 2 points must be specified to make a line.

        :param color: Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :param pointList: A 2D array with each row containing [x,y,z,width] elements. Width is in m.
        :param alignEndPointTangents: (Optional) Sets the tangent of the first and last point to be the same.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type color: float array[3]
        :type pointList: float 2D array[4][n]
        :type alignEndPointTangents: boolean
        :type waitForConfirmation: boolean
        :return:
            - **status** - `True` if successful, `False` otherwise
        :rtype: boolean
        """
        c = CommModularContainer()
        c.classID = self.ID_SPLINE_LINE
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_SPLINE_LINE_SET_POINTS
        c.payload = bytearray(struct.pack(">fffB", color[0], color[1], color[2], alignEndPointTangents))

        for point in pointList:
            c.payload = c.payload + bytearray(struct.pack(">ffff", point[0], point[1], point[2], point[3]))


        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_SPLINE_LINE, self.actorNumber, self.FCN_SPLINE_LINE_SET_POINTS_ACK)
                if c == None:
                    if (self._verbose):
                        print('spawn_id: Communication timeout (spline classID {}, actorNumber {}).'.format(self.classID, c.actorNumber))
                    return False

            return True
        else:
            if (self._verbose):
                print('spawn_id: Communication failed (spline classID {}, actorNumber {}).'.format(self.classID, c.actorNumber))
            return False

    def circle_from_center(self, radius, lineWidth=0.1, color=[1,0,0], numSplinePoints=8, waitForConfirmation=True):
        """After spawning the origin of the spline actor, this method is used to create a circle. Configuration 1 is recommended when spawning the line.

        :param radius: Radius in m
        :param lineWidth: Line width in m
        :param color: Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :param numSplinePoints: The number of points distributed around the circle. Splines will automatically round the edges, but more points will be needed for larger circles to achieve an accurate circle.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type radius: float
        :type lineWidth: float
        :type color: float array[3]
        :type numSplinePoints: integer
        :type waitForConfirmation: boolean
        :return:
            - **status** - `True` if successful, `False` otherwise
        :rtype: boolean
        """
        _waitForConfirmation = waitForConfirmation;
        points = []
        for angle in range(0, numSplinePoints):
            points.append([radius*math.sin(angle/numSplinePoints*math.pi*2), radius*math.cos(angle/numSplinePoints*math.pi*2), 0, lineWidth])

        points.append(points[0])

        return self.set_points(color, pointList=points, alignEndPointTangents=True, waitForConfirmation=_waitForConfirmation)

    def arc_from_center(self, radius, startAngle=0, endAngle=math.pi/2, lineWidth=1, color=[1,0,0], numSplinePoints=8, waitForConfirmation=True):

        """After spawning the origin of the spline actor, this method is used to create an arc. Configuration 1 is recommended when spawning the line.

        :param radius: Radius in m
        :param startAngle: Angle relative to the spawn orientation in radians
        :param endAngle: Angle relative to the spawn orientation in radians
        :param lineWidth: Line width in m
        :param color: Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :param numSplinePoints: The number of points distributed around the circle. Splines will automatically round the edges, but more points will be needed for larger circles to achieve an accurate circle.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type radius: float
        :type startAngle: float
        :type endAngle: float
        :type lineWidth: float
        :type color: float array[3]
        :type numSplinePoints: integer
        :type waitForConfirmation: boolean
        :return:
            - **status** - `True` if successful, `False` otherwise
        :rtype: boolean
        """

        points = []

        for angle in range(0, numSplinePoints+1):
            points.append([radius*math.cos(angle/numSplinePoints*(endAngle-startAngle)+startAngle), radius*math.sin(angle/numSplinePoints*(endAngle-startAngle)+startAngle), 0, lineWidth])

        return self.set_points(color, pointList=points, alignEndPointTangents=False)

    def arc_from_center_degrees(self, radius, startAngle=0, endAngle=90, lineWidth=1, color=[1,0,0], numSplinePoints=4, waitForConfirmation=True):
        """After spawning the origin of the spline actor, this method is used to create an arc. Configuration 1 is recommended when spawning the line.

        :param radius: Radius in m
        :param startAngle: Angle relative to the spawn orientation in degrees
        :param endAngle: Angle relative to the spawn orientation in degrees
        :param lineWidth: Line width in m
        :param color: Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :param numSplinePoints: The number of points distributed around the circle. Splines will automatically round the edges, but more points will be needed for larger circles to achieve an accurate circle.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type radius: float
        :type startAngle: float
        :type endAngle: float
        :type lineWidth: float
        :type color: float array[3]
        :type numSplinePoints: integer
        :type waitForConfirmation: boolean
        :return:
            - **status** - `True` if successful, `False` otherwise
        :rtype: boolean
        """

        return  self.arc_from_center(radius, startAngle/180*math.pi, endAngle/180*math.pi, lineWidth, color, numSplinePoints, waitForConfirmation)

    def rounded_rectangle_from_center(self, cornerRadius, xWidth, yLength, lineWidth=0.1, color=[1,0,0], waitForConfirmation=True):

        """After spawning the origin of the spline actor, this method is used to create a rounded rectangle. Configuration 1 is recommended when spawning the line.

        :param cornerRadius: Corner radius in m
        :param xWidth: Dimension in m of the rectangle in the local x axis
        :param yLength: Dimension in m of the rectangle in the local y axis
        :param lineWidth: Line width in m
        :param color: Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type cornerRadius: float
        :type xWidth: float
        :type yLength: float
        :type lineWidth: float
        :type color: float array[3]
        :type numSplinePoints: integer
        :type waitForConfirmation: boolean
        :return:
            - **status** - `True` if successful, `False` otherwise
        :rtype: boolean
        """

        points = self._spawn_spline_rounded_rectangle_from_center_point_list(cornerRadius, xWidth, yLength, lineWidth)

        return self.set_points(color, pointList=points, alignEndPointTangents=True)

    def _spawn_spline_rounded_rectangle_from_center_point_list(self, cornerRadius, xWidth, yLength, lineWidth=1):
        if (xWidth <= cornerRadius*2):
            xWidth = cornerRadius*2

        if (yLength <= cornerRadius*2):
            yLength = cornerRadius*2

        circleSegmentLength = math.pi*cornerRadius*2/8

        xCount = math.ceil((xWidth - 2*cornerRadius)/circleSegmentLength)
        yCount = math.ceil((yLength - 2*cornerRadius)/circleSegmentLength)

        # Y
        # ▲
        # │
        # ┼───► X
        #
        #   4───────3
        #   │       │
        #   │   ┼   │
        #   │       │
        #   1───────2

        offset225deg = cornerRadius-cornerRadius*math.sin(math.pi/8)
        offset45deg = cornerRadius-cornerRadius*math.sin(math.pi/8*2)
        offset675deg = cornerRadius-cornerRadius*math.sin(math.pi/8*3)

        # corner 1
        points = []
        points.append([-xWidth/2, -yLength/2+cornerRadius, 0, lineWidth])
        points.append([-xWidth/2+offset675deg, -yLength/2+offset225deg, 0, lineWidth])
        points.append([-xWidth/2+offset45deg, -yLength/2+offset45deg, 0, lineWidth])
        points.append([-xWidth/2+offset225deg, -yLength/2+offset675deg, 0, lineWidth])
        points.append([-xWidth/2+cornerRadius,-yLength/2, 0, lineWidth])

        # x1
        if (xWidth > cornerRadius*2):
            sideSegmentLength = (xWidth - 2*cornerRadius)/xCount

            for sideCount in range(1,xCount):
                 points.append([-xWidth/2+cornerRadius + sideCount*sideSegmentLength,-yLength/2, 0, lineWidth])

            points.append([xWidth/2-cornerRadius,-yLength/2, 0, lineWidth])

        # corner 2
        points.append([xWidth/2-offset225deg, -yLength/2+offset675deg, 0, lineWidth])
        points.append([xWidth/2-offset45deg, -yLength/2+offset45deg, 0, lineWidth])
        points.append([xWidth/2-offset675deg, -yLength/2+offset225deg, 0, lineWidth])
        points.append([xWidth/2, -yLength/2+cornerRadius, 0, lineWidth])

        # y1
        if (yLength > cornerRadius*2):
            sideSegmentLength = (yLength - 2*cornerRadius)/yCount

            for sideCount in range(1,yCount):
                points.append([xWidth/2, -yLength/2+cornerRadius  + sideCount*sideSegmentLength, 0, lineWidth])

            points.append([xWidth/2, yLength/2-cornerRadius, 0, lineWidth])

        # corner 3
        points.append([xWidth/2-offset675deg, yLength/2-offset225deg, 0, lineWidth])
        points.append([xWidth/2-offset45deg, yLength/2-offset45deg, 0, lineWidth])
        points.append([xWidth/2-offset225deg, yLength/2-offset675deg, 0, lineWidth])
        points.append([xWidth/2-cornerRadius, yLength/2, 0, lineWidth])

        # x2
        if (xWidth > cornerRadius*2):
            sideSegmentLength = (xWidth - 2*cornerRadius)/xCount

            for sideCount in range(1,xCount):
                points.append([xWidth/2-cornerRadius - sideCount*sideSegmentLength, yLength/2, 0, lineWidth])

            points.append([-xWidth/2+cornerRadius, yLength/2, 0, lineWidth])

        # corner 4
        points.append([-xWidth/2+offset225deg, yLength/2-offset675deg, 0, lineWidth])
        points.append([-xWidth/2+offset45deg, yLength/2-offset45deg, 0, lineWidth])
        points.append([-xWidth/2+offset675deg, yLength/2-offset225deg, 0, lineWidth])
        points.append([-xWidth/2, yLength/2-cornerRadius, 0, lineWidth])

        # y2
        if (yLength > cornerRadius*2):
            sideSegmentLength = (yLength - 2*cornerRadius)/yCount

            for sideCount in range(1,yCount):
                points.append([-xWidth/2, yLength/2-cornerRadius - sideCount*sideSegmentLength, 0, lineWidth])

            points.append(points[0])

        return points















from qvl.qlabs import QuanserInteractiveLabs, CommModularContainer
from quanser.common import GenericError
import math
import os

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsSRV02:


    ID_SRV02 = 40

    FCN_SRV02_COMMAND_AND_REQUEST_STATE = 10
    FCN_SRV02_COMMAND_AND_REQUEST_STATE_RESPONSE = 11


    # Initialize class
    def __init__(self):

       return

    def spawn(self, qlabs, actorNumber, location, rotation, configuration=0, waitForConfirmation=True):
        return qlabs.spawn(actorNumber, self.ID_SRV02, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], 1.0, 1.0, 1.0, configuration, waitForConfirmation)

    def spawn_degrees(self, qlabs, actorNumber, location, rotation, configuration=0, waitForConfirmation=True):

        return qlabs.spawn(actorNumber, self.ID_SRV02, location[0], location[1], location[2], rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi, 1.0, 1.0, 1.0, configuration, waitForConfirmation)

    def command_and_request_state(self, qlabs, actorNumber, angle, waitForConfirmation=True):
        c = CommModularContainer()
        c.classID = self.ID_SRV02
        c.actorNumber = actorNumber
        c.actorFunction = self.FCN_SRV02_COMMAND_AND_REQUEST_STATE
        c.payload = bytearray(struct.pack(">ff", angle, 0))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        qlabs.flush_receive()

        if (qlabs.send_container(c)):
            if (waitForConfirmation):
                c = qlabs.wait_for_container(self.ID_SRV02, actorNumber, self.FCN_SRV02_COMMAND_AND_REQUEST_STATE_RESPONSE)

            return True
        else:
            return False

    def command_and_request_state_degrees(self, qlabs, actorNumber, angle, waitForConfirmation=True):

        return self.command_and_request_state(qlabs, actorNumber, angle/180*math.pi, waitForConfirmation)
















from qvl.actor import QLabsActor
import math
import struct

class QLabsStopSign(QLabsActor):
    """This class is for spawning stop signs."""

    ID_STOP_SIGN = 10020
    """Class ID"""

    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_STOP_SIGN
       return























from qvl.qlabs import CommModularContainer

import math
import struct

######################### MODULAR CONTAINER CLASS #########################

class QLabsSystem:
    """The System is a special class that allows you to modify elements of the user interface and application."""

    ID_SYSTEM = 1000
    """Class ID."""
    FCN_SYSTEM_SET_TITLE_STRING = 10
    FCN_SYSTEM_SET_TITLE_STRING_ACK = 11
    FCN_SYSTEM_EXIT_APP = 100
    FCN_SYSTEM_EXIT_APP_ACK = 101

    _qlabs = None
    _verbose = False

    def __init__(self, qlabs, verbose=False):
       """ Constructor method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       return

    def set_title_string(self, titleString, waitForConfirmation=True):
        """Sets the title string in the upper left of the window to custom text. This can be useful when doing screen recordings or labeling experiment configurations.

        :param titleString: User defined string to replace the default title text
        :param waitForConfirmation: (Optional) Wait for confirmation of the before proceeding. This makes the method a blocking operation.
        :type titleString: string
        :type waitForConfirmation: boolean
        :return: `True` if successful, `False` otherwise.
        :rtype: boolean
        """
        c = CommModularContainer()
        c.classID = self.ID_SYSTEM
        c.actorNumber = 0
        c.actorFunction = self.FCN_SYSTEM_SET_TITLE_STRING
        c.payload = bytearray(struct.pack(">I", len(titleString)))
        c.payload = c.payload + bytearray(titleString.encode('utf-8'))

        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):

            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_SYSTEM, 0, self.FCN_SYSTEM_SET_TITLE_STRING_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False
        
    def exit_application(self, delay, waitForConfirmation=True):
        """Requests QLabs to exit after the specified time delay.

        :param delay: Delay time before the application exits
        :param waitForConfirmation: (Optional) Wait for confirmation of the before proceeding. This makes the method a blocking operation.
        :type titleString: float
        :type waitForConfirmation: boolean
        :return: `True` if successful, `False` otherwise.
        :rtype: boolean
        """
        c = CommModularContainer()
        c.classID = self.ID_SYSTEM
        c.actorNumber = 0
        c.actorFunction = self.FCN_SYSTEM_EXIT_APP
        c.payload = bytearray(struct.pack(">f", delay))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):

            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_SYSTEM, 0, self.FCN_SYSTEM_SET_TITLE_STRING_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False        
        



























        from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import math
import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsTrafficCone(QLabsActor):
    """This class is for spawning traffic cones."""

    ID_TRAFFIC_CONE = 10000
    """Class ID"""
    
    FCN_TRAFFIC_CONE_SET_MATERIAL_PROPERTIES = 10
    FCN_TRAFFIC_CONE_SET_MATERIAL_PROPERTIES_ACK = 11
    FCN_TRAFFIC_CONE_GET_MATERIAL_PROPERTIES = 12
    FCN_TRAFFIC_CONE_GET_MATERIAL_PROPERTIES_RESPONSE = 13
    

    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_TRAFFIC_CONE
       return
       
    def set_material_properties(self, materialSlot=0, color=[0,0,0], roughness=0.4, metallic=False, waitForConfirmation=True):
        """Sets the visual surface properties of the cone. The default colors are orange for material slot 0, and black for slot 1.

        :param materialSlot: Material index to be modified.  Setting an index for an unsupported slot will be ignored.
        :param color: Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :param roughness: A value between 0.0 (completely smooth and reflective) to 1.0 (completely rough and diffuse). Note that reflections are rendered using screen space reflections. Only objects visible in the camera view will be rendered in the reflection of the object.
        :param metallic: Metallic (True) or non-metallic (False)
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.
        :type color: byte
        :type color: float array[3]
        :type roughness: float
        :type metallic: boolean
        :type waitForConfirmation: boolean
        :return: True if successful, False otherwise
        :rtype: boolean

        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_TRAFFIC_CONE
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_TRAFFIC_CONE_SET_MATERIAL_PROPERTIES
        c.payload = bytearray(struct.pack(">BffffB", materialSlot, color[0], color[1], color[2], roughness, metallic))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_TRAFFIC_CONE, self.actorNumber, self.FCN_TRAFFIC_CONE_SET_MATERIAL_PROPERTIES_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False
            
   


















   from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsTrafficLight(QLabsActor):


    ID_TRAFFIC_LIGHT = 10051
    """Class ID"""

    FCN_TRAFFIC_LIGHT_SET_STATE = 10
    FCN_TRAFFIC_LIGHT_SET_STATE_ACK = 11
    FCN_TRAFFIC_LIGHT_SET_COLOR = 12
    FCN_TRAFFIC_LIGHT_SET_COLOR_ACK = 13
    FCN_TRAFFIC_LIGHT_GET_COLOR = 14
    FCN_TRAFFIC_LIGHT_GET_COLOR_RESPONSE = 15



    STATE_RED = 0
    """State constant for red light"""
    STATE_GREEN = 1
    """State constant for green light"""
    STATE_YELLOW = 2
    """State constant for yellow light"""

    deprecation_warned = False


    COLOR_NONE = 0
    """Color constant for all lights off"""

    COLOR_RED = 1
    """Color constant for red light"""

    COLOR_YELLOW = 2
    """Color constant for yellow light"""

    COLOR_GREEN = 3
    """Color constant for green light"""



    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_TRAFFIC_LIGHT
       return

    def set_state(self, state, waitForConfirmation=True):
        """DEPRECATED. Please use set_color instead. This method sets the light state (red/yellow/green) of a traffic light actor.

        :param state: An integer constant corresponding to a light state (see class constants)
        :param waitForConfirmation: (Optional) Wait for confirmation of the state change before proceeding. This makes the method a blocking operation.
        :type state: uint32
        :type waitForConfirmation: boolean
        :return: `True` if successful, `False` otherwise
        :rtype: boolean

        """

        if (not self._is_actor_number_valid()):
            return False
        
        if self.deprecation_warned == False:
            print("The set_state method and the STATE member constants have been deprecated and will be removed in a future version of the API. Please use set_color with the COLOR member constants instead.")
            self.deprecation_warned = True

        c = CommModularContainer()
        c.classID = self.ID_TRAFFIC_LIGHT
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_TRAFFIC_LIGHT_SET_STATE
        c.payload = bytearray(struct.pack(">B", state))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_TRAFFIC_LIGHT, self.actorNumber, self.FCN_TRAFFIC_LIGHT_SET_STATE_ACK)
                if (c == None):
                    return False

            return True
        else:
            return False
        

    def set_color(self, color, waitForConfirmation=True):
        """Set the light color index of a traffic light actor

        :param color: An integer constant corresponding to a light color index (see class constants)
        :param waitForConfirmation: (Optional) Wait for confirmation of the color change before proceeding. This makes the method a blocking operation.
        :type color: uint32
        :type waitForConfirmation: boolean
        :return: `True` if successful, `False` otherwise
        :rtype: boolean

        """

        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_TRAFFIC_LIGHT
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_TRAFFIC_LIGHT_SET_COLOR
        c.payload = bytearray(struct.pack(">B", color))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_TRAFFIC_LIGHT, self.actorNumber, self.FCN_TRAFFIC_LIGHT_SET_COLOR_ACK)
                if (c == None):
                    return False

            return True
        else:
            return False        
        
    def get_color(self):
        """Get the light color index of a traffic light actor

        :return:
            - **status** - `True` if successful, `False` otherwise
            - **color** - Color index. The color index is only valid if status is true.

        :rtype: boolean, uint32        

        """

        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_TRAFFIC_LIGHT
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_TRAFFIC_LIGHT_GET_COLOR
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(self.ID_TRAFFIC_LIGHT, self.actorNumber, self.FCN_TRAFFIC_LIGHT_GET_COLOR_RESPONSE)
            if (c == None):
              return False, 0
            
            if len(c.payload) == 1:
                return True, c.payload[0]
            else:
                return False, 0

        else:
            return False, 0























from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import numpy as np
import math
import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsWalls(QLabsActor):
    """ This class is for spawning both static and dynamic walls."""

    ID_WALL = 10080
    """Class ID"""

    WALL_FOAM_BOARD = 0

    COMBINE_AVERAGE = 0
    COMBINE_MIN = 1
    COMBINE_MULTIPLY = 2
    COMBINE_MAX = 3

    FCN_WALLS_ENABLE_DYNAMICS = 14
    FCN_WALLS_ENABLE_DYNAMICS_ACK = 15
    FCN_WALLS_SET_TRANSFORM = 16
    FCN_WALLS_SET_TRANSFORM_ACK = 17
    FCN_WALLS_ENABLE_COLLISIONS = 18
    FCN_WALLS_ENABLE_COLLISIONS_ACK = 19
    FCN_WALLS_SET_PHYSICS_PROPERTIES = 20
    FCN_WALLS_SET_PHYSICS_PROPERTIES_ACK = 21

    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_WALL
       return

    def set_enable_dynamics(self, enableDynamics, waitForConfirmation=True):
        """Sets the physics properties of the wall.

        :param enableDynamics: Enable (True) or disable (False) the wall dynamics. A dynamic actor can be pushed with other static or dynamic actors.  A static actor will generate collisions, but will not be affected by interactions with other actors.
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.
        :type enableDynamics: boolean
        :type waitForConfirmation: boolean
        :return: True if successful, False otherwise
        :rtype: boolean

        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_WALL
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_WALLS_ENABLE_DYNAMICS
        c.payload = bytearray(struct.pack(">B", enableDynamics))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_WALL, self.actorNumber, self.FCN_WALLS_ENABLE_DYNAMICS_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False

    def set_enable_collisions(self, enableCollisions, waitForConfirmation=True):
        """Enables and disables physics collisions. When disabled, other physics or velocity-based actors will be able to pass through.

        :param enableCollisions: Enable (True) or disable (False) the collision.
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.
        :type enableCollisions: boolean
        :type waitForConfirmation: boolean
        :return: True if successful, False otherwise
        :rtype: boolean

        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_WALL
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_WALLS_ENABLE_COLLISIONS
        c.payload = bytearray(struct.pack(">B", enableCollisions))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_WALL, self.actorNumber, self.FCN_WALLS_ENABLE_COLLISIONS_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False

    def set_physics_properties(self, enableDynamics, mass=1.0, linearDamping=0.01, angularDamping=0.0, staticFriction=0.0, dynamicFriction=0.7, frictionCombineMode=COMBINE_AVERAGE, restitution=0.3, restitutionCombineMode=COMBINE_AVERAGE, waitForConfirmation=True):
        """Sets the dynamic properties of the wall.

        :param enableDynamics: Enable (True) or disable (False) the wall dynamics. A dynamic actor can be pushed with other static or dynamic actors.  A static actor will generate collisions, but will not be affected by interactions with other actors.
        :param mass: (Optional) Sets the mass of the actor in kilograms.
        :param linearDamping: (Optional) Sets the damping of the actor for linear motions.
        :param angularDamping: (Optional) Sets the damping of the actor for angular motions.
        :param staticFriction: (Optional) Sets the coefficient of friction when the actor is at rest. A value of 0.0 is frictionless.
        :param dynamicFriction: (Optional) Sets the coefficient of friction when the actor is moving relative to the surface it is on. A value of 0.0 is frictionless.
        :param frictionCombineMode: (Optional) Defines how the friction between two surfaces with different coefficients should be calculated (see COMBINE constants).
        :param restitution: (Optional) The coefficient of restitution defines how plastic or elastic a collision is. A value of 0.0 is plastic and will absorb all energy. A value of 1.0 is elastic and will bounce forever. A value greater than 1.0 will add energy with each collision.
        :param restitutionCombineMode: (Optional) Defines how the restitution between two surfaces with different coefficients should be calculated (see COMBINE constants).
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.

        :type enableDynamics: boolean
        :type mass: float
        :type linearDamping: float
        :type angularDamping: float
        :type staticFriction: float
        :type dynamicFriction: float
        :type frictionCombineMode: byte
        :type restitution: float
        :type restitutionCombineMode: byte
        :type waitForConfirmation: boolean
        :return: True if successful, False otherwise
        :rtype: boolean

        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_WALL
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_WALLS_SET_PHYSICS_PROPERTIES
        c.payload = bytearray(struct.pack(">BfffffBfB", enableDynamics, mass, linearDamping, angularDamping, staticFriction, dynamicFriction, frictionCombineMode, restitution, restitutionCombineMode))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_WALL, self.actorNumber, self.FCN_WALLS_SET_PHYSICS_PROPERTIES_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False

    def set_transform(self, location, rotation, scale, waitForConfirmation=True):
        """Sets the location, rotation in radians, and scale. If a wall is parented to another actor then the location, rotation, and scale are relative to the parent actor.

        :param location: An array of floats for x, y and z coordinates in full-scale units. Multiply physical QCar locations by 10 to get full scale locations.
        :param rotation: An array of floats for the roll, pitch, and yaw in radians
        :param scale: An array of floats for the scale in the x, y, and z directions.
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type waitForConfirmation: boolean
        :return: True if successful or False otherwise
        :rtype: boolean
        """
        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_WALL
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_WALLS_SET_TRANSFORM
        c.payload = bytearray(struct.pack(">fffffffff", location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], scale[0], scale[1], scale[2]))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_WALL, self.actorNumber, self.FCN_WALLS_SET_TRANSFORM_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False

    def set_transform_degrees(self, location, rotation, scale, waitForConfirmation=True):
        """Sets the location, rotation in degrees, and scale. If a wall is parented to another actor then the location, rotation, and scale are relative to the parent actor.

        :param location: An array of floats for x, y and z coordinates in full-scale units. Multiply physical QCar locations by 10 to get full scale locations.
        :param rotation: An array of floats for the roll, pitch, and yaw in degrees
        :param scale: An array of floats for the scale in the x, y, and z directions.
        :param waitForConfirmation: (Optional) Wait for confirmation of the operation before proceeding. This makes the method a blocking operation.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type waitForConfirmation: boolean
        :return: True if successful or False otherwise
        :rtype: boolean
        """

        return self.set_transform(location, [rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi], scale, waitForConfirmation)

























from qvl.qlabs import CommModularContainer
import math
import struct

######################### MODULAR CONTAINER CLASS #########################

class QLabsWeighScale:


    ID_WEIGH_SCALE = 120

    FCN_WEIGH_SCALE_REQUEST_LOAD_MASS = 91
    FCN_WEIGH_SCALE_RESPONSE_LOAD_MASS = 92

    # Initialize class
    def __init__(self):

       return

    def spawn(self, qlabs, actorNumber, location, rotation, waitForConfirmation=True):
        return qlabs.spawn(actorNumber, self.ID_WEIGH_SCALE, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], 1, 1, 1, 0, waitForConfirmation)

    def spawn_degrees(self, qlabs, actorNumber, location, rotation, waitForConfirmation=True):

        return qlabs.spawn(actorNumber, self.ID_WEIGH_SCALE, location[0], location[1], location[2], rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi, 1, 1, 1, 0, waitForConfirmation)

    def spawn_and_parent_with_relative_transform(self, qlabs, actorNumber, location, rotation, parentClass, parentActorNumber, parentComponent, waitForConfirmation=True):
        return qlabs.spawn_and_parent_with_relative_transform(actorNumber, self.ID_WEIGH_SCALE, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], 1, 1, 1, 0, parentClass, parentActorNumber, parentComponent, waitForConfirmation)

    def get_measured_mass(self, qlabs, actorNumber):
        c = CommModularContainer()
        c.classID = self.ID_WEIGH_SCALE
        c.actorNumber = actorNumber
        c.actorFunction = self.FCN_WEIGH_SCALE_REQUEST_LOAD_MASS
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        qlabs.flush_receive()

        if (qlabs.send_container(c)):
            c = qlabs.wait_for_container(self.ID_WEIGH_SCALE, actorNumber, self.FCN_WEIGH_SCALE_RESPONSE_LOAD_MASS)

            if (len(c.payload) == 4):
                mass,  = struct.unpack(">f", c.payload)
                return mass
            else:
                return -1.0
        else:
            return -1.0
























from qvl.qlabs import QuanserInteractiveLabs, CommModularContainer
import math
import struct

######################### MODULAR CONTAINER CLASS #########################

class QLabsWidget:
    """ This class is for the spawning of widgets."""

    FCN_GENERIC_ACTOR_SPAWNER_DESTROY_ALL_SPAWNED_WIDGETS = 18
    FCN_GENERIC_ACTOR_SPAWNER_DESTROY_ALL_SPAWNED_WIDGETS_ACK = 19
    FCN_GENERIC_ACTOR_SPAWNER_SPAWN_WIDGET = 20
    FCN_GENERIC_ACTOR_SPAWNER_SPAWN_WIDGET_ACK = 21
    FCN_GENERIC_ACTOR_SPAWNER_SPAWN_AND_PARENT_RELATIVE = 50
    FCN_GENERIC_ACTOR_SPAWNER_SPAWN_AND_PARENT_RELATIVE_ACK = 51
    FCN_GENERIC_ACTOR_SPAWNER_WIDGET_SPAWN_CONFIGURATION = 100
    FCN_GENERIC_ACTOR_SPAWNER_WIDGET_SPAWN_CONFIGURATION_ACK = 101


    CUBE = 0
    """See configurations"""
    CYLINDER = 1
    """See configurations"""
    SPHERE = 2
    """See configurations"""
    AUTOCLAVE_CAGE = 3
    PLASTIC_BOTTLE = 4
    """See configurations"""
    METAL_CAN = 5
    """See configurations"""

    _qlabs = None
    _verbose = False

    # Initialize class
    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose

       return

    def spawn(self, location, rotation, scale, configuration, color=[1,1,1], measuredMass=0, IDTag=0, properties="", waitForConfirmation=True):
        """Spawns a widget in an instance of QLabs at a specific location and rotation using radians.

        :param location: An array of floats for x, y and z coordinates.
        :param rotation: An array of floats for the roll, pitch, and yaw in radians.
        :param scale: An array of floats for the scale in the x, y, and z directions.
        :param configuration: See configuration options
        :param color: Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :param measuredMass: A float value for use with mass sensor instrumented actors. This does not alter the dynamic properties.
        :param IDTag: An integer value for use with IDTag sensor instrumented actors.
        :param properties: A string for use with properties sensor instrumented actors. This can contain any string that is available for use to parse out user-customized parameters.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type color: float array[3]
        :type measuredMass: float
        :type IDTag: uint8
        :type properties: string
        :type waitForConfirmation: boolean
        :return:
            - **status** - `True` if successful, `False` otherwise
        :rtype: boolean

        """
        c = CommModularContainer()
        c.classID = CommModularContainer.ID_GENERIC_ACTOR_SPAWNER
        c.actorNumber = 0
        c.actorFunction = self.FCN_GENERIC_ACTOR_SPAWNER_SPAWN_WIDGET
        c.payload = bytearray(struct.pack(">IfffffffffffffBI", configuration, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], scale[0], scale[1], scale[2], color[0], color[1], color[2], measuredMass, IDTag, len(properties)))
        c.payload = c.payload + bytearray(properties.encode('utf-8'))

        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):

            if waitForConfirmation:
                c = self._qlabs.wait_for_container(CommModularContainer.ID_GENERIC_ACTOR_SPAWNER, 0, self.FCN_GENERIC_ACTOR_SPAWNER_SPAWN_WIDGET_ACK)
                if (c == None):
                    return False
                else:
                    return True

            return True
        else:
            return False

    def spawn_degrees(self, location, rotation, scale, configuration, color=[1,1,1], measuredMass=0, IDTag=0, properties="", waitForConfirmation=True):
        """Spawns a widget in an instance of QLabs at a specific location and rotation using degrees.

        :param location: An array of floats for x, y and z coordinates.
        :param rotation: An array of floats for the roll, pitch, and yaw in degrees.
        :param scale: An array of floats for the scale in the x, y, and z directions.
        :param configuration: See configuration options.
        :param color: Red, Green, Blue components of the RGB color on a 0.0 to 1.0 scale.
        :param measuredMass: A float value for use with mass sensor instrumented actors. This does not alter the dynamic properties.
        :param IDTag: An integer value for use with IDTag sensor instrumented actors.
        :param properties: A string for use with properties sensor instrumented actors. This can contain any string that is available for use to parse out user-customized parameters.
        :param waitForConfirmation: (Optional) Make this operation blocking until confirmation of the spawn has occurred.
        :type location: float array[3]
        :type rotation: float array[3]
        :type scale: float array[3]
        :type configuration: uint32
        :type color: float array[3]
        :type measuredMass: float
        :type IDTag: uint8
        :type properties: string
        :type waitForConfirmation: boolean
        :return:
            - **status** - `True` if successful, `False` otherwise
        :rtype: boolean

        """
        return self.spawn(location, [rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi], scale, configuration, color, measuredMass, IDTag, properties, waitForConfirmation)

    def destroy_all_spawned_widgets(self):
        """Destroys all spawned widgets, but does not destroy actors.

        :return: `True` if successful, `False` otherwise
        :rtype: boolean

        """
        actorNumber = 0
        c = CommModularContainer()

        c.classID = CommModularContainer.ID_GENERIC_ACTOR_SPAWNER
        c.actorNumber = actorNumber
        c.actorFunction = self.FCN_GENERIC_ACTOR_SPAWNER_DESTROY_ALL_SPAWNED_WIDGETS
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(CommModularContainer.ID_GENERIC_ACTOR_SPAWNER, actorNumber, self.FCN_GENERIC_ACTOR_SPAWNER_DESTROY_ALL_SPAWNED_WIDGETS_ACK)
            if (c == None):
                return False
            else:
                return True

        else:
            return False

    def widget_spawn_shadow(self, enableShadow=True):
        """If spawning a large number of widgets causes performance degradation, you can try disabling the widget shadows. This function must be called in advance of widget spawning and all subsequence widgets will be spawned with the specified shadow setting.

        :param enableShadow: (Optional) Show (`True`) or hide (`False`) widget shadows.
        :type enableShadow: boolean
        :return: `True` if successful, `False` otherwise
        :rtype: boolean

        """
        actorNumber = 0
        c = CommModularContainer()

        c.classID = CommModularContainer.ID_GENERIC_ACTOR_SPAWNER
        c.actorNumber = actorNumber
        c.actorFunction = self.FCN_GENERIC_ACTOR_SPAWNER_WIDGET_SPAWN_CONFIGURATION
        c.payload = bytearray(struct.pack(">B", enableShadow))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if (self._qlabs.send_container(c)):
            c = self._qlabs.wait_for_container(CommModularContainer.ID_GENERIC_ACTOR_SPAWNER, actorNumber, self.FCN_GENERIC_ACTOR_SPAWNER_WIDGET_SPAWN_CONFIGURATION_ACK)
            if (c == None):
                return False
            else:
                return True

        else:
            return False















from qvl.actor import QLabsActor
import math
import struct

class QLabsYieldSign(QLabsActor):
    """This class is for spawning yield signs."""

    ID_YIELD_SIGN = 10070
    """Class ID"""

    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_YIELD_SIGN
       return



"""
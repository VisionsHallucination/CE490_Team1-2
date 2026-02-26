import sys
import os
import time
import signal
import threading
import numpy as np
import math
import heapq

# --- QLabs & Hardware Imports ---
import cv2
from qvl.qlabs import QuanserInteractiveLabs
from qvl.real_time import QLabsRealTime
from qvl.qcar2 import QLabsQCar2
from qvl.free_camera import QLabsFreeCamera
from qvl.basic_shape import QLabsBasicShape
from qvl.system import QLabsSystem
from qvl.walls import QLabsWalls
from qvl.qcar_flooring import QLabsQCarFlooring
from qvl.stop_sign import QLabsStopSign
from qvl.yield_sign import QLabsYieldSign
from qvl.roundabout_sign import QLabsRoundaboutSign
from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_light import QLabsTrafficLight
from hal.utilities.image_processing import ImageProcessing
from pal.products.qcar import QCar, QCarGPS, QCarCameras

# --- Hardware & Math Imports ---
from pal.products.qcar import QCar, QCarGPS
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
from custom_roadmap import CustomRoadMap

# --- Diagnostics ---
import matplotlib
matplotlib.use("TkAgg")   # or "TkAgg" if Qt not installed
import matplotlib.pyplot as plt
from collections import deque

# Colors
MAGENTA = [1.0, 0.0, 1.0]
GREEN   = [0.0, 1.0, 0.0]
BLUE    = [0.0, 0.0, 1.0]
ORANGE  = [1.0, 0.65, 0.0]

#Locations
PICKUP_XY  = np.array([0.125, 4.395])  
DROPOFF_XY = np.array([-0.905, 0.800]) 
HUB_XY     = np.array([-1.055, -0.93]) 

# ===========================
# ===========================
# 1. CONFIGURATION
# ===========================
# This sequence loops the outer track (adjust as needed for specific map nodes)
NODE_SEQUENCE = [
    10,
    2,
    4,
    14,
    16,
    18,
    11,
    12,
    7,
    5,
    3,
    1,
    8,
    23,
    21,
    16,
    17,
    20,
    22,
    9,
    0,
    2,
    4,
    6,
    13,
    19,
    17,
    15,
    6,
    0,
    2,
    4,
    6,
    8,
    10,
]

V_REF = 1.0  # Locked cruise speed
CONTROLLER_RATE = 100  # 100Hz loop
START_DELAY = 2.0  # Time for EKF to stabilize before moving

# Path overlay (debug visualization)
DRAW_PATH_OVERLAY = False
DRAW_ALL_ROADS = False  # Draw entire road network
PATH_SAMPLE_STEP = 10  # plot every Nth waypoint
PATH_Z = 0.02

# Initial Pose (Matches Setup_Real_Scenario default)
INITIAL_POS = [-1.205, -0.83, 0.005]
INITIAL_ROT = [0, 0, -44.7]

KILL_PROGRAM = False

offsets = []
desiredCamTrack = 300
camCorrFct = 0
offsetFrameBuf = 5
camCorrHist = deque(maxlen=300)

destHoldTime = 1 # How long to pause at destination
vStopped = .1 # What speed counts as vehicle stopped
destThd = .25 # How far can the car be while concidered at the stop

def sig_handler(*args):
    global KILL_PROGRAM
    KILL_PROGRAM = True

signal.signal(signal.SIGINT, sig_handler)

# Adding class for current state
# V1.1
# Changelog:
# 1/26/26: Created class
# 2/19/26: Add state transitions
class VehState:
    IDLE = 0
    DRIVE_EMPTY = 1
    PICKUP = 2
    DROPOFF = 3

    def __init__(self, qcar2):
        self.state = self.IDLE 
        self.qcar2 = qcar2
        self.qcar2.set_led_strip_uniform(MAGENTA)

    def update(self, req):  
        match req:
            case self.IDLE: # Idle
                if(self.state == self.DROPOFF):
                    self.state = self.IDLE
                    self.qcar2.set_led_strip_uniform(MAGENTA)
                else:
                    print("Invalid state transition")
            case self.DRIVE_EMPTY: # Drive with no passenger
                if((self.state == self.IDLE) | (self.state == self.DROPOFF)):
                    self.state = self.DRIVE_EMPTY
                    self.qcar2.set_led_strip_uniform(GREEN)
                else:
                    print("Invalid state transition")
            case self.PICKUP: # Picked up passenger
                if(self.state == self.DRIVE_EMPTY):
                    self.state = self.PICKUP
                    self.qcar2.set_led_strip_uniform(BLUE)
                else:
                    print("Invalid state transition")
            case self.DROPOFF: # Drop off passenger
                if(self.state == self.PICKUP):
                    self.state = self.DROPOFF
                    self.qcar2.set_led_strip_uniform(ORANGE)
                else:
                    print("Invalid state transition")
            case _: 
                print(f"[WARN] Invalid state request: {req}")

    def getState(self):
        return self.state

# ===========================
# 2. CONTROLLERS (From main.py)
# ===========================


class SpeedController:
    """Locked speed controller with anti-surge logic."""

    def __init__(self, kp=0.04, ki=0.15):
        self.maxThrottle = 1.0
        self.kp = kp
        self.ki = ki
        self.ei = 0

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e
        self.ei = np.clip(self.ei, -0.2, 0.2)
        return np.clip(self.kp * e + self.ki * self.ei, 0.0, self.maxThrottle)


class SteeringController:
    """Dampened Stanley Controller."""
    # input params: waypoints, proportion gain,

    def __init__(self, waypoints, k=0.4, cyclic=True):
        self.maxSteeringAngle = np.pi / 6
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.k = k
        self.cyclic = cyclic

    # input params: Current position (p), yaw angle(th), speed                                                                                                                     
    def update(self, p, th, speed):
        calc_speed = max(speed, 0.2)

        # previous waypoint                   
        wp_1 = self.wp[:, np.mod(self.wpi, self.N - 1)]
        # next waypoint               
        wp_2 = self.wp[:, np.mod(self.wpi + 1, self.N - 1)]
        # path vector             
        v_seg = wp_2 - wp_1
        # path length             
        v_mag = np.linalg.norm(v_seg)
        # path direction
        v_uv = v_seg / v_mag if v_mag > 0 else np.array([1, 0])
        #target yaw angle                 
        tangent = np.arctan2(v_uv[1], v_uv[0])
        # segment distance travelled                            
        s = np.dot(p - wp_1, v_uv)

        #load next waypoint if distance travelled > length                                                  
        if abs(s) >= v_mag:
            if self.cyclic or self.wpi < self.N - 2:
                self.wpi += 1

        
        # ep = expected current point of car                                    
        ep = wp_1 + v_uv * s
        # current path error                    
        ct = ep - p
        # current offset from centerline                                
        side_dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)
        # overall lateral distance from centerline                                          
        ect = np.linalg.norm(ct) * np.sign(side_dir)

        #if(True):
        #    if(camCorrFct > ect):
        #        ect = camCorrFct
        # yaw error           
        psi = wrap_to_pi(tangent - th)

        steering = psi + np.arctan2(self.k * ect, calc_speed)
        return np.clip(
            wrap_to_pi(steering), -self.maxSteeringAngle, self.maxSteeringAngle
        )
    def newWaypoint(self, waypoints):
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0



# ===========================
# 3. ENVIRONMENT SETUP (From Setup_Real_Scenario.py)
# ===========================


def traffic_light_logic(trafficLight1, trafficLight2, trafficLight3, trafficLight4):
    """Cycles traffic lights in a background thread."""
    intersection1Flag = 0
    while not KILL_PROGRAM:
        if intersection1Flag == 0:
            trafficLight1.set_color(color=QLabsTrafficLight.COLOR_RED)
            trafficLight3.set_color(color=QLabsTrafficLight.COLOR_RED)
            trafficLight2.set_color(color=QLabsTrafficLight.COLOR_GREEN)
            trafficLight4.set_color(color=QLabsTrafficLight.COLOR_GREEN)

        if intersection1Flag == 1:
            trafficLight1.set_color(color=QLabsTrafficLight.COLOR_RED)
            trafficLight3.set_color(color=QLabsTrafficLight.COLOR_RED)
            trafficLight2.set_color(color=QLabsTrafficLight.COLOR_YELLOW)
            trafficLight4.set_color(color=QLabsTrafficLight.COLOR_YELLOW)

        if intersection1Flag == 2:
            trafficLight1.set_color(color=QLabsTrafficLight.COLOR_GREEN)
            trafficLight3.set_color(color=QLabsTrafficLight.COLOR_GREEN)
            trafficLight2.set_color(color=QLabsTrafficLight.COLOR_RED)
            trafficLight4.set_color(color=QLabsTrafficLight.COLOR_RED)

        if intersection1Flag == 3:
            trafficLight1.set_color(color=QLabsTrafficLight.COLOR_YELLOW)
            trafficLight3.set_color(color=QLabsTrafficLight.COLOR_YELLOW)
            trafficLight2.set_color(color=QLabsTrafficLight.COLOR_RED)
            trafficLight4.set_color(color=QLabsTrafficLight.COLOR_RED)

        intersection1Flag = (intersection1Flag + 1) % 4
        time.sleep(5)


def setup_environment(qlabs, initialPosition, initialOrientation):
    """Full environment setup from Setup_Real_Scenario.py"""

    # Set the Workspace Title
    hSystem = QLabsSystem(qlabs)
    hSystem.set_title_string("Simple Node Follower", waitForConfirmation=True)

    ### Flooring
    x_offset = 0.13
    y_offset = 1.67
    hFloor = QLabsQCarFlooring(qlabs)
    hFloor.spawn_degrees([x_offset, y_offset, 0.001], rotation=[0, 0, -90])

    ### Walls
    hWall = QLabsWalls(qlabs)
    hWall.set_enable_dynamics(False)

    for y in range(5):
        hWall.spawn_degrees(
            location=[-2.4 + x_offset, (-y * 1.0) + 2.55 + y_offset, 0.001],
            rotation=[0, 0, 0],
        )
    for x in range(5):
        hWall.spawn_degrees(
            location=[-1.9 + x + x_offset, 3.05 + y_offset, 0.001], rotation=[0, 0, 90]
        )
    for y in range(6):
        hWall.spawn_degrees(
            location=[2.4 + x_offset, (-y * 1.0) + 2.55 + y_offset, 0.001],
            rotation=[0, 0, 0],
        )
    for x in range(4):
        hWall.spawn_degrees(
            location=[-0.9 + x + x_offset, -3.05 + y_offset, 0.001], rotation=[0, 0, 90]
        )

    hWall.spawn_degrees(
        location=[-2.03 + x_offset, -2.275 + y_offset, 0.001], rotation=[0, 0, 48]
    )
    hWall.spawn_degrees(
        location=[-1.575 + x_offset, -2.7 + y_offset, 0.001], rotation=[0, 0, 48]
    )

    # Spawn QCar
    car2 = QLabsQCar2(qlabs)
    car2.spawn_id(
        actorNumber=0,
        location=initialPosition,
        rotation=initialOrientation,
        scale=[0.1, 0.1, 0.1],
        configuration=0,
        waitForConfirmation=True,
    )

    # Spawn Cameras
    camera1Loc = [0.15, 1.7, 5]
    camera1Rot = [0, 90, 0]
    camera1 = QLabsFreeCamera(qlabs)
    camera1.spawn_degrees(location=camera1Loc, rotation=camera1Rot)

    # Stop Signs
    myStopSign = QLabsStopSign(qlabs)
    myStopSign.spawn_degrees(
        location=[-1.5, 3.6, 0.006],
        rotation=[0, 0, -35],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )
    myStopSign.spawn_degrees(
        location=[-1.5, 2.2, 0.006],
        rotation=[0, 0, 35],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )
    myStopSign.spawn_degrees(
        location=[2.410, 0.206, 0.006],
        rotation=[0, 0, -90],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )
    myStopSign.spawn_degrees(
        location=[1.766, 1.697, 0.006],
        rotation=[0, 0, 90],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )

    # Roundabout Signs
    myRoundaboutSign = QLabsRoundaboutSign(qlabs)
    myRoundaboutSign.spawn_degrees(
        location=[2.392, 2.522, 0.006],
        rotation=[0, 0, -90],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )
    myRoundaboutSign.spawn_degrees(
        location=[0.698, 2.483, 0.006],
        rotation=[0, 0, -145],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )
    myRoundaboutSign.spawn_degrees(
        location=[0.007, 3.973, 0.006],
        rotation=[0, 0, 135],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )

    # Yield Signs
    myYieldSign = QLabsYieldSign(qlabs)
    myYieldSign.spawn_degrees(
        location=[0.0, -1.3, 0.006],
        rotation=[0, 0, -180],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )
    myYieldSign.spawn_degrees(
        location=[2.4, 3.2, 0.006],
        rotation=[0, 0, -90],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )
    myYieldSign.spawn_degrees(
        location=[1.1, 2.8, 0.006],
        rotation=[0, 0, -145],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )
    myYieldSign.spawn_degrees(
        location=[0.49, 3.8, 0.006],
        rotation=[0, 0, 135],
        scale=[0.1, 0.1, 0.1],
        waitForConfirmation=False,
    )

    # Crosswalks
    myCrossWalk = QLabsCrosswalk(qlabs)
    myCrossWalk.spawn_degrees(
        location=[-2 + x_offset, -1.475 + y_offset, 0.01],
        rotation=[0, 0, 0],
        scale=[0.1, 0.1, 0.075],
        configuration=0,
    )
    myCrossWalk.spawn_degrees(
        location=[-0.5, 0.95, 0.006],
        rotation=[0, 0, 90],
        scale=[0.1, 0.1, 0.075],
        configuration=0,
    )
    myCrossWalk.spawn_degrees(
        location=[0.15, 0.32, 0.006],
        rotation=[0, 0, 0],
        scale=[0.1, 0.1, 0.075],
        configuration=0,
    )
    myCrossWalk.spawn_degrees(
        location=[0.75, 0.95, 0.006],
        rotation=[0, 0, 90],
        scale=[0.1, 0.1, 0.075],
        configuration=0,
    )
    myCrossWalk.spawn_degrees(
        location=[0.13, 1.57, 0.006],
        rotation=[0, 0, 0],
        scale=[0.1, 0.1, 0.075],
        configuration=0,
    )
    myCrossWalk.spawn_degrees(
        location=[1.45, 0.95, 0.006],
        rotation=[0, 0, 90],
        scale=[0.1, 0.1, 0.075],
        configuration=0,
    )

    # Splines (Line Guidance)
    mySpline = QLabsBasicShape(qlabs)
    mySpline.spawn_degrees(
        location=[2.21, 0.2, 0.006],
        rotation=[0, 0, 0],
        scale=[0.27, 0.02, 0.001],
        waitForConfirmation=False,
    )
    mySpline.spawn_degrees(
        location=[1.951, 1.68, 0.006],
        rotation=[0, 0, 0],
        scale=[0.27, 0.02, 0.001],
        waitForConfirmation=False,
    )
    mySpline.spawn_degrees(
        location=[-0.05, -1.02, 0.006],
        rotation=[0, 0, 90],
        scale=[0.38, 0.02, 0.001],
        waitForConfirmation=False,
    )

    # Start Real-Time Model
    rtModel = os.path.normpath(
        os.path.join(os.environ["RTMODELS_DIR"], "QCar2/QCar2_Workspace_studio")
    )
    QLabsRealTime().start_real_time_model(rtModel)

    return car2

def loop_path_waypoints(full_loop_wp, xy, goal_xy):
    xy = np.array(xy, dtype=float).reshape(2)
    goal_xy = np.array(goal_xy, dtype=float).reshape(2)

    pts = full_loop_wp[:2, :].T  # (N,2)

    i0 = int(np.argmin(np.linalg.norm(pts - xy[None, :], axis=1)))
    i1 = int(np.argmin(np.linalg.norm(pts - goal_xy[None, :], axis=1)))

    if i0 <= i1:
        seg = full_loop_wp[:, i0 : i1 + 1]
    else:
        # wrap around end of loop
        seg = np.hstack([full_loop_wp[:, i0:], full_loop_wp[:, : i1 + 1]])

    # Guarantee at least 2 points
    if seg.shape[1] < 2:
        seg = np.hstack([seg, seg])

    return seg

# Task pulls camera data, marks yellow/white pixes, then uses those to determine lane center
# Lane center determined by pixels on left to right closest to lane center
# V1.0
# Changelog:
# 2/23/26: Push current code, does not work 
def lane_offset_thread(camera, sleep_s=0.01):
    global KILL_PROGRAM, offsets, camCorrFct

    # HSV thresholds (tweak if needed)
    YELLOW_LO = (10, 60, 60)
    YELLOW_HI = (50, 255, 255)
    WHITE_LO  = (0, 0, 200)
    WHITE_HI  = (180, 45, 255)

    kernel = np.ones((5, 5), np.uint8)

    #Thread main                                    
    while not KILL_PROGRAM:
        # Get Frame
        camera.readAll()
        rawFrame = camera.csiLeft.imageData

        if rawFrame.ndim == 2:
            rawFrame = cv2.cvtColor(rawFrame, cv2.COLOR_GRAY2BGR)
        if rawFrame.shape[-1] == 4:
            rawFrame = rawFrame[:, :, :3]                

        roi = rawFrame[:, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)                                                
        
        # Create mask
        yellow_mask = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
        white_mask  = cv2.inRange(hsv, WHITE_LO,  WHITE_HI)
        mask = cv2.bitwise_or(yellow_mask, white_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Get mask height     
        H, W = mask.shape 
        #Grab lower half of frame
        y0 = int(H * 0.50) 
        roi = mask[y0:, :] 
        hR, wR = roi.shape 
        
        # Define a trapezoid to morph ROI
        src = np.float32([
            [wR * 0.2, hR * 0.55],  # top-left
            [wR * 0.8, hR * 0.55],  # top-right
            [wR * 1, hR * 0.98],  # bottom-right
            [wR * 0, hR * 0.98],  # bottom-left
        ])

        dst = np.float32([
            [wR - 1, hR - 1],
            [0,      hR - 1],
            [0,      0],
            [wR - 1, 0],
        ])
        # Apply Transform
        M = cv2.getPerspectiveTransform(src, dst)
        procImg = cv2.warpPerspective(roi, M, (wR, hR), flags=cv2.INTER_NEAREST)
        procImg = cv2.rotate(procImg, cv2.ROTATE_180)
        procImg = cv2.morphologyEx(procImg, cv2.MORPH_OPEN, np.ones((3,3), np.uint8)) 
        
        #Grab center, and search region
        mid = wR // 2
        third = wR // 4
        leftROI = procImg[:, :third]   

        closestX = []

        #Find rightmost signals for each row
        for row in (leftROI > 0):
            idx = np.flatnonzero(row)
            if idx.size == 0:
                continue
            
            gaps = np.where(np.diff(idx) > 10)[0] 
            if gaps.size:
                closestX.append(int(idx[gaps[0]]))
            else:
                closestX.append(int(idx[-1]))

        #Reject if lack 'hits' in frame
        if len(closestX) < 30:
            #offsets.clear()
            #camCorrFct = 0.0
            time.sleep(sleep_s)
            continue
        
        # lane marker is mean of closest edge location
        LaneX = float(np.median(closestX))
        offset = (mid - LaneX)               

        # Reject inplausable data
        if offset <= 0 or offset >= mid:
            #offsets.clear()
            #camCorrFct = 0.0
            time.sleep(sleep_s)
            continue
    
        # Update offset buffer
        offsets.append(offset) 
        if len(offsets) > offsetFrameBuf:
            offsets.pop(0)

        #Find short term average offset, to find tracking error
        avgOffset = sum(offsets) / len(offsets)
        errX = avgOffset - desiredCamTrack

        Kcam = 1 #Error gain
        #Calculate correction Factor to pass to steering controller
        camCorrFct = np.clip(Kcam * (errX / mid), -0.5, 0.5)
        time.sleep(sleep_s)


def draw_path_overlay(qlabs, waypoints, sample_step=10, z=0.02):
    """Draw a waypoint overlay using small QLabs basic shapes."""
    if waypoints is None or waypoints.size == 0:
        return

    overlay_shape = QLabsBasicShape(qlabs)
    actor_id = 200  # start id for overlay markers

    for i in range(0, waypoints.shape[1], sample_step):
        x = float(waypoints[0, i])
        y = float(waypoints[1, i])
        overlay_shape.spawn_id_degrees(
            actorNumber=actor_id,
            location=[x, y, z],
            rotation=[0, 0, 0],
            scale=[0.03, 0.03, 0.003],
            configuration=0,
            waitForConfirmation=False,
        )
        actor_id += 1


def draw_all_roads(qlabs, roadmap, sample_step=10, z=0.02):
    """Draw all edges in the roadmap as green path overlays."""
    overlay_shape = QLabsBasicShape(qlabs)
    actor_id = 1000  # start id for road network markers

    for edge in roadmap.edges:
        if edge.waypoints is None or edge.waypoints.size == 0:
            continue

        for i in range(0, edge.waypoints.shape[1], sample_step):
            x = float(edge.waypoints[0, i])
            y = float(edge.waypoints[1, i])
            overlay_shape.spawn_id_degrees(
                actorNumber=actor_id,
                location=[x, y, z],
                rotation=[0, 0, 0],
                scale=[0.02, 0.02, 0.002],
                configuration=1,  # green color
                waitForConfirmation=False,
            )
            actor_id += 1
            if actor_id > 9000:  # safety limit
                return


# ===========================
# 4. MAIN
# ===========================
def main():
    os.system("cls")
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    if not qlabs.open("localhost"):
        print("Unable to connect to QLabs")
        return

    print("Connected. Resetting Environment...")
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()

    # 1. Setup Complex Environment (Walls, Signs, etc.)
    car_actor = setup_environment(qlabs, INITIAL_POS, INITIAL_ROT)

    # 2. Setup Traffic Lights
    trafficLight1 = QLabsTrafficLight(qlabs)
    trafficLight2 = QLabsTrafficLight(qlabs)
    trafficLight3 = QLabsTrafficLight(qlabs)
    trafficLight4 = QLabsTrafficLight(qlabs)

    trafficLight1.spawn_id_degrees(
        actorNumber=1,
        location=[0.6, 1.55, 0.006],
        rotation=[0, 0, 0],
        scale=[0.1, 0.1, 0.1],
        configuration=0,
        waitForConfirmation=False,
    )
    trafficLight2.spawn_id_degrees(
        actorNumber=2,
        location=[-0.6, 1.28, 0.006],
        rotation=[0, 0, 90],
        scale=[0.1, 0.1, 0.1],
        configuration=0,
        waitForConfirmation=False,
    )
    trafficLight3.spawn_id_degrees(
        actorNumber=3,
        location=[-0.37, 0.3, 0.006],
        rotation=[0, 0, 180],
        scale=[0.1, 0.1, 0.1],
        configuration=0,
        waitForConfirmation=False,
    )
    trafficLight4.spawn_id_degrees(
        actorNumber=4,
        location=[0.75, 0.48, 0.006],
        rotation=[0, 0, -90],
        scale=[0.1, 0.1, 0.1],
        configuration=0,
        waitForConfirmation=False,
    )

    tl_thread = threading.Thread(
        target=traffic_light_logic,
        args=(trafficLight1, trafficLight2, trafficLight3, trafficLight4),
        daemon=True,
    )
    tl_thread.start()

    # 3. Setup Hardware & Pathing (Simpler main.py approach)
    qcar = QCar(readMode=1, frequency=CONTROLLER_RATE)
    gps = QCarGPS(initialPose=INITIAL_POS)
    ekf = QCarEKF(x_0=INITIAL_POS)

    roadmap = CustomRoadMap()
    # ==========================================
    # SEQUENCE VALIDATION CHECK
    # ==========================================
    # 1. Check for Closed Loop (Start must equal End)
    if NODE_SEQUENCE[0] != 10 or NODE_SEQUENCE[-1] != 10:
        print(
            f"\n[ERROR] Invalid Sequence: The path must be a loop that starts and ends at Node 10."
        )
        print(f"  Start Node: {NODE_SEQUENCE[0]}")
        print(f"  End Node:   {NODE_SEQUENCE[-1]}")
        print(
            "  Please ensure the first and last nodes in NODE_SEQUENCE are identical.\n"
        )
        qlabs.close()
        return

    # 2. Check Connectivity (Do edges exist?)
    # Create a set of all valid connections currently in the map
    valid_edges = set()
    for edge in roadmap.edges:
        # Map node objects back to their integer IDs
        from_id = roadmap.nodes.index(edge.fromNode)
        to_id = roadmap.nodes.index(edge.toNode)
        valid_edges.add((from_id, to_id))

    # Verify every step in the user's sequence
    for i in range(len(NODE_SEQUENCE) - 1):
        curr_node = NODE_SEQUENCE[i]
        next_node = NODE_SEQUENCE[i + 1]

        if (curr_node, next_node) not in valid_edges:
            print(f"\n[ERROR] Broken Path Detected!")
            print(
                f"  There is no defined edge from Node {curr_node} -> Node {next_node}."
            )
            print("  The path planner cannot generate a route for this section.")
            print(
                "  Check 'edgeConfigs' in custom_roadmap.py or fix your NODE_SEQUENCE.\n"
            )
            qlabs.close()
            return

    # If we pass these checks, proceed to generate path
    print("Sequence validated successfully.")
    waypointSequence = roadmap.generate_path(NODE_SEQUENCE)

    if waypointSequence is None:
        print(f"ERROR: Failed to generate path for sequence {NODE_SEQUENCE}")
        print(f"Roadmap has {len(roadmap.nodes)} nodes and {len(roadmap.edges)} edges")
        qlabs.close()
        return

    if DRAW_ALL_ROADS:
        print("Drawing road network overlay...")
        draw_all_roads(qlabs, roadmap, PATH_SAMPLE_STEP, PATH_Z)

    if DRAW_PATH_OVERLAY:
        draw_path_overlay(qlabs, waypointSequence, PATH_SAMPLE_STEP, PATH_Z)

    speed_ctrl = SpeedController()
    steer_ctrl = SteeringController(waypoints=waypointSequence, cyclic=True)
    # Example of creating vehSTate instance
    curState = VehState(car_actor)
    
    #Previous centering function call, commented out 
    #camera = QCarCameras(frameWidth=820, frameHeight=820, frameRate=30, enableLeft=True)
    
    #lane_thread = threading.Thread(target=lane_offset_thread, args=(camera,), daemon=True)
    #lane_thread.start()

    print(f"Environment Ready. Following Nodes: {NODE_SEQUENCE}")
    pathForState = -1
    # 4. Main Control Loop
    with qcar, gps:
        t0 = time.time()
        while not KILL_PROGRAM:
            
            t = time.time() - t0
            dt = 1.0 / CONTROLLER_RATE

            # Read Sensors
            qcar.read()
            if gps.readGPS():
                y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
                ekf.update([qcar.motorTach, 0], dt, y_gps, qcar.gyroscope[2])
            else:
                ekf.update([qcar.motorTach, 0], dt, None, qcar.gyroscope[2])

            x, y, th = ekf.x_hat[0, 0], ekf.x_hat[1, 0], ekf.x_hat[2, 0]
            v = qcar.motorTach

            # Calculate Steering (Front Axle)
            p_front = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2            

            if t < START_DELAY:
                qcar.write(0, 0)
                time.sleep(dt)
                continue

            xy = np.array([x, y])

            if curState.state == VehState.IDLE:
                goal_xy = PICKUP_XY
            elif curState.state == VehState.DRIVE_EMPTY:
                goal_xy = PICKUP_XY
            elif curState.state == VehState.PICKUP:
                goal_xy = DROPOFF_XY
            elif curState.state == VehState.DROPOFF:
                goal_xy = HUB_XY

            if pathForState != curState.state:
                newWaypointSequence = loop_path_waypoints(waypointSequence, xy, goal_xy)
                steer_ctrl.newWaypoint(newWaypointSequence)
                pathForState = curState.state

            # Stop gating at pickup/dropoff/hub
            dist = float(np.linalg.norm(goal_xy - xy))
            atWaypoint = dist < destThd
            
            if atWaypoint:
                qcar.write(0, 0)

                if abs(v) < vStopped:
                    stop_timer += dt
                else:
                    stop_timer = 0.0

                if stop_timer >= destHoldTime:
                    stop_timer = 0.0

                    if curState.state == VehState.IDLE:
                        curState.update(VehState.DRIVE_EMPTY)   
                        pathForState = None
                    elif curState.state == VehState.DRIVE_EMPTY:
                        curState.update(VehState.PICKUP)        
                        pathForState = None
                    elif curState.state == VehState.PICKUP:
                        curState.update(VehState.DROPOFF)      
                        pathForState = None
                    elif curState.state == VehState.DROPOFF:
                        curState.update(VehState.IDLE)          
                        pathForState = None

            else:
                thr = speed_ctrl.update(v, V_REF, dt)
                str_ang = steer_ctrl.update(p_front, th, v)
                qcar.write(thr, str_ang)

    # Cleanup
    print("Stopping...")
    QLabsRealTime().terminate_all_real_time_models()


if __name__ == "__main__":
    main()

import sys
import os
import time
import signal
import threading
import math
import heapq
import multiprocessing
import queue as pyqueue
from collections import deque

import cv2
import torch
import numpy as np
from ultralytics import YOLO

# --- QLabs & Hardware Imports ---
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
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap

from custom_roadmap import CustomRoadMap
from imageReadOrNodes3 import build_ce490v3_path_from_sequence

# Colors
MAGENTA = [1.0, 0.0, 1.0]
GREEN = [0.0, 1.0, 0.0]
BLUE = [0.0, 0.0, 1.0]
ORANGE = [1.0, 0.65, 0.0]

# Locations
PICKUP_STACK = [
    np.array([0.125, 4.395]),
    np.array([0.500, 3.200]),
    # np.array([x3, y3]),
]

DROPOFF_STACK = [
    np.array([-0.905, 0.800]),
    np.array([0.0, -0.2]),
    # np.array([x3, y3]),
]

HUB_XY = np.array([-1.055, -0.93])

# ===========================
# 1. CONFIGURATION
# ===========================
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

V_REF = 1.0
CONTROLLER_RATE = 100
START_DELAY = 2.0

# Perception / object detection thresholds
STOP_SIGN_MIN_WIDTH = 50
STOP_SIGN_WAIT_TIME_S = 1.0
RED_LIGHT_MIN_WIDTH = 18
RED_LIGHT_MIN_HEIGHT = 30
PEDESTRIAN_MIN_WIDTH_FOR_STOP = 20
STALE_OBJECT_TIMEOUT = 1.5
CENTER_MIN_X = 300
CENTER_MAX_X = 400

# Path overlay
DRAW_PATH_OVERLAY = False
DRAW_ALL_ROADS = False
PATH_SAMPLE_STEP = 10
PATH_Z = 0.02

# Initial Pose
INITIAL_POS = [-1.205, -0.83, 0.005]
INITIAL_ROT = [0, 0, -44.7]

KILL_PROGRAM = False
Kill_Thread = False

offsets = []
desiredCamTrack = 300
camCorrFct = 0
offsetFrameBuf = 5
camCorrHist = deque(maxlen=300)

destHoldTime = 1
vStopped = 0.1
destThd = 0.25
REANCHOR_LOOKAHEAD_WPS = 120


def sig_handler(*args):
    global KILL_PROGRAM
    KILL_PROGRAM = True


signal.signal(signal.SIGINT, sig_handler)


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
            case self.IDLE:
                if self.state == self.DROPOFF:
                    self.state = self.IDLE
                    self.qcar2.set_led_strip_uniform(MAGENTA)
                else:
                    print("Invalid state transition")

            case self.DRIVE_EMPTY:
                if (self.state == self.IDLE) or (self.state == self.DROPOFF):
                    self.state = self.DRIVE_EMPTY
                    self.qcar2.set_led_strip_uniform(GREEN)
                else:
                    print("Invalid state transition")

            case self.PICKUP:
                if self.state == self.DRIVE_EMPTY:
                    self.state = self.PICKUP
                    self.qcar2.set_led_strip_uniform(BLUE)
                else:
                    print("Invalid state transition")

            case self.DROPOFF:
                if self.state == self.PICKUP:
                    self.state = self.DROPOFF
                    self.qcar2.set_led_strip_uniform(ORANGE)
                else:
                    print("Invalid state transition")

            case _:
                print(f"[WARN] Invalid state request: {req}")

    def getState(self):
        return self.state


def peek_stack(stack):
    return stack[0] if len(stack) > 0 else None


def pop_stack(stack):
    return stack.pop(0) if len(stack) > 0 else None


# ===========================
# 2. CONTROLLERS
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

    def __init__(self, waypoints, k=0.6, cyclic=False):
        self.maxSteeringAngle = np.pi / 6
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.k = k
        self.cyclic = cyclic

    def update(self, p, th, speed):
        calc_speed = max(speed, 0.2)

        # Re-anchor only within a forward window to avoid cross-path index teleports.
        wp_points = self.wp[:2, :].T
        search_start = max(0, self.wpi)
        search_end = min(self.N - 1, self.wpi + REANCHOR_LOOKAHEAD_WPS)
        local_points = wp_points[search_start : search_end + 1]

        if local_points.shape[0] > 0:
            nearest_rel = int(np.argmin(np.linalg.norm(local_points - p[:2], axis=1)))
            nearest_idx = search_start + nearest_rel
            if nearest_idx > self.wpi:
                self.wpi = nearest_idx

        if self.cyclic:
            i1 = self.wpi % self.N
            i2 = (self.wpi + 1) % self.N
        else:
            i1 = min(self.wpi, self.N - 2)
            i2 = min(self.wpi + 1, self.N - 1)

        wp_1 = self.wp[:, i1]
        wp_2 = self.wp[:, i2]

        v_seg = wp_2 - wp_1
        v_mag = np.linalg.norm(v_seg)
        v_uv = v_seg / v_mag if v_mag > 0 else np.array([1.0, 0.0])

        tangent = np.arctan2(v_uv[1], v_uv[0])
        s = np.dot(p - wp_1, v_uv)

        # Calculate distance to the next waypoint
        dist_to_next_wp = np.linalg.norm(p - wp_2)

        # Dynamic switching threshold
        switch_threshold = min(v_mag * 0.35, 0.15)

        if s >= v_mag or dist_to_next_wp < switch_threshold:
            if self.cyclic or self.wpi < self.N - 2:
                self.wpi += 1
        else:
            heading = np.array([np.cos(th), np.sin(th)])
            to_next = wp_2[:2] - p[:2]
            if np.dot(heading, to_next) < -0.1 and (self.cyclic or self.wpi < self.N - 2):
                self.wpi += 1

        ep = wp_1 + v_uv * s
        ct = ep - p
        side_dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)
        ect = np.linalg.norm(ct) * np.sign(side_dir)

        psi = wrap_to_pi(tangent - th)
        steering = psi + np.arctan2(self.k * ect, calc_speed)

        return np.clip(
            wrap_to_pi(steering),
            -self.maxSteeringAngle,
            self.maxSteeringAngle,
        )

    def newWaypoint(self, waypoints):
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0


# ===========================
# 3. ENVIRONMENT SETUP
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

    hSystem = QLabsSystem(qlabs)
    hSystem.set_title_string("Simple Node Follower", waitForConfirmation=True)

    x_offset = 0.13
    y_offset = 1.67

    hFloor = QLabsQCarFlooring(qlabs)
    hFloor.spawn_degrees([x_offset, y_offset, 0.001], rotation=[0, 0, -90])

    hWall = QLabsWalls(qlabs)
    hWall.set_enable_dynamics(False)

    for y in range(5):
        hWall.spawn_degrees(
            location=[-2.4 + x_offset, (-y * 1.0) + 2.55 + y_offset, 0.001],
            rotation=[0, 0, 0],
        )

    for x in range(5):
        hWall.spawn_degrees(
            location=[-1.9 + x + x_offset, 3.05 + y_offset, 0.001],
            rotation=[0, 0, 90],
        )

    for y in range(6):
        hWall.spawn_degrees(
            location=[2.4 + x_offset, (-y * 1.0) + 2.55 + y_offset, 0.001],
            rotation=[0, 0, 0],
        )

    for x in range(4):
        hWall.spawn_degrees(
            location=[-0.9 + x + x_offset, -3.05 + y_offset, 0.001],
            rotation=[0, 0, 90],
        )

    hWall.spawn_degrees(
        location=[-2.03 + x_offset, -2.275 + y_offset, 0.001],
        rotation=[0, 0, 48],
    )
    hWall.spawn_degrees(
        location=[-1.575 + x_offset, -2.7 + y_offset, 0.001],
        rotation=[0, 0, 48],
    )

    car2 = QLabsQCar2(qlabs)
    car2.spawn_id(
        actorNumber=0,
        location=initialPosition,
        rotation=initialOrientation,
        scale=[0.1, 0.1, 0.1],
        configuration=0,
        waitForConfirmation=True,
    )

    camera1Loc = [0.15, 1.7, 5]
    camera1Rot = [0, 90, 0]
    camera1 = QLabsFreeCamera(qlabs)
    camera1.spawn_degrees(location=camera1Loc, rotation=camera1Rot)

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

    rtModel = os.path.normpath(
        os.path.join(os.environ["RTMODELS_DIR"], "QCar2/QCar2_Workspace_studio")
    )
    QLabsRealTime().start_real_time_model(rtModel)

    return car2


def dijkstra(roadmap, start_idx, goal_idx):
    pq = [(0.0, start_idx)]
    dist = {start_idx: 0.0}
    prev = {start_idx: None}
    visited = set()

    adj = {}
    for e in roadmap.edges:
        a = roadmap.nodes.index(e.fromNode)
        b = roadmap.nodes.index(e.toNode)
        adj.setdefault(a, []).append(b)

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        if u == goal_idx:
            break

        p1 = roadmap.nodes[u].pose[:2, 0]

        for v in adj.get(u, []):
            p2 = roadmap.nodes[v].pose[:2, 0]
            w = float(np.linalg.norm(p1 - p2))
            nd = d + w

            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if goal_idx not in prev and goal_idx != start_idx:
        return None

    path = []
    cur = goal_idx
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()

    return path


def lane_offset_thread(camera, sleep_s=0.01):
    global KILL_PROGRAM, offsets, camCorrFct

    YELLOW_LO = (10, 60, 60)
    YELLOW_HI = (50, 255, 255)
    WHITE_LO = (0, 0, 200)
    WHITE_HI = (180, 45, 255)

    kernel = np.ones((5, 5), np.uint8)

    while not KILL_PROGRAM:
        camera.readAll()
        rawFrame = camera.csiLeft.imageData

        if rawFrame.ndim == 2:
            rawFrame = cv2.cvtColor(rawFrame, cv2.COLOR_GRAY2BGR)
        if rawFrame.shape[-1] == 4:
            rawFrame = rawFrame[:, :, :3]

        roi = rawFrame[:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
        white_mask = cv2.inRange(hsv, WHITE_LO, WHITE_HI)
        mask = cv2.bitwise_or(yellow_mask, white_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        H, W = mask.shape
        y0 = int(H * 0.50)
        roi = mask[y0:, :]
        hR, wR = roi.shape

        src = np.float32(
            [
                [wR * 0.2, hR * 0.55],
                [wR * 0.8, hR * 0.55],
                [wR * 1.0, hR * 0.98],
                [wR * 0.0, hR * 0.98],
            ]
        )

        dst = np.float32(
            [
                [wR - 1, hR - 1],
                [0, hR - 1],
                [0, 0],
                [wR - 1, 0],
            ]
        )

        M = cv2.getPerspectiveTransform(src, dst)
        procImg = cv2.warpPerspective(roi, M, (wR, hR), flags=cv2.INTER_NEAREST)
        procImg = cv2.rotate(procImg, cv2.ROTATE_180)
        procImg = cv2.morphologyEx(procImg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        mid = wR // 2
        third = wR // 4
        leftROI = procImg[:, :third]

        closestX = []

        for row in (leftROI > 0):
            idx = np.flatnonzero(row)
            if idx.size == 0:
                continue

            gaps = np.where(np.diff(idx) > 10)[0]
            if gaps.size:
                closestX.append(int(idx[gaps[0]]))
            else:
                closestX.append(int(idx[-1]))

        if len(closestX) < 30:
            time.sleep(sleep_s)
            continue

        laneX = float(np.median(closestX))
        offset = mid - laneX

        if offset <= 0 or offset >= mid:
            time.sleep(sleep_s)
            continue

        offsets.append(offset)
        if len(offsets) > offsetFrameBuf:
            offsets.pop(0)

        avgOffset = sum(offsets) / len(offsets)
        errX = avgOffset - desiredCamTrack

        Kcam = 1
        camCorrFct = np.clip(Kcam * (errX / mid), -0.5, 0.5)
        time.sleep(sleep_s)


def draw_path_overlay(qlabs, waypoints, sample_step=10, z=0.02):
    if waypoints is None or waypoints.size == 0:
        return

    overlay_shape = QLabsBasicShape(qlabs)
    actor_id = 200

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
    overlay_shape = QLabsBasicShape(qlabs)
    actor_id = 1000

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
                configuration=1,
                waitForConfirmation=False,
            )
            actor_id += 1
            if actor_id > 9000:
                return


def camera_thread_worker(camera):
    print("Camera Feed Thread Started")
    while True:
        camera.readAll()
        croppedRGB = camera.csiLeft.imageData[350:820, :]
        hsvBuf = cv2.cvtColor(croppedRGB, cv2.COLOR_BGR2HSV)

        yellow_bin = ImageProcessing.binary_thresholding(
            hsvBuf, np.array([0, 0, 200]), np.array([45, 255, 255])
        )
        white_bin = ImageProcessing.binary_thresholding(
            hsvBuf, np.array([0, 0, 200]), np.array([180, 50, 255])
        )
        binaryImage = cv2.bitwise_or(yellow_bin, white_bin)
        cv2.imshow("Combined Lane Detection", binaryImage)
        cv2.waitKey(1)


def perception_thread(perception_queue, actor_id):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(script_dir, "model", "best.pt")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(MODEL_PATH).to(device)

        camera = QCarCameras(
            frameWidth=820,
            frameHeight=410,
            frameRate=30,
            enableLeft=True,
        )

        while not Kill_Thread:
            ok = camera.readAll()
            if ok:
                image = camera.csiLeft.imageData[:, :]
                results = model(image, device=device, conf=0.4, verbose=False)[0]

                detections = []
                for box in results.boxes:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    x_center, y_center, width, height = box.xywh[0]
                    x_top_left = x_center.item() - (width.item() / 2)
                    y_top_left = y_center.item() - (height.item() / 2)

                    detection_data = {
                        "class": class_name,
                        "width": width.item(),
                        "height": height.item(),
                        "x": x_top_left,
                        "y": y_top_left,
                    }
                    detections.append(detection_data)

                output_data = {"detections": detections}

                if not perception_queue.full():
                    perception_queue.put(output_data)

                annotated_image = results.plot()
                window_name = f"YOLO Detection - Car {actor_id}"
                cv2.imshow(window_name, annotated_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.01)

    except Exception as e:
        print(f"[Perception-{actor_id}] An error occurred: {e}", file=sys.stderr)
    finally:
        print(f"[Perception-{actor_id}] Stopping thread.")
        cv2.destroyAllWindows()


# ===========================
# 4. MAIN
# ===========================
def main():
    global Kill_Thread

    os.system("cls")
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")

    if not qlabs.open("localhost"):
        print("Unable to connect to QLabs")
        return

    print("Connected. Resetting Environment...")
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()

    car_actor = setup_environment(qlabs, INITIAL_POS, INITIAL_ROT)

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

    qcar = QCar(readMode=1, frequency=CONTROLLER_RATE)
    gps = QCarGPS(initialPose=INITIAL_POS)
    ekf = QCarEKF(x_0=INITIAL_POS)

    # Keep roadmap for later Dijkstra replanning
    roadmap = CustomRoadMap()

    # Initial loop/path comes from imageReadOrNodes3.py
    if len(NODE_SEQUENCE) < 2:
        print("[ERROR] NODE_SEQUENCE must contain at least 2 nodes.")
        qlabs.close()
        return

    try:
        map_name, node_coords, waypointSequence, path_meta = build_ce490v3_path_from_sequence(
            node_sequence=NODE_SEQUENCE,
            edge_margin_m=0.09,
            waypoint_step_m=0.02,
            debug_masks=True,
        )
    except Exception as e:
        print(f"[ERROR] Failed to build path from imageReadOrNodes3.py: {e}")
        qlabs.close()
        return

    if waypointSequence is None or waypointSequence.size == 0:
        print(f"[ERROR] Image-based path builder returned no waypoints for {NODE_SEQUENCE}")
        qlabs.close()
        return

    print(f"Path built successfully from map '{map_name}'.")
    print(f"Generated {waypointSequence.shape[1]} ordered waypoints.")

    if DRAW_ALL_ROADS:
        print("Drawing road network overlay...")
        draw_all_roads(qlabs, roadmap, PATH_SAMPLE_STEP, PATH_Z)

    if DRAW_PATH_OVERLAY:
        draw_path_overlay(qlabs, waypointSequence, PATH_SAMPLE_STEP, PATH_Z)

    speed_ctrl = SpeedController()
    steer_ctrl = SteeringController(waypoints=waypointSequence, cyclic=True)
    curState = VehState(car_actor)

    perception_queue = multiprocessing.Queue(maxsize=5)
    perception_process = multiprocessing.Process(
        target=perception_thread,
        args=(perception_queue, car_actor.actorNumber),
        daemon=True,
    )
    perception_process.start()

    print(f"Environment Ready. Following Nodes: {NODE_SEQUENCE}")

    with qcar, gps:
        t0 = time.time()
        pathForState = None
        stop_timer = 0.0

        current_detections = []
        last_perception_update = time.time()
        stop_sign_start_time = 0.0
        is_stopped_for_sign = False
        atWaypoint = False
        active_stop_xy = None
        lastStr = 0.0

        while not KILL_PROGRAM:
            t = time.time() - t0
            dt = 1.0 / CONTROLLER_RATE

            try:
                while True:
                    data = perception_queue.get_nowait()
                    current_detections = data.get("detections", [])
                    last_perception_update = time.time()
            except pyqueue.Empty:
                pass

            should_stop = False
            now = time.time()
            if now - last_perception_update > STALE_OBJECT_TIMEOUT:
                current_detections = []

            stop_sign_visible = False
            red_light_visible = False
            green_light_visible = False
            pedestrian_visible = False
            yellow_light_visible = False

            for obj in current_detections:
                cls = obj.get("class")
                w = float(obj.get("width", 0.0))
                h = float(obj.get("height", 0.0))
                x_px = float(obj.get("x", 0.0))

                if cls == "stop_sign" and w > STOP_SIGN_MIN_WIDTH:
                    stop_sign_visible = True

                if (
                    cls == "red_light"
                    and w > RED_LIGHT_MIN_WIDTH
                    and h > RED_LIGHT_MIN_HEIGHT
                    and (CENTER_MIN_X < x_px < CENTER_MAX_X)
                ):
                    red_light_visible = True

                if (
                    cls == "green_light"
                    and w > RED_LIGHT_MIN_WIDTH
                    and h > RED_LIGHT_MIN_HEIGHT
                    and (CENTER_MIN_X < x_px < CENTER_MAX_X)
                ):
                    green_light_visible = True

                if (
                    cls == "yellow_light"
                    and w > RED_LIGHT_MIN_WIDTH
                    and h > RED_LIGHT_MIN_HEIGHT
                    and (CENTER_MIN_X < x_px < CENTER_MAX_X)
                ):
                    yellow_light_visible = True

                if cls == "pedestrian" and w > PEDESTRIAN_MIN_WIDTH_FOR_STOP and (100 < x_px < 450):
                    pedestrian_visible = True

            if stop_sign_visible and not is_stopped_for_sign:
                should_stop = True
                if stop_sign_start_time == 0.0:
                    stop_sign_start_time = now
                elif now - stop_sign_start_time > STOP_SIGN_WAIT_TIME_S:
                    is_stopped_for_sign = True
                    stop_sign_start_time = 0.0
            elif not stop_sign_visible:
                stop_sign_start_time = 0.0
                is_stopped_for_sign = False

            if red_light_visible:
                should_stop = True
            if yellow_light_visible:
                should_stop = True
            if green_light_visible:
                should_stop = False
            if pedestrian_visible:
                should_stop = True

            qcar.read()

            if gps.readGPS():
                y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
                ekf.update([qcar.motorTach, 0], dt, y_gps, qcar.gyroscope[2])
            else:
                ekf.update([qcar.motorTach, 0], dt, None, qcar.gyroscope[2])

            x = ekf.x_hat[0, 0]
            y = ekf.x_hat[1, 0]
            th = ekf.x_hat[2, 0]
            v = qcar.motorTach

            p_front = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2

            if t < START_DELAY:
                qcar.write(0, 0)
                time.sleep(dt)
                continue

            xy = np.array([x, y])

            if curState.state == VehState.IDLE:
                goal_xy = peek_stack(PICKUP_STACK) if len(PICKUP_STACK) > 0 else HUB_XY
            elif curState.state == VehState.DRIVE_EMPTY:
                goal_xy = peek_stack(PICKUP_STACK) if len(PICKUP_STACK) > 0 else HUB_XY
            elif curState.state == VehState.PICKUP:
                goal_xy = peek_stack(DROPOFF_STACK) if len(DROPOFF_STACK) > 0 else HUB_XY
            elif curState.state == VehState.DROPOFF:
                goal_xy = HUB_XY
            else:
                goal_xy = HUB_XY

            if pathForState != curState.state:
                start_n = min(
                    range(len(roadmap.nodes)),
                    key=lambda i: np.linalg.norm(roadmap.nodes[i].pose[:2, 0] - xy),
                )

                goal_n = min(
                    range(len(roadmap.nodes)),
                    key=lambda i: np.linalg.norm(roadmap.nodes[i].pose[:2, 0] - goal_xy),
                )

                node_path = dijkstra(roadmap, start_n, goal_n)

                if node_path is not None:
                    newWaypointSequence = roadmap.generate_path(node_path)

                    goal_col = np.array(goal_xy, dtype=float).reshape(2, 1)
                    newWaypointSequence = np.hstack([newWaypointSequence, goal_col])

                    steer_ctrl.cyclic = False
                    steer_ctrl.newWaypoint(newWaypointSequence)

                    active_stop_xy = newWaypointSequence[:2, -1].copy()

                    print(
                        f"[INFO] Replanned state={curState.state} "
                        f"start_n={start_n} goal_n={goal_n} stop={active_stop_xy}"
                    )
                else:
                    print(f"[WARN] No path found to goal {goal_xy}")
                    active_stop_xy = goal_xy.copy()

                pathForState = curState.state

            dist = float(np.linalg.norm(goal_xy - xy))
            if dist < destThd:
                atWaypoint = True

            if atWaypoint:
                lastStr = lastStr * 0.95
                qcar.write(0, lastStr)

                if abs(v) < vStopped:
                    stop_timer += dt
                else:
                    stop_timer = 0.0

                if stop_timer >= destHoldTime:
                    stop_timer = 0.0

                    if curState.state == VehState.IDLE:
                        if len(PICKUP_STACK) > 0:
                            curState.update(VehState.DRIVE_EMPTY)
                            pathForState = None

                    elif curState.state == VehState.DRIVE_EMPTY:
                        if len(PICKUP_STACK) > 0:
                            serviced_pickup = pop_stack(PICKUP_STACK)
                            print(f"[INFO] Picked up at {serviced_pickup}")
                        curState.update(VehState.PICKUP)
                        pathForState = None

                    elif curState.state == VehState.PICKUP:
                        if len(DROPOFF_STACK) > 0:
                            serviced_dropoff = pop_stack(DROPOFF_STACK)
                            print(f"[INFO] Dropped off at {serviced_dropoff}")
                        curState.update(VehState.DROPOFF)
                        pathForState = None

                    elif curState.state == VehState.DROPOFF:
                        curState.update(VehState.IDLE)
                        pathForState = None

                    atWaypoint = False

            else:
                if should_stop:
                    qcar.write(0, lastStr)
                else:
                    thr = speed_ctrl.update(v, V_REF, dt)
                    str_ang = steer_ctrl.update(p_front, th, v)
                    qcar.write(thr, str_ang)
                    lastStr = str_ang

            time.sleep(dt)

    print("Stopping...")

    try:
        Kill_Thread = True
        if "perception_process" in locals() and perception_process.is_alive():
            perception_process.terminate()
            perception_process.join(timeout=1.0)
    except Exception:
        pass

    QLabsRealTime().terminate_all_real_time_models()


if __name__ == "__main__":
    main()
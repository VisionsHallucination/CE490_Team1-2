import sys
import cv2
import os
import time
import signal
import threading
import numpy as np
import math
import heapq
import multiprocessing
import queue as pyqueue
from pal.products.qcar import QCarCameras

# --- QLabs & Hardware Imports ---
import cv2
from hal.utilities.image_processing import ImageProcessing
from pal.products.qcar import QCar, QCarGPS, QCarCameras

# --- Hardware & Math Imports ---
from pal.products.qcar import QCar, QCarGPS
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
from custom_roadmap import CustomRoadMap
from collections import deque

# Colors
MAGENTA = [1.0, 0.0, 1.0]
GREEN   = [0.0, 1.0, 0.0]
BLUE    = [0.0, 0.0, 1.0]
ORANGE  = [1.0, 0.65, 0.0]

#Locations
PICKUP_STACK = [
    np.array([0.125, 4.395]),
    np.array([0.500, 3.200]),
    # np.array([x3, y3]),
]

DROPOFF_STACK = [
    np.array([-0.905, 0.800]),
    np.array([0, -0.2]),
    # np.array([x3, y3]),
]

HUB_XY     = np.array([-1.055, -0.93]) 
TAXI_HUB_POS = [-1.319, -0.485, 0.006]
calibrationPose = [0, 2, -np.pi/2]   # or [0,0,-np.pi/2] depending on square
calComm = True


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

V_REF = .5  # Locked cruise speed
CONTROLLER_RATE = 100  # 100Hz loop
START_DELAY = 2.0  # Time for EKF to stabilize before moving

# Perception / object detection thresholds (from Setup_Real_Scenario.py)
STOP_SIGN_MIN_WIDTH = 50
STOP_SIGN_WAIT_TIME_S = 1.0
RED_LIGHT_MIN_WIDTH = 18
RED_LIGHT_MIN_HEIGHT = 30
PEDESTRIAN_MIN_WIDTH_FOR_STOP = 20
STALE_OBJECT_TIMEOUT = 1.5
CENTER_MIN_X = 300
CENTER_MAX_X = 400

# Path overlay (debug visualization)
DRAW_PATH_OVERLAY = False
DRAW_ALL_ROADS = False  # Draw entire road network
PATH_SAMPLE_STEP = 10  # plot every Nth waypoint
PATH_Z = 0.02

# Initial Pose (Matches Setup_Real_Scenario default)
INITIAL_POS = [0, 0, 0]
#INITIAL_ROT = [0, 0, -44.7]

KILL_PROGRAM = False

offsets = []
desiredCamTrack = 300
camCorrFct = 0
offsetFrameBuf = 5
camCorrHist = deque(maxlen=300)

destHoldTime = 1 # How long to pause at destination
vStopped = .1 # What speed counts as vehicle stopped
destThd = .40 # How far can the car be while concidered at the stop
REANCHOR_LOOKAHEAD_WPS = 40
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

    def __init__(self,):
        self.state = self.IDLE 
        #self.qcar2.set_led_strip_uniform(MAGENTA)

    def update(self, req):  
        if (req == self.IDLE):  # Idle
            if(self.state == self.DROPOFF):
                self.state = self.IDLE
                #self.qcar2.set_led_strip_uniform(MAGENTA)
            else:
                print("Invalid state transition")
        elif(req == self.DRIVE_EMPTY): # Drive with no passenger
            if((self.state == self.IDLE) or (self.state == self.DROPOFF)):
                self.state = self.DRIVE_EMPTY
                #self.qcar2.set_led_strip_uniform(GREEN)
            else:
                print("Invalid state transition")
        elif(req == self.PICKUP): # Picked up passenger
            if(self.state == self.DRIVE_EMPTY):
                self.state = self.PICKUP
                #self.qcar2.set_led_strip_uniform(BLUE)
            else:
                print("Invalid state transition")
        elif(req == self.DROPOFF): # Drop off passenger
            if(self.state == self.PICKUP):
                self.state = self.DROPOFF
                #self.qcar2.set_led_strip_uniform(ORANGE)
            else:
                print("Invalid state transition")
        else:
            print(f"[WARN] Invalid state request: {req}")

    def getState(self):
        return self.state

def peek_stack(stack):
    return stack[0] if len(stack) > 0 else None

def pop_stack(stack):
    return stack.pop(0) if len(stack) > 0 else None

# ===========================
# 2. CONTROLLERS (From main.py)
# ===========================


class SpeedController:
    """Locked speed controller with anti-surge logic."""

    def __init__(self, kp=0.1, ki=1):
        self.maxThrottle = .3
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

    def __init__(self, waypoints, k=1, cyclic=False):
        self.maxSteeringAngle = np.pi / 6
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.k = k
        self.cyclic = cyclic

    # input params: Current position (p), yaw angle(th), speed                                                                                                                     
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
		# --- FIX START ---
        # Calculate distance to the next waypoint (wp_2)
        dist_to_next_wp = np.linalg.norm(p - wp_2)

        # Dynamic threshold: Allow cutting corners on long segments (up to 1.0m),
        # but require precision on short Bezier segments (limited by v_mag).
        switch_threshold = min(v_mag * 0.35, 0.15)

        # Switch if we pass the line OR if we are close enough to the endpoint
        if s >= v_mag or dist_to_next_wp < switch_threshold:
            if self.cyclic or self.wpi < self.N - 2:
                self.wpi += 1
        else:
            # If the next waypoint is behind us, jump forward to reduce loop-de-loops
            heading = np.array([np.cos(th), np.sin(th)])
            to_next = wp_2[:2] - p[:2]
            if np.dot(heading, to_next) < -0.1 and (
                self.cyclic or self.wpi < self.N - 2
            ):
                self.wpi += 1
        # --- FIX END ---

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


def dijkstra(roadmap, start_idx, goal_idx):
    pq = [(0.0, start_idx)]
    dist = {start_idx: 0.0}
    prev = {start_idx: None}
    visited = set()

    # Build adjacency directly from roadmap edges
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

# ===========================
# 3.4 CAMERA / LANE DEBUG 
# ===========================

def camera_thread_worker(camera):
    """Independent thread to keep the camera feed alive constantly.

    This is primarily for visualizing lane thresholding in real-time.
    """
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

# ===========================
# 3.5 PERCEPTION
# ===========================
Kill_Thread = False

# ===========================
# 4. MAIN
# ===========================
def main():
    os.system("clear")
    print("Connecting to QLabs...")

    roadmap = CustomRoadMap()

    start_node = NODE_SEQUENCE[0]
    initialPose = roadmap.nodes[start_node].pose[:2, 0]
    initialPose = np.array([
        initialPose[0],
        initialPose[1],
        calibrationPose[2]   # <-- match heading
    ])

    # 3. Setup Hardware & Pathing (Simpler main.py approach)
    qcar = QCar(readMode=1, frequency=CONTROLLER_RATE)
    ekf = QCarEKF(x_0=initialPose)
    gps = QCarGPS(initialPose=calibrationPose, calibrate=calComm)

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
            return

    # If we pass these checks, proceed to generate path
    print("Sequence validated successfully.")
    waypointSequence = roadmap.generate_path(NODE_SEQUENCE)

    if waypointSequence is None:
        print(f"ERROR: Failed to generate path for sequence {NODE_SEQUENCE}")
        print(f"Roadmap has {len(roadmap.nodes)} nodes and {len(roadmap.edges)} edges")
        return

    speed_ctrl = SpeedController()
    steer_ctrl = SteeringController(waypoints=waypointSequence, cyclic=False)
    # Example of creating vehSTate instance
    curState = VehState()

    print(f"Environment Ready. Following Nodes: {NODE_SEQUENCE}")
    # 4. Main Control Loop
    with qcar, gps:
        t0 = time.time()
        pathForState = None
        stop_timer = 0.0
        atWaypoint = False
        active_stop_xy = None
        lastStr = 0
        launch_xy = None
        delay_reanchored = False
        min_dist_seen = float('inf')  # track closest approach to goal
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
                #print(f"DELAY t={t:.2f}")
                continue

            xy = np.array([x, y])

            # One-time re-anchor after START_DELAY so wpi isn't stuck at 0
            if not delay_reanchored:
                wp_xy = steer_ctrl.wp[:2, :].T
                closest_i = int(np.argmin(np.linalg.norm(wp_xy - xy, axis=1)))
                steer_ctrl.wpi = min(closest_i, steer_ctrl.N - 2)
                delay_reanchored = True
                print(f"[INFO] Post-delay re-anchor wpi={steer_ctrl.wpi} pos=({x:.3f},{y:.3f}) th={th:.3f}")

            if curState.state == VehState.IDLE:
                goal_xy = peek_stack(PICKUP_STACK) if len(PICKUP_STACK) > 0 else HUB_XY
            elif curState.state == VehState.DRIVE_EMPTY:
                goal_xy = peek_stack(PICKUP_STACK) if len(PICKUP_STACK) > 0 else HUB_XY
            elif curState.state == VehState.PICKUP:
                goal_xy = peek_stack(DROPOFF_STACK) if len(DROPOFF_STACK) > 0 else HUB_XY
            elif curState.state == VehState.DROPOFF:
                goal_xy = HUB_XY

            if launch_xy is None:
                launch_xy = xy.copy() 
            moved_enough = (np.linalg.norm(xy - launch_xy) > 0.10) or (abs(v) > 0.02)

            if pathForState is None:
                allow_replan = True
            else:
                allow_replan = moved_enough

            if allow_replan and pathForState != curState.state:
                start_n = min(
                    range(len(roadmap.nodes)),
                    key=lambda i: np.linalg.norm(roadmap.nodes[i].pose[:2, 0] - xy),
                )
                goal_n = min(
                    range(len(roadmap.nodes)),
                    key=lambda i: np.linalg.norm(roadmap.nodes[i].pose[:2, 0] - goal_xy),
                )

                node_path = dijkstra(roadmap, start_n, goal_n)

                if node_path is not None :
                    newWaypointSequence = roadmap.generate_path(node_path)
                    steer_ctrl.cyclic = False
                    steer_ctrl.newWaypoint(newWaypointSequence)
                    wp_xy = newWaypointSequence[:2, :].T
                    closest_i = int(np.argmin(np.linalg.norm(wp_xy - xy, axis=1)))
                    steer_ctrl.wpi = min(closest_i, steer_ctrl.N - 2)
                        
                    active_stop_xy = newWaypointSequence[:2, -1].copy()
                       
                    print(f"[INFO] Replanned state={curState.state} start_n={start_n} goal_n={goal_n} path={node_path} N_wp={steer_ctrl.N} stop=({active_stop_xy[0]:.3f},{active_stop_xy[1]:.3f}) goal=({goal_xy[0]:.3f},{goal_xy[1]:.3f})")
                        
                else:
                    print(f"[WARN] No path found to goal {goal_xy}")
                    active_stop_xy = goal_xy.copy()
                    
                pathForState = curState.state
                min_dist_seen = float('inf')  # reset for new goal

            # Stop gating at pickup/dropoff/hub
            stop_target = active_stop_xy if active_stop_xy is not None else goal_xy
            dist = float(np.linalg.norm(stop_target - xy))

            # Track closest approach - trigger stop if we were close and are now moving away
            if dist < min_dist_seen:
                min_dist_seen = dist

            # Trigger atWaypoint if:
            #   (a) we're within destThd of the goal, OR
            #   (b) we got within destThd at some point and dist is now increasing (drove past it)
            near_path_end = (not steer_ctrl.cyclic) and (steer_ctrl.wpi >= steer_ctrl.N - 3)

            if not atWaypoint:
                if dist < destThd:
                    atWaypoint = True
                    print(f"[INFO] atWaypoint triggered (dist): dist={dist:.3f} stop=({stop_target[0]:.2f},{stop_target[1]:.2f})")
                elif near_path_end and min_dist_seen < 1.5:
                    # Reached end of road path and were reasonably close to stop point
                    atWaypoint = True
                    print(f"[INFO] atWaypoint triggered (end of path): wpi={steer_ctrl.wpi}/{steer_ctrl.N} min_dist={min_dist_seen:.3f}")
                elif min_dist_seen < destThd and dist > min_dist_seen + 0.10:
                    # Were close enough but drove past
                    atWaypoint = True
                    print(f"[INFO] atWaypoint triggered (drove past): min_dist={min_dist_seen:.3f} dist={dist:.3f}")
            
            
            
            if atWaypoint:
                lastStr = lastStr * .95
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
                    launch_xy = xy.copy()
                    min_dist_seen = float('inf')
                    print(f"[INFO] State transition complete, new state={curState.state}")


            else:
                thr = speed_ctrl.update(v, V_REF, dt)
                str_ang = steer_ctrl.update(p_front, th, v)

                print(f"DRIVE t={t:.2f} v={v:.3f} thr={thr:.3f} str={str_ang:.3f} dist={dist:.3f} min={min_dist_seen:.3f} pos=({x:.2f},{y:.2f}) stop=({stop_target[0]:.2f},{stop_target[1]:.2f}) wpi={steer_ctrl.wpi}/{steer_ctrl.N}")
                qcar.write(thr, str_ang)
                lastStr = str_ang

    # Cleanup
    print("Stopping...")
    try:
        global Kill_Thread
        Kill_Thread = True
        
    except Exception:
        pass



if __name__ == "__main__":
    main()

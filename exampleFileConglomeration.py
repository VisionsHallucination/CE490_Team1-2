'''
Docstring for exampleFileConglomeration
'''




'''
application:
'''


from pal.utilities.probe import Observer

observer = Observer()

imageWidth = 640
imageHeight = 480
observer.add_display(imageSize = [imageHeight + 40, 4*imageWidth + 120, 3],
                    scalingFactor=2,
                    name='360 CSI')

observer.launch()










## imaging_360.py
# This example demonstrates how to read all 4 csi cameras and display in a single openCV window. If you encounter any errors, 
# use the hardware_test_csi_camera_single.py script to find out which camera is giving you trouble. 

from pal.utilities.vision import Camera2D
from pal.utilities.probe import Probe
import time
import struct
import numpy as np 

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Timing Parameters and methods 
startTime = time.time()
def elapsed_time():
    return time.time() - startTime

sampleRate     = 30.0
sampleTime     = 1/sampleRate
simulationTime = 60.0
print('Sample Time: ', sampleTime)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
# Additional parameters
ipHost, ipDriver = '192.168.3.10', 'localhost'
counter = 0
imageWidth = 640
imageHeight = 480
imageBuffer360 = np.zeros((imageHeight + 40, 4*imageWidth + 120, 3), dtype=np.uint8) # 20 px padding between pieces  
        
# Stitch images together with black padding
horizontalBlank     = np.zeros((20, 4*imageWidth+120, 3), dtype=np.uint8)
verticalBlank       = np.zeros((imageHeight, 20, 3), dtype=np.uint8)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Initialize the CSI cameras and probe
myCam1 = Camera2D(cameraId="0", frameWidth=imageWidth, frameHeight=imageHeight, frameRate=sampleRate)
myCam2 = Camera2D(cameraId="1", frameWidth=imageWidth, frameHeight=imageHeight, frameRate=sampleRate)
myCam3 = Camera2D(cameraId="3", frameWidth=imageWidth, frameHeight=imageHeight, frameRate=sampleRate)
myCam4 = Camera2D(cameraId="2", frameWidth=imageWidth, frameHeight=imageHeight, frameRate=sampleRate)
probe = Probe(ip = ipHost)
probe.add_display(imageSize = [imageHeight + 40, 4*imageWidth + 120, 3], scaling = True,
                            scalingFactor= 2, name="360 CSI")

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Main Loop
try:
    while elapsed_time() < simulationTime:
        start = time.time()
        if not probe.connected:
            probe.check_connection()
        if probe.connected:
        # Start timing this iteration

            # Capture RGB Image from CSI
            flag1=myCam1.read()
            flag2=myCam2.read()
            flag3=myCam3.read()
            flag4=myCam4.read()

            imageBuffer360 = np.concatenate(
                                            (horizontalBlank, 
                                                np.concatenate((    verticalBlank, 
                                                                    myCam2.imageData[:,320:640], 
                                                                    verticalBlank, 
                                                                    myCam3.imageData, 
                                                                    verticalBlank, 
                                                                    myCam4.imageData, 
                                                                    verticalBlank, 
                                                                    myCam1.imageData, 
                                                                    verticalBlank, 
                                                                    myCam2.imageData[:,0:320], 
                                                                    verticalBlank), 
                                                                    axis = 1), 
                                                horizontalBlank
                                                ), 
                                                axis=0
                                            )

            if all([flag1,flag2,flag3,flag4]): counter += 1
            if counter % 4 == 0:
                sending = probe.send(name="360 CSI",
                                    imageData=imageBuffer360)

        # End timing this iteration
        end = time.time()

        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )
        
        # Pause/sleep for sleepTime in milliseconds
        if sleepTime <= 0:
            sleepTime = 0 
        time.sleep(sleepTime)

except KeyboardInterrupt:
    print("User interrupted!")

finally:
    # Terminate all webcam objects    
    probe.terminate()
    myCam1.terminate()
    myCam2.terminate()
    myCam3.terminate()
    myCam4.terminate()
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
















from pal.utilities.probe import Observer

observer = Observer()

imageWidth  = 1640
imageHeight = 820
observer.add_display(imageSize = [imageHeight, imageWidth, 3],
						scalingFactor= 4, name="Detection Overlay")
observer.launch()





















## task_lane_following.py
# This example combines both the left csi and motor commands to
# allow the QCar to follow a yellow lane. Use the joystick to manually drive the QCar
# to a starting position and enable the line follower by holding the X button on the LogitechF710
# To troubleshoot your camera use the hardware_test_csi_camera_single.py found in the hardware tests

# from pal.utilities.vision import Camera2D
from pal.products.qcar import QCar, QCarCameras
from pal.utilities.math import Filter
from pal.utilities.gamepad import LogitechF710
from pal.utilities.probe import Probe
from hal.utilities.image_processing import ImageProcessing

import time
import numpy as np
import cv2
import math

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## Timing Parameters and methods
sampleRate = 60
sampleTime = 1/sampleRate
print('Sample Time: ', sampleTime)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# Additional parameters
ipHost, ipDriver = '192.168.3.10', 'localhost'
counter 	= 0
imageWidth  = 1640
imageHeight = 820
# cameraID 	= '2'

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#Setting Filter
steeringFilter = Filter().low_pass_first_order_variable(25, 0.033)
next(steeringFilter)
dt = 0.033

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## Initialize the CSI cameras
# myCam = Camera2D(cameraId=cameraID, frameWidth=imageWidth, frameHeight=imageHeight, frameRate=sampleRate)
myCam = QCarCameras(frameWidth=imageWidth, frameHeight=imageHeight, frameRate=sampleRate, enableFront=True)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## QCar, Gamepad, and probe Initialization
myCar = QCar(readMode=1, frequency=60)
gpad  = LogitechF710()
probe = Probe(ip = ipHost)
probe.add_display(imageSize = [imageHeight, imageWidth, 3], scaling = True,
					scalingFactor= 4, name="Detection Overlay")

def control_from_gamepad(LB, RT, leftLateral, A):
	'''	User control function for use with the LogitechF710
	LB on gamepad is used to enable motor commands based on the RT input.
	Button A on gamepad is used to reverse the motor direction.
	'''
	if LB == 1:
			if A == 1 :
				throttle_axis = -0.3 * RT #going backward
				steering_axis = leftLateral * 0.5
			else:
				throttle_axis = 0.3 * RT #going forward
				steering_axis = leftLateral * 0.5
	else:
		throttle_axis = 0
		steering_axis = 0

	command = np.array([throttle_axis, steering_axis])
	return command


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## Main Loop
try:
	while True:
		start = time.time()
		# Capture RGB Image from CSI
		flags=myCam.readAll()
		if any(flags): counter +=1

		# Crop out a piece of the RGB to improve performance
		croppedRGB = myCam.csiFront.imageData[524:674, 0:820]

		# Convert to HSV and then threshold it for yellow
		hsvBuf = cv2.cvtColor(croppedRGB, cv2.COLOR_BGR2HSV)

		binaryImage = ImageProcessing.binary_thresholding(frame= hsvBuf,
													lowerBounds=np.array([10, 50, 100]),
													upperBounds=np.array([45, 255, 255]))
		
		# Overlay detected yellow lane over raw RGB image
		binaryImage=binaryImage/255
		processed = myCam.csiFront.imageData
		processed[524:674, 0:820,2]=processed[524:674, 0:820,2]+(255-processed[524:674, 0:820,2])*binaryImage
		processed[524:674, 0:820,1]=processed[524:674, 0:820,1]*(1-binaryImage)
		processed[524:674, 0:820,0]=processed[524:674, 0:820,0]*(1-binaryImage)

		# Send the processed image to the observer on the loacl PC to display
		if not probe.connected:
			probe.check_connection()
		if probe.connected and counter%2==0:
			sending = probe.send(name="Detection Overlay",imageData=processed)

		# Find slope and intercept of linear fit from the binary image
		slope, intercept = ImageProcessing.find_slope_intercept_from_binary(binary=binaryImage)

		# steering from slope and intercept
		rawSteering = 1.5*(slope - 0.3419) + (1/150)*(intercept+5)
		steering = steeringFilter.send((np.clip(rawSteering, -0.5, 0.5), dt))

		# Write steering to qcar
		new = gpad.read()
		QCarCommand = control_from_gamepad(gpad.buttonLeft, gpad.trigger, gpad.leftJoystickY, gpad.buttonA)
		if gpad.buttonX == 1:
			if math.isnan(steering):
				QCarCommand[1] = 0
			else:
				QCarCommand[1] = steering
			QCarCommand[0] = QCarCommand[0]*np.cos(steering)

		LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
		myCar.read_write_std(QCarCommand[0],QCarCommand[1],LEDs)

		end = time.time()
		dt = end - start

except KeyboardInterrupt:
	print("User interrupted!")
		
finally:
	# Terminate camera and QCar
	myCam.terminate()
	myCar.terminate()
	probe.terminate()
	gpad.terminate()
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

















import numpy as np
import cv2
import time
from pit.LaneNet.nets import LaneNet
from pal.utilities.vision import Camera3D


## Timing Parameters and methods 
def elapsed_time():
    return time.time() - startTime

sampleRate     = 30.0
sampleTime     = 1/sampleRate
simulationTime = 30.0
print('Sample Time: ', sampleTime)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
# Additional parameters
imageWidth  = 640
imageHeight = 480

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
# Initialize the LaneNet model
myLaneNet = LaneNet(
                    # modelPath = 'path/to/model', 
                    imageHeight = imageHeight,
                    imageWidth = imageWidth,
                    rowUpperBound = 228
                    )

# Initialize the RealSense camera for RGB 
myCamRGB  = Camera3D(mode='RGB', frameWidthRGB=imageWidth, frameHeightRGB=imageHeight)

try:
    startTime = time.time()
    while elapsed_time()<simulationTime:
        start = time.time()
        # Read the RGB
        myCamRGB.read_RGB()

        rgbProcessed=myLaneNet.pre_process(myCamRGB.imageBufferRGB)
        binaryPred , instancePred = myLaneNet.predict(rgbProcessed)
        annotatedImg = myLaneNet.render(showFPS = True)

        cv2.imshow('Extracted Lane', annotatedImg)

        # End timing this iteration
        end = time.time()

        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )

        # Pause/sleep for sleepTime in milliseconds
        msSleepTime = int(1000*sleepTime)
        if msSleepTime <= 0:
            msSleepTime = 1
        cv2.waitKey(msSleepTime)

except KeyboardInterrupt:
    print("User interrupted!")

finally:
    myCamRGB.terminate()
















## task_task_manual_drive.py
# This example demonstrates how to use the LogitechF710 to send throttle and steering
# commands to the QCar depending on 2 driving styles.
# Use the hardware_test_basic_io.py to troubleshoot uses trying to drive the QCar.

from pal.products.qcar import QCar
from pal.utilities.gamepad import LogitechF710
from pal.utilities.math import *

import os
import time
import struct
import numpy as np

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## Timing Parameters and methods
startTime = time.time()
def elapsed_time():
    return time.time() - startTime

sampleRate     = 50
sampleTime     = 1/sampleRate
simulationTime = 60.0
print('Sample Time: ', sampleTime)

# Additional parameters
counter = 0

# Initialize motor command array
QCarCommand = np.array([0,0])

# Set up a differentiator to get encoderSpeed from encoderCounts
diff = Calculus().differentiator_variable(sampleTime)
_ = next(diff)
timeStep = sampleTime

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## QCar and Gamepad Initialization
# Changing readmode to 0 to use imediate I/O
readMode = 0

myCar = QCar(readMode=readMode)
gpad = LogitechF710()

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## Driving Configuration: Use 3 toggles or 4 toggles mode as you see fit:
# Common to both 3 or 4 mode
#   Steering                    - Left Lateral axis
#   Arm                         - buttonLeft
# In 3 mode:
#   Throttle (Drive or Reverse) - Right Longitudonal axis
# In 4 mode:
#   Throttle                    - Right Trigger (always positive)
#   Button A                    - Reverse if held, Drive otherwise
configuration = '4' # change to '4' if required

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# Reset startTime before Main Loop
startTime = time.time()

## Main Loop
try:
    while elapsed_time() < simulationTime:
        # Start timing this iteration
        start = elapsed_time()

        # Read Gamepad states
        new = gpad.read()

        # Basic IO - write motor commands
        if configuration == '3':
            if new and gpad.buttonLeft:
                QCarCommand = np.array([0.3*gpad.rightJoystickY, 0.5*gpad.leftJoystickX])
        elif configuration == '4':
            if new and gpad.buttonLeft:
                if gpad.buttonA:
                    QCarCommand = np.array([-0.3*gpad.trigger, 0.5*gpad.leftJoystickX])
                else:
                    QCarCommand = np.array([0.3*gpad.trigger, 0.5*gpad.leftJoystickX])
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])

        # Adjust LED indicators based on steering and reverse indicators based on reverse gear
        if QCarCommand[1] > 0.3:
            LEDs[0] = 1
            LEDs[2] = 1
        elif QCarCommand[1] < -0.3:
            LEDs[1] = 1
            LEDs[3] = 1
        if QCarCommand[0] < 0:
            LEDs[5] = 1

        # Perform I/O
        myCar.read_write_std(throttle= QCarCommand[0],
                             steering= QCarCommand[1],
							 LEDs= LEDs)

        batteryVoltage = myCar.batteryVoltage

        # Estimate linear speed in m/s
        linearSpeed   = myCar.motorTach
        # encoderSpeed  = myCar.motorTach/myCar.CPS_TO_MPS
        # End timing this iteration
        end = elapsed_time()

        # Calculate computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - computationTime%sampleTime

        # Pause/sleep and print out the current timestamp
        time.sleep(sleepTime)

        if new:
            os.system('clear')
            print("Car Speed:\t\t\t{0:1.2f}\tm/s\nRemaining battery capacity:\t{1:4.2f}\t%\nMotor throttle:\t\t\t{2:4.2f}\t% PWM\nSteering:\t\t\t{3:3.2f}\trad"
                                                            .format(linearSpeed, 100 - (batteryVoltage - 10.5)*100/(12.6 - 10.5), QCarCommand[0], QCarCommand[1]))
        timeAfterSleep = elapsed_time()
        timeStep = timeAfterSleep - start
        counter += 1

except KeyboardInterrupt:
    print("User interrupted!")

finally:
    myCar.terminate()
    gpad.terminate()
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --




















## LIDAR_Point_Cloud.py
# This example uses the LiDAR point cloud to construct a temporary local map of the QCar's environment
# To troubleshoot the physical LiDAR use the hardware_test_rp_lidar_a2.py found in the hardware_tests directory

from pal.products.qcar import QCarLidar
from pal.utilities.math import *
import time
import numpy as np
import cv2

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## Timing Parameters and methods
startTime = time.time()
def elapsed_time():
	return time.time() - startTime

sampleRate 	   = 30
sampleTime 	   = 1/sampleRate
simulationTime = 30.0
print('Sample Time: ', sampleTime)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## Additional parameters and buffers
pixelsPerMeter 	  = 50 # pixels per meter
sideLengthScale = 8 * pixelsPerMeter # 8 meters width, or 400 pixels side length
decay 		      = 0.9 # 90% decay rate on old map data
maxDistance  	  = 3.9
map    		  	  = np.zeros((sideLengthScale, sideLengthScale), dtype=np.float32) # map object


# Lidar settings
numMeasurements 	 = 1000	# Points
lidarMeasurementMode 	 = 2
lidarInterpolationMode = 0

# LIDAR initialization and measurement buffers
myLidar = QCarLidar(
	numMeasurements=numMeasurements,
	rangingDistanceMode=lidarMeasurementMode,
	interpolationMode=lidarInterpolationMode
)


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## Main Loop
try:
	while elapsed_time() < simulationTime:
		# decay existing map
		map = decay*map

		# Start timing this iteration
		start = time.time()

		# Capture LIDAR data
		myLidar.read()

		# convert angles from lidar frame to body frame
		anglesInBodyFrame = myLidar.angles * -1 + np.pi


		# Find the points where it exceed the max distance and drop them off
		idx = [i for i, v in enumerate(myLidar.distances) if v < maxDistance]

		# convert distances and angles to XY contour
		x = myLidar.distances[idx]*np.cos(anglesInBodyFrame[idx])
		y = myLidar.distances[idx]*np.sin(anglesInBodyFrame[idx])

		# convert XY contour to pixels contour and update those pixels in the map
		pX = (sideLengthScale/2 - x*pixelsPerMeter).astype(np.uint16)
		pY = (sideLengthScale/2 - y*pixelsPerMeter).astype(np.uint16)

		map[pX, pY] = 1

		# End timing this iteration
		end = time.time()

		# Calculate the computation time, and the time that the thread should pause/sleep for
		computationTime = end - start
		sleepTime = sampleTime - ( computationTime % sampleTime )

		# Display the map at full resolution
		cv2.imshow('Map', map)

		# Pause/sleep for sleepTime in milliseconds
		msSleepTime = int(1000*sleepTime)
		if msSleepTime <= 0:
			msSleepTime = 1 # this check prevents an indefinite sleep as cv2.waitKey waits indefinitely if input is 0
		cv2.waitKey(msSleepTime)


except KeyboardInterrupt:
	print("User interrupted!")

finally:
	# Terminate the LIDAR object
	myLidar.terminate()
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --





















## rgbd_imaging.py
# This example combines the depth and RGB sensors from the Intel Realsense D435 to display objects 
# within a specified distance. For troubleshooting the Realsense camera use hardware_test_intelrealsense.py
# found in the hardwate_tests folder.  

from pal.utilities.vision import Camera3D
from hal.utilities.image_processing import ImageProcessing

import time
import struct
import numpy as np 
import cv2

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Timing Parameters and methods 
startTime = time.time()
def elapsed_time():
    return time.time() - startTime

sampleRate     = 30.0
sampleTime     = 1/sampleRate
simulationTime = 30.0
print('Sample Time: ', sampleTime)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
# Additional parameters
imageWidth  = 1280
imageHeight = 720

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Initialize the RealSense camera for RGB and Depth data
myCam1  = Camera3D(mode='RGB&DEPTH', frameWidthRGB=imageWidth, frameHeightRGB=imageHeight)
# max_distance_view = 5
MAX_DISTANCE = 0.6 # pixels in RGB image farther than this will appear white  
MIN_DISTANCE = 0.0001 # pixels in RGB image closer than this will appear black

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

## Main Loop
flag = True
try:
    while elapsed_time() < simulationTime:
        # Start timing this iteration
        start = time.time()

        # Read the RGB and Depth data (latter in meters)
        myCam1.read_RGB()
        myCam1.read_depth(dataMode='M')
        
        # Threshold the depth image based on min and max distance set above, and cast it to uint8 (to be used as a mask later)
        binaryNow = ImageProcessing.binary_thresholding(myCam1.imageBufferDepthM, MIN_DISTANCE, MAX_DISTANCE).astype(np.uint8)
        
        # Initialize binaryBefore to keep a 1 step time history of the binary to do a temporal difference filter later. 
        # At the first time step, flag = True. Initialize binaryBefore and then set flag = False to not do this again.
        if flag:
            binaryBefore = binaryNow
            flag = False
        
        # clean  =  closing filter applied ON ( binaryNow BITWISE AND ( BITWISE NOT of ( the ABSOLUTE of ( difference between binary now and before ) ) ) )
        binaryClean = ImageProcessing.image_filtering_close(cv2.bitwise_and( cv2.bitwise_not(np.abs(binaryNow - binaryBefore)/255), binaryNow/255 ), dilate=3, erode=1, total=1)

        # grab a smaller chunk of the depth data and scale it back to full resolution to account for field-of-view differences and physical distance between the RGB/Depth cameras.
        binaryClean = cv2.resize(binaryClean[81:618, 108:1132], (1280, 720)).astype(np.uint8)

        # Apply the binaryClean mask to the RGB image captured, and then display it.
        maskedRGB = cv2.bitwise_and(myCam1.imageBufferRGB, myCam1.imageBufferRGB, mask=binaryClean)
        cv2.imshow('Original', cv2.resize(maskedRGB, (640, 360)))
        
        # End timing this iteration
        end = time.time()

        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )

        # Pause/sleep for sleepTime in milliseconds
        msSleepTime = int(1000*sleepTime)
        if msSleepTime <= 0:
            msSleepTime = 1
        cv2.waitKey(msSleepTime)
        binaryBefore = binaryNow

except KeyboardInterrupt:
    print("User interrupted!")

finally:    
    # Terminate RealSense camera object
    myCam1.terminate()
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
















import numpy as np
import time
import cv2
from pit.YOLO.nets import YOLOv8
from pit.YOLO.utils import QCar2DepthAligned

## Timing Parameters and methods 
def elapsed_time():
    return time.time() - startTime

sampleRate     = 30.0
sampleTime     = 1/sampleRate
simulationTime = 30.0
print('Sample Time: ', sampleTime)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
# Additional parameters
imageWidth  = 640
imageHeight = 480

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
# Initialize YOLOv8 segmentation model
myYolo  = YOLOv8(
                 # modelPath = 'path/to/model', 
                 imageHeight= imageHeight,
                 imageWidth = imageWidth,
                )

# Initialize Depth/RGB alignment RT model
QCarImg = QCar2DepthAligned()

try:
    startTime = time.time()
    while elapsed_time()<simulationTime:
        start = time.time()

        # Get aligned RGB and Depth images
        QCarImg.read()
            
        rgbProcessed = myYolo.pre_process(QCarImg.rgb)
        predecion = myYolo.predict(inputImg = rgbProcessed,
                                   classes = [2,9,11],
                                   confidence = 0.3,
                                   half = True,
                                   verbose = False
                                   )
        
        processedResults=myYolo.post_processing(alignedDepth = QCarImg.depth,
                                                clippingDistance = 5)
        for object in processedResults:
            print(object.__dict__)
        print('---------------------------')

        # annotatedImg=myYolo.render()
        annotatedImg=myYolo.post_process_render(showFPS = True)
        cv2.imshow('Object Segmentation', annotatedImg)

        # End timing this iteration
        end = time.time()

        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )

        # Pause/sleep for sleepTime in milliseconds
        msSleepTime = int(1000*sleepTime)
        if msSleepTime <= 0:
            msSleepTime = 1
        cv2.waitKey(msSleepTime)

except KeyboardInterrupt:
    print("User interrupted!")
    
finally:
    QCarImg.terminate()



























'''
hardware:
'''




'''hardware_stop.py

This example correctly terminates the Lidar and QCar DAQ if they are still
in use. If the LIDAR is still spinning, or the QCar motor drive did not shut
off properly, use this script.
'''
from pal.products.qcar import QCar, QCarLidar


# Initializing QCar and Lidar
myLidar = QCarLidar()
myCar = QCar()

# Terminating DAQs if currently running
myCar.terminate()
myLidar.terminate()


















'''hardware_test_basic_io.py

This example demonstrates how to use the QCar class to perform basic I/O.
Learn how to write throttle and steering, as well as LED commands to the
vehicle, and read sensor data such as battery voltage. See the QCar class
definition for other sensor buffers such as motorTach, accelometer, gyroscope
etc.
'''
import numpy as np
import time
from pal.products.qcar import QCar, IS_PHYSICAL_QCAR

if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    qlabs_setup.setup()


#Initial Setup
sampleRate = 200
runTime = 5.0 # seconds

with QCar(readMode=1, frequency=sampleRate) as myCar:
    t0 = time.time()
    while time.time() - t0  < runTime:
        t = time.time()

        # Read from onboard sensors
        myCar.read()

        # Basic IO - write motor commands
        throttle = 0.1 * np.sin(t*2*np.pi/5)
        steering = 0.3 * np.sin(t*2*np.pi/2.5)

        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        if steering > 0.15:
            LEDs[0] = 1
            LEDs[2] = 1
        elif steering < -0.15:
            LEDs[1] = 1
            LEDs[3] = 1
        if throttle < 0:
            LEDs[5] = 1

        myCar.write(throttle, steering, LEDs)

        print(
            f'time: {(t-t0):.2f}'
            + f', Battery Voltage: {myCar.batteryVoltage:.2f}'
            + f', Motor Current: {myCar.motorCurrent:.2f}'
            + f', Motor Encoder: {myCar.motorEncoder}'
            + f', Motor Tach: {myCar.motorTach:.2f}'
            + f', Accelerometer: {myCar.accelerometer}'
            + f', Gyroscope: {myCar.gyroscope}'
        )

















        from pal.utilities.probe import Observer

observer = Observer()

for i in range(4):
    observer.add_display(imageSize = [410,820,3],
                        scalingFactor=2,
                        name='CSI'+str(i))

observer.launch()















'''hardware_test_csi_cameras.py

This example demonstrates how to read and display image data
from the 4 csi cameras.
'''
import time
import cv2
from pal.products.qcar import QCarCameras, IS_PHYSICAL_QCAR
import os
from pal.utilities.probe import Probe

# Initial Setup
ipHost, ipDriver = '192.168.3.10', 'localhost'
runTime = 30.0 # seconds
counter = 0
cameras = QCarCameras(
    enableBack=True,
    enableFront=True,
    enableLeft=True,
    enableRight=True,
)

try:
    t0 = time.time()
    probe = Probe(ip = ipHost)
    for i in range(4):
        probe.add_display(imageSize = [410, 820, 3], scaling = True,
                            scalingFactor= 2, name="CSI"+str(i))
    while time.time() - t0 < runTime:
        # print(probe.agents)
        if not probe.connected:
            probe.check_connection()
        if probe.connected:
            flags = cameras.readAll()
            if all(flags): counter +=1
            if counter % 40 == 0:
                for i, c in enumerate(cameras.csi):
                    sending = probe.send(name="CSI"+str(i),
                                    imageData=c.imageData)
except KeyboardInterrupt:
    print('User interrupted.')
finally:
    # Termination
    cameras.terminate()
    probe.terminate()
    
    '''hardware_test_csi_cameras.py

This example demonstrates how to read and display image data
from the 4 csi cameras.
'''
import time
import cv2
from pal.products.qcar import QCarCameras, IS_PHYSICAL_QCAR
import os
from pal.utilities.probe import Probe

# Initial Setup
ipHost, ipDriver = '192.168.3.10', 'localhost'
runTime = 30.0 # seconds
counter = 0
cameras = QCarCameras(
    enableBack=True,
    enableFront=True,
    enableLeft=True,
    enableRight=True,
)

try:
    t0 = time.time()
    probe = Probe(ip = ipHost)
    for i in range(4):
        probe.add_display(imageSize = [410, 820, 3], scaling = True,
                            scalingFactor= 2, name="CSI"+str(i))
    while time.time() - t0 < runTime:
        # print(probe.agents)
        if not probe.connected:
            probe.check_connection()
        if probe.connected:
            flags = cameras.readAll()
            if all(flags): counter +=1
            if counter % 40 == 0:
                for i, c in enumerate(cameras.csi):
                    sending = probe.send(name="CSI"+str(i),
                                    imageData=c.imageData)
except KeyboardInterrupt:
    print('User interrupted.')
finally:
    # Termination
    cameras.terminate()
    probe.terminate()






















    '''hardware_test_gamepad.py

This example demonstrates how to read data from the Logitech F710 gamepad.
The data received from buttons independently might change depending on the OS
(windows vs. linux)
'''
from pal.utilities.gamepad import LogitechF710
import time
import os


# Timing and Initialization
startTime = time.time()
def elapsed_time():
    return time.time() - startTime

simulationTime = 60
sampleRate = 100
sampleTime = 1/sampleRate
gpad = LogitechF710()

# Restart starTime just before Main Loop
startTime = time.time()

## Main Loop
try:
    while elapsed_time() < simulationTime:
        # Start timing this iteration
        start = elapsed_time()

        # Basic IO - write motor commands
        new = gpad.read()

        if new:
            # Clear the Screen for better readability
            os.system('clear')

            # Print out the gamepad IO read
            print("Left Laterial:\t\t{0:.2f}\nLeft Longitudonal:\t{1:.2f}\nTrigger:\t\t{2:.2f}\nRight Lateral:\t\t{3:.2f}\nRight Longitudonal:\t{4:.2f}"
                .format(gpad.leftJoystickX, gpad.leftJoystickY, gpad.trigger, gpad.rightJoystickX, gpad.rightJoystickY))
            print("Button A:\t\t{0:.0f}\nButton B:\t\t{1:.0f}\nButton X:\t\t{2:.0f}\nButton Y:\t\t{3:.0f}\nButton LB:\t\t{4:.0f}\nButton RB:\t\t{5:.0f}"
                .format(gpad.buttonA, gpad.buttonB, gpad.buttonX, gpad.buttonY, gpad.buttonLeft, gpad.buttonRight))
            print("Up:\t\t\t{0:.0f}\nRight:\t\t\t{1:.0f}\nDown:\t\t\t{2:.0f}\nLeft:\t\t\t{3:.0f}"
                .format(gpad.up, gpad.right, gpad.down, gpad.left))

        # End timing this iteration
        end = elapsed_time()

        # Calculate computation time, and the time that the thread should
        # pause/sleep for
        computation_time = end - start
        sleep_time = sampleTime - computation_time%sampleTime

        # Pause/sleep and print out the current timestamp
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("User interrupted!")

finally:
    # Terminate Joystick properly
    gpad.terminate()






















    '''hardware_test_intelrealsense.py

This example demonstrates how to read and display depth & RGB image data
from the Intel Realsense camera.
'''
import time
import cv2
from pal.products.qcar import QCarRealSense, IS_PHYSICAL_QCAR

if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    qlabs_setup.setup()


#Initial Setup
runTime = 30.0 # seconds
max_distance = 2 # meters (for depth camera)

with QCarRealSense(mode='RGB, Depth') as myCam:
    t0 = time.time()
    while time.time() - t0 < runTime:

        myCam.read_RGB()
        cv2.imshow('My RGB', myCam.imageBufferRGB)

        myCam.read_depth(dataMode='PX')
        cv2.imshow('My Depth', myCam.imageBufferDepthPX/max_distance)

        cv2.waitKey(100)






















'''hardware_test_intelrealsense.py

This example demonstrates how to read and display depth & RGB image data
from the Intel Realsense camera.
'''
import time
import cv2
from pal.products.qcar import QCarRealSense, IS_PHYSICAL_QCAR

if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    qlabs_setup.setup()


#Initial Setup
runTime = 20.0 # seconds

with QCarRealSense(mode='IR') as myCam:
    t0 = time.time()
    while time.time() - t0 < runTime:

        myCam.read_IR()

        cv2.imshow('Left IR Camera', myCam.imageBufferIRLeft)
        cv2.imshow('Right IR Camera', myCam.imageBufferIRRight)

        cv2.waitKey(100)






















        '''This example demonstrates how to read and display data from the QCar Lidar
'''
import time
import matplotlib.pyplot as plt
from pal.products.qcar import QCarLidar
# from pal.utilities.lidar import Lidar

# polar plot object for displaying LIDAR data later on
ax = plt.subplot(111, projection='polar')
plt.show(block=False)

runTime = 10.0 # seconds
# Lidar settings
numMeasurements 	 = 1000	# Points
lidarMeasurementMode 	 = 2
lidarInterpolationMode = 0

# LIDAR initialization and measurement buffers
myLidar = QCarLidar(
	numMeasurements=numMeasurements,
	rangingDistanceMode=lidarMeasurementMode,
	interpolationMode=lidarInterpolationMode
)


t0 = time.time()
while time.time() - t0  < runTime:
    plt.cla()

    # Capture LIDAR data
    myLidar.read()

    ax.scatter(myLidar.angles, myLidar.distances, marker='.')
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)

    plt.pause(0.1)

myLidar.terminate()


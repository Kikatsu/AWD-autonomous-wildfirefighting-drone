from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal
import cv2
import numpy as np
import time
import tensorflow.lite as tflite

# Connect to the drone
vehicle = connect('/dev/ttyAMA0', baud=57600, wait_ready=True)

# Load the trained TensorFlow Lite fire detection model
interpreter = tflite.Interpreter(model_path="fire_detection_model.tflite")
interpreter.allocate_tensors()

# Load the Aruco dictionary
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters_create()

# Define GPS landing stations for mountain terrain
mountain_landing_zones = [
    LocationGlobalRelative(43.2567, 76.9286, 50),  # Station 1
    LocationGlobalRelative(43.2590, 76.9300, 50),  # Station 2
    LocationGlobalRelative(43.2610, 76.9350, 50)   # Station 3
]

# Battery replacement station
battery_swap_station = LocationGlobal(43.2550, 76.9270, 10)  # Ground-level station

# Function to capture an image from the drone's camera
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# Function to detect Aruco markers
def detect_aruco_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    return ids[0][0] if ids is not None else None

# Function to detect fire using AI
def detect_fire(image):
    image_resized = cv2.resize(image, (224, 224))
    image_resized = np.expand_dims(image_resized, axis=0).astype(np.float32) / 255.0

    input_tensor_index = interpreter.get_input_details()[0]['index']
    output_tensor_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_tensor_index, image_resized)
    interpreter.invoke()
    result = interpreter.get_tensor(output_tensor_index)[0]

    return result[0] > 0.8  # Fire detection threshold (80%)

# Function to send fire alert
def send_fire_alert():
    latitude = vehicle.location.global_relative_frame.lat
    longitude = vehicle.location.global_relative_frame.lon
    print(f"🔥 FIRE DETECTED! Coordinates: {latitude}, {longitude}")

# Function to move the drone to a GPS waypoint
def go_to(location):
    vehicle.simple_goto(location)
    time.sleep(10)  # Simulate time to reach waypoint

# Function to check battery level
def check_battery():
    return vehicle.battery.level  # Get battery percentage

# Function for safe landing using Aruco markers
def land_on_marker():
    print("🛬 Searching for landing marker...")
    while True:
        frame = capture_image()
        if frame is not None and detect_aruco_marker(frame):
            print("✅ Landing marker found! Preparing to land...")
            vehicle.mode = VehicleMode("LAND")
            break

# **Main Flight Operation**
print("🚀 Starting autonomous patrol...")
vehicle.mode = VehicleMode("GUIDED")
vehicle.armed = True

# Wait until drone is ready for takeoff
while not vehicle.armed:
    print("⏳ Waiting for takeoff...")
    time.sleep(1)

vehicle.simple_takeoff(10)  # Take off to 10 meters for marker scan
time.sleep(5)

# **Scan for Aruco marker ID**
frame = capture_image()
terrain_mode = "mountain"  # Default mode
if frame is not None:
    marker_id = detect_aruco_marker(frame)
    if marker_id == 1:
        terrain_mode = "flat"
        print("🌍 Flat terrain detected. Flying at 50-100m altitude.")
    else:
        print("⛰️ Mountain terrain detected. Using landing stations.")

# **Flight Strategy Based on Terrain**
if terrain_mode == "flat":
    go_to(LocationGlobalRelative(vehicle.location.global_frame.lat, vehicle.location.global_frame.lon, 100))  # Climb to 100m
    print("🔍 Scanning area for fires from altitude.")
else:
    for landing_zone in mountain_landing_zones:
        go_to(landing_zone)
        print(f"📍 Landed at station {landing_zone.lat}, {landing_zone.lon}")
        frame = capture_image()
        if frame is not None and detect_fire(frame):
            send_fire_alert()
            break

# **Fire Detection Routine**
while True:
    frame = capture_image()
    if frame is not None and detect_fire(frame):
        send_fire_alert()
        break

    # **Battery Check**
    battery_level = check_battery()
    print(f"🔋 Battery level: {battery_level}%")
    if battery_level < 20:
        print("⚠️ Battery low! Returning to battery swap station.")
        go_to(battery_swap_station)
        land_on_marker()
        break

# **Return to home base**
if vehicle.mode.name != "LAND":
    print("🏠 Returning to home base...")
    vehicle.mode = VehicleMode("RTL")

# Close the vehicle connection
vehicle.close()

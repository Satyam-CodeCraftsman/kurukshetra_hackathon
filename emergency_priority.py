#prioritizing emergency vehicles such as ambulance or firebrigade (team- trikaal drishti)
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Ensure you have the model file

# Open Webcam
cap = cv2.VideoCapture(0)

# Define traffic light colors
traffic_lights = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0)}
current_light = "RED"

# Define class IDs for emergency vehicles (Check YOLO's class list for your model)
EMERGENCY_VEHICLE_CLASSES = [2, 5]  # Example: Class 2 (Car), Class 5 (Bus), replace with correct ones

# Function to detect and count vehicles, and check for emergency vehicles
def detect_vehicles(frame):
    results = model(frame)
    vehicle_count = 0
    emergency_detected = False

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            cls = int(cls)
            
            if cls in [2, 3, 5, 7]:  # Detect normal vehicles (Car, Bus, Truck, Motorcycle)
                vehicle_count += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            if cls in EMERGENCY_VEHICLE_CLASSES:  # Detect emergency vehicles
                emergency_detected = True
                cv2.putText(frame, "EMERGENCY VEHICLE DETECTED!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return vehicle_count, emergency_detected, frame

# Function to update traffic light
def update_traffic_light(vehicle_count, emergency_detected):
    global current_light

    if emergency_detected:
        current_light = "GREEN"  # Give priority to emergency vehicle
    elif vehicle_count > 10:
        current_light = "GREEN"
    elif 5 <= vehicle_count <= 10:
        current_light = "YELLOW"
    else:
        current_light = "RED"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect vehicles and check for emergency vehicles
    vehicle_count, emergency_detected, processed_frame = detect_vehicles(frame)

    # Update traffic light based on normal and emergency vehicle detection
    update_traffic_light(vehicle_count, emergency_detected)

    # Display Traffic Light Status
    cv2.circle(processed_frame, (50, 50), 20, traffic_lights[current_light], -1)
    cv2.putText(processed_frame, f"Traffic Light: {current_light}", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, traffic_lights[current_light], 2)

    # Display Vehicle Count
    cv2.putText(processed_frame, f"Vehicle Count: {vehicle_count}", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("AI Traffic Management System", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

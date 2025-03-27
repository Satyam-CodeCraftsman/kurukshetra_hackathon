# congestion based traffic light management system (team trikaal drishti )
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import time

# Load YOLO models
vehicle_model = YOLO("yolov8n.pt")  # General object detection
plate_model = YOLO("license_plate_detector.pt")  
reader = easyocr.Reader(['en'])

# Define traffic light colors
traffic_lights = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0)}
current_light = "RED"

# Fine database (stores vehicle numbers and fines)
fine_database = {}

# Open Webcam
cap = cv2.VideoCapture(0)

def detect_vehicles(frame):
    """Detects vehicles using YOLOv8."""
    results = vehicle_model(frame)
    vehicle_count = 0
    vehicle_boxes = []

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            if int(cls) in [2, 3, 5, 7]:  # Only count vehicles (Car, Bus, Truck, Motorcycle)
                vehicle_count += 1
                vehicle_boxes.append((int(x1), int(y1), int(x2), int(y2)))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    return vehicle_count, frame, vehicle_boxes

def detect_number_plate(frame):
    """Detects license plates using a YOLO model and returns cropped plates."""
    results = plate_model(frame)
    plates = []

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            plate_img = frame[int(y1):int(y2), int(x1):int(x2)]
            plates.append((plate_img, (x1, y1, x2, y2)))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    return plates, frame

def recognize_plate(plate_img):
    """Extracts text from a detected license plate using OCR."""
    result = reader.readtext(plate_img)
    return result[0][-2] if result else None

def update_traffic_light(vehicle_count):
    """Updates the traffic light based on congestion."""
    global current_light
    if vehicle_count > 10:
        current_light = "GREEN"
    elif 5 <= vehicle_count <= 10:
        current_light = "YELLOW"
    else:
        current_light = "RED"

def process_frame(frame):
    """Processes each frame for vehicle detection, plate recognition, and violation tracking."""
    global fine_database
    vehicle_count, processed_frame, vehicle_boxes = detect_vehicles(frame)
    update_traffic_light(vehicle_count)

    # If the light is RED, check for violations
    if current_light == "RED":
        plates, processed_frame = detect_number_plate(processed_frame)

        for plate_img, coords in plates:
            plate_number = recognize_plate(plate_img)
            if plate_number:
                if plate_number not in fine_database:
                    fine_database[plate_number] = 1
                    print(f"🚨 Fine issued to {plate_number} for signal jumping!")
                    
                    # Save proof image
                    cv2.imwrite(f"violations/{plate_number}_{int(time.time())}.jpg", plate_img)

                    # Draw bounding box
                    x1, y1, x2, y2 = coords
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(processed_frame, f"Fine Issued: {plate_number}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display Traffic Light Status
    cv2.circle(processed_frame, (50, 50), 20, traffic_lights[current_light], -1)
    cv2.putText(processed_frame, f"Traffic Light: {current_light}", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, traffic_lights[current_light], 2)

    # Display Vehicle Count
    cv2.putText(processed_frame, f"Vehicle Count: {vehicle_count}", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return processed_frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame)
    cv2.imshow("Traffic Violation Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

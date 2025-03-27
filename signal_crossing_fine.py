#fine for those who jumps signal (team-trikaal drishti)
import cv2
import numpy as np
import easyocr
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# Load the YOLO model for license plate detection
plate_model = YOLO("license_plate_detector.pt")  # Ensure this file exists

# Initialize OCR reader
reader = easyocr.Reader(["en"])

# Open webcam
cap = cv2.VideoCapture(0)

# Define traffic light state
traffic_light = "RED"  # Change dynamically as needed

# CSV file for logging fines
fine_log = "fines.csv"

# Function to detect number plates
def detect_plate(frame):
    results = plate_model(frame)
    plates = []

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Extract plate region
            plate_img = frame[y1:y2, x1:x2]
            plates.append((plate_img, (x1, y1, x2, y2)))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return plates, frame

# Function to recognize text from number plate
def recognize_text(plate_img):
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(plate_gray)

    for (bbox, text, prob) in result:
        if prob > 0.5:  # Confidence threshold
            return text.strip().replace(" ", "")  # Clean text

    return None

# Function to log fine
def log_fine(number_plate):
    data = {"Plate Number": [number_plate], "Time": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")], "Fine": ["Signal Jump"]}
    df = pd.DataFrame(data)

    try:
        existing_data = pd.read_csv(fine_log)
        df = pd.concat([existing_data, df], ignore_index=True)
    except FileNotFoundError:
        pass  # First-time logging

    df.to_csv(fine_log, index=False)
    print(f"Fine logged for: {number_plate}")

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect number plates
    plates, processed_frame = detect_plate(frame)

    for plate_img, bbox in plates:
        plate_text = recognize_text(plate_img)

        if plate_text:
            x1, y1, x2, y2 = bbox
            cv2.putText(processed_frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # If the traffic light is red, log a fine
            if traffic_light == "RED":
                log_fine(plate_text)

    # Show traffic light status
    color = (0, 0, 255) if traffic_light == "RED" else (0, 255, 0)
    cv2.circle(processed_frame, (50, 50), 20, color, -1)
    cv2.putText(processed_frame, f"Light: {traffic_light}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("ANPR System", processed_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

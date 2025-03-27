#accident_detection (team-trikaal drishti)
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import smtplib
from email.message import EmailMessage
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

# Store previous vehicle positions for motion tracking
previous_positions = {}

# Function to send an ambulance notification
def send_ambulance_notification(location="Unknown Location"):
    sender_email = "traffic_management@gmail.com"
    receiver_email = "ambulance_service@example.com"
    password = "xyz"
    
    msg = EmailMessage()
    msg.set_content(f"🚨 ACCIDENT DETECTED at {location}! Send an ambulance immediately!")
    msg["Subject"] = "URGENT: Accident Detected!"
    msg["From"] = sender_email
    msg["To"] = receiver_email
    
    try:
        server = smtplib.SMTP("smtp.example.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        print("🚑 Ambulance notification sent!")
    except Exception as e:
        print("Failed to send notification:", e)

# Detect accident in video frames
def detect_accident(frame):
    global previous_positions
    results = model(frame)
    accident_detected = False
    new_positions = {}

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])  # Class ID from YOLO model
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            width, height = x2 - x1, y2 - y1
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Vehicle classes (COCO dataset)
            if confidence > 0.5 and class_id in [2, 3, 5, 7]:  # Bicycle, Car, Bus, Truck
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Vehicle {class_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                new_positions[center_x] = (center_x, center_y)

                # 🚨 **1. Flipped Vehicle Detection (Direct Accident)**
                if class_id in [3, 5, 7]:  # Cars, Buses, Trucks
                    if width > height * 2:  # Car flipped (width is 2x height)
                        accident_detected = True
                        cv2.putText(frame, "🚨 ACCIDENT DETECTED!", (x1, y1 - 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # 🚨 **2. Sudden Stop Detection (Direct Accident)**
                if center_x in previous_positions:
                    old_x, old_y = previous_positions[center_x]
                    distance = np.sqrt((center_x - old_x) ** 2 + (center_y - old_y) ** 2)

                    if distance < 3:  # If vehicle stops completely
                        accident_detected = True
                        cv2.putText(frame, "🚨 ACCIDENT DETECTED!", (x1, y1 - 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    previous_positions = new_positions  # Update positions for next frame
    return accident_detected

# Main function for video playback
def main():
    video_path = "traffic_video.mp4"  # Change to your video file
    cap = cv2.VideoCapture(video_path)  

    # Get video properties (Optional: Save processed video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output_video.avi', fourcc, fps, (width, height))

    last_notification_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends

        accident_detected = detect_accident(frame)

        # 🚨 **Send notification if accident detected**
        if accident_detected and time.time() - last_notification_time > 30:
            send_ambulance_notification("Intersection A, Main Street")
            last_notification_time = time.time()

        # Save processed frame (Optional)
        out.write(frame)

        # Display video
        cv2.imshow("Accident Detection - Video Playback", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit early
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

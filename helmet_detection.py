import cv2
import easyocr
from ultralytics import YOLO

# Load YOLO model trained for helmet detection
model = YOLO("helmet_detection.pt")  # Download and use a trained model

# Initialize OCR for number plate detection
reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture(0)  # Use webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Perform detection

    helmet_detected = False
    person_detected = False
    bike_detected = False

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            cls = int(cls)

            # Custom YOLO model classes (check the dataset used)
            if cls == 0:  # Person
                person_detected = True
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            elif cls == 1:  # Motorcycle
                bike_detected = True
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            elif cls == 2:  # Helmet (check dataset class ID)
                helmet_detected = True
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

            elif cls == 3:  # License Plate
                license_plate = frame[int(y1):int(y2), int(x1):int(x2)]
                text = reader.readtext(license_plate)
                
                if text:
                    plate_number = text[0][-2]  # Extract license plate number
                    cv2.putText(frame, f"Plate: {plate_number}", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # If bike & person are detected but no helmet
    if bike_detected and person_detected and not helmet_detected:
        cv2.putText(frame, "No Helmet! Fine Imposed", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

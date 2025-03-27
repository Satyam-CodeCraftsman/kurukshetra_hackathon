import cv2
import face_recognition
import os

# Folder containing multiple images of the criminal
criminal_images_folder = "criminals"  # Store criminal images here

criminal_encodings = []
criminal_name = "John Doe"
crime_details = "Wanted for Robbery"
alert_file = "alert.txt"

# Load multiple images of the criminal and encode them
for filename in os.listdir(criminal_images_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(criminal_images_folder, filename)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            criminal_encodings.append(encodings[0])  # Store face encodings

# Open the video file
video_path = 0  # Replace with your actual video file (using 0 for web cam // you may use video file)
cap = cv2.VideoCapture(video_path)

criminal_detected = False  # Flag to track if criminal is found

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(criminal_encodings, face_encoding, tolerance=0.5)
        
        if True in matches:
            # Draw a rectangle around the criminal's face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
            cv2.putText(frame, f"{criminal_name} - {crime_details}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Mark criminal detected
            cv2.putText(frame, "Criminal Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Generate alert file if not already created
            if not criminal_detected:
                with open(alert_file, "w") as f:
                    f.write(f"ALERT: {criminal_name} detected in cam.mp4\n")
                    f.write(f"Crime: {crime_details}\n")
                    f.write("Immediate action required!\n")
                criminal_detected = True  # Prevent duplicate alerts

    cv2.imshow("Criminal Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Alert file '{alert_file}' generated if a criminal was detected.")

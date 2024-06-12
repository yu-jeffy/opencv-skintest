import cv2
import numpy as np
import face_recognition

# Define a function to classify skin tone
def classify_skin_tone(mean_bgr):
    if mean_bgr[2] > 150:
        return 'Light'
    elif mean_bgr[2] > 100:
        return 'Fair'
    else:
        return 'Dark'

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = frame[:, :, ::-1]

    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_frame)

    if face_locations:
        for face_location in face_locations:
            top, right, bottom, left = face_location

            # Extract the face region
            face_region = frame[top:bottom, left:right]

            if face_region.size != 0:
                # Calculate the mean color of the face region
                mean_color = cv2.mean(face_region)[:3]

                # Classify the skin tone
                skin_tone = classify_skin_tone(mean_color)

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Display the skin tone classification
                cv2.putText(frame, f'Skin Tone: {skin_tone}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Display "White" or "Black" based on skin tone
                race_text = "No N-Word Pass Detected" if skin_tone in ["Light", "Fair"] else "N-Word Pass Detected" if skin_tone == "Dark" else "Unknown"
                cv2.putText(frame, race_text, (left, top - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame
    cv2.imshow('Skin Tone Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

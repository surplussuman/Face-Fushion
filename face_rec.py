import face_recognition
import cv2
import pickle

# Load the trained model
with open('models/trained_model.pkl', 'rb') as f:
    data = pickle.load(f)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and get encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Calculate distances between the face encoding and known encodings
        distances = face_recognition.face_distance(data["encodings"], face_encoding)
        
        # Find the best match
        min_distance = min(distances)
        if min_distance < 0.6:  # Threshold for considering a match (you can adjust this)
            best_match_index = distances.tolist().index(min_distance)
            name = data["labels"][best_match_index]
            confidence = (1 - min_distance) * 100  # Convert distance to confidence percentage
        else:
            name = "Unknown"
            confidence = 0

        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Display name and confidence
        text = f"{name} ({confidence:.2f}%)"
        cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

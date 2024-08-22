import face_recognition
import cv2
import pickle

# Load the trained model
with open('models/trained_model_cnn.pkl', 'rb') as f:
    data = pickle.load(f)

# Initialize webcam
cap = cv2.VideoCapture('rtsp://admin:admin@123@172.16.21.15:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif')

# print("Loaded encodings:", data["encodings"])
# print("Loaded labels:", data["labels"])


while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    frame = cv2.resize(frame, (640,480))

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and get encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # print("Face locations:", face_locations)
    # print("Face encodings:", face_encodings)


    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Calculate distances between the face encoding and known encodings
        distances = face_recognition.face_distance(data["encodings"], face_encoding)

        # print("Distances", distances)
        
        # Find the best match
        min_distance = min(distances)
        if min_distance < 0.6:  # Threshold for considering a match (you can adjust this)
            best_match_index = distances.tolist().index(min_distance)
            name = data["labels"][best_match_index]
            confidence = (1 - min_distance) * 100  # Convert distance to confidence percentage
        else:
            name = "Unknown"
            confidence = 0

        # print(f"Detected {name} with confidence {confidence:.2f}%")


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

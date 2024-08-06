import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch
import pickle
import face_recognition

# Load the trained model and label encoder
with open('models/facenet_recognition_model.pkl', 'rb') as f:
    classifier, label_encoder = pickle.load(f)
    

# Initialize the FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    for (top, right, bottom, left) in face_locations:
        face = rgb_frame[top:bottom, left:right]
        face = cv2.resize(face, (160, 160))
        face = np.transpose(face, (2, 0, 1))
        face = torch.tensor(face, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            embedding = model(face).numpy().flatten()
        
        # Predict the label
        probs = classifier.predict_proba([embedding])[0]
        label = label_encoder.inverse_transform([np.argmax(probs)])[0]
        confidence = np.max(probs) * 100
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Display the label and confidence
        text = f"{label} ({confidence:.2f}%)"
        cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

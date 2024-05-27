import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch
import pickle
import face_recognition

# Load the trained model and label encoder
with open('multimodal_face_recognition_model.pkl', 'rb') as f:
    classifier, label_encoder = pickle.load(f)

# Initialize the FaceNet models
facenet_rgb = InceptionResnetV1(pretrained='vggface2').eval()
facenet_ir = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize webcam
cap_rgb = cv2.VideoCapture(0)
cap_ir = cv2.VideoCapture(1)  # Assuming IR camera is the second camera

while True:
    # Read the frames from the webcams
    ret_rgb, frame_rgb = cap_rgb.read()
    ret_ir, frame_ir = cap_ir.read()

    # Convert the RGB frame to RGB color space
    rgb_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
    ir_frame = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_locations_rgb = face_recognition.face_locations(rgb_frame)
    face_locations_ir = face_recognition.face_locations(ir_frame)
    
    for (top_rgb, right_rgb, bottom_rgb, left_rgb), (top_ir, right_ir, bottom_ir, left_ir) in zip(face_locations_rgb, face_locations_ir):
        # Process RGB face
        face_rgb = rgb_frame[top_rgb:bottom_rgb, left_rgb:right_rgb]
        face_rgb = cv2.resize(face_rgb, (160, 160))
        face_rgb = np.transpose(face_rgb, (2, 0, 1))
        face_rgb = torch.tensor(face_rgb, dtype=torch.float32).unsqueeze(0)
        
        # Process IR face
        face_ir = ir_frame[top_ir:bottom_ir, left_ir:right_ir]
        face_ir = cv2.resize(face_ir, (160, 160))
        face_ir = np.expand_dims(face_ir, axis=0)
        face_ir = np.expand_dims(face_ir, axis=0)
        face_ir = torch.tensor(face_ir, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            embedding_rgb = facenet_rgb(face_rgb).numpy().flatten()
            embedding_ir = facenet_ir(face_ir).numpy().flatten()
        
        # Concatenate RGB and IR embeddings
        embedding = np.concatenate((embedding_rgb, embedding_ir))
        
        # Predict the label
        probs = classifier.predict_proba([embedding])[0]
        label = label_encoder.inverse_transform([np.argmax(probs)])[0]
        confidence = np.max(probs) * 100
        
        # Draw rectangle around the face
        cv2.rectangle(frame_rgb, (left_rgb, top_rgb), (right_rgb, bottom_rgb), (0, 255, 0), 2)
        
        # Display the label and confidence
        text = f"{label} ({confidence:.2f}%)"
        cv2.putText(frame_rgb, text, (left_rgb, top_rgb - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Multimodal Face Recognition', frame_rgb)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcams and close the OpenCV windows
cap_rgb.release()
cap_ir.release()
cv2.destroyAllWindows

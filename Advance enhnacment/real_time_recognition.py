'''import cv2
import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
from enhancement import preprocess_image, apply_super_resolution, detect_face
import joblib

# Load pre-trained FaceNet models for RGB and IR
facenet_rgb = InceptionResnetV1(pretrained='vggface2').eval()
facenet_ir = InceptionResnetV1(pretrained='casia-webface').eval()

# Load the trained SVM model and label encoder
clf = joblib.load('models/face_fusion_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

def process_frame(frame):
    # Super resolution
    frame = apply_super_resolution(frame)
    
    # Detect faces
    faces = detect_face(frame)
    
    embeddings = []
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = preprocess_image(face)
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)  # Convert back to 3-channel image
        face = cv2.resize(face, (160, 160))
        face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            embedding_rgb = facenet_rgb(face).numpy().flatten()
            embedding_ir = facenet_ir(face).numpy().flatten()
            embedding = np.concatenate((embedding_rgb, embedding_ir))
        embeddings.append(embedding)
    
    return embeddings, faces

# Example usage
cap = cv2.VideoCapture(0)  # Use 0 for webcam or 'path_to_cctv_video.mp4' for video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    embeddings, faces = process_frame(frame)
    for embedding, (x, y, w, h) in zip(embeddings, faces):
        pred_proba = clf.predict_proba([embedding])[0]
        pred_label = clf.classes_[np.argmax(pred_proba)]
        label = label_encoder.inverse_transform([pred_label])[0]
        
        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''

# Face recognition using webcam
'''import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib

# Load pre-trained FaceNet models for RGB and IR
facenet_rgb = InceptionResnetV1(pretrained='vggface2').eval()
facenet_ir = InceptionResnetV1(pretrained='casia-webface').eval()

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained model
clf = joblib.load('face_fusion_model.pkl')
le = joblib.load('label_encoder.pkl')

def preprocess_image(img, is_rgb=True):
    if is_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to 3-channel image
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if not already
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to 3-channel image
    img = cv2.resize(img, (160, 160))
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img

# Real-time recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_rgb = frame[y:y+h, x:x+w]
        face_ir = gray[y:y+h, x:x+w]

        face_rgb = preprocess_image(face_rgb, is_rgb=True)
        face_ir = preprocess_image(face_ir, is_rgb=False)

        with torch.no_grad():
            embedding_rgb = facenet_rgb(face_rgb).numpy().flatten()
            embedding_ir = facenet_ir(face_ir).numpy().flatten()

        combined_embedding = np.concatenate((embedding_rgb, embedding_ir))
        prediction = clf.predict([combined_embedding])
        predicted_label = le.inverse_transform(prediction)[0]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Real-Time Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
# Face rec on cctv
'''import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from joblib import load
from enhancement import preprocess_image

# Load the SVM model and label encoder
svm_model = load('models/face_fusion_model.pkl')
label_encoder = load('models/label_encoder.pkl')

# Load pre-trained FaceNet models for RGB and IR
facenet_rgb = InceptionResnetV1(pretrained='vggface2').eval()
facenet_ir = InceptionResnetV1(pretrained='casia-webface').eval()

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_features(img_rgb, img_ir):
    # Process RGB image
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (160, 160))
    img_rgb = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        embedding_rgb = facenet_rgb(img_rgb).numpy().flatten()

    # Process IR image
    img_ir = cv2.resize(img_ir, (160, 160))
    img_ir = cv2.cvtColor(img_ir, cv2.COLOR_GRAY2RGB)  # Ensure 3-channel input
    img_ir = torch.tensor(img_ir).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        embedding_ir = facenet_ir(img_ir).numpy().flatten()
    
    return np.concatenate((embedding_rgb, embedding_ir))

# Read from video file
video_path = 'video.mkv'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.resize(frame, (640,480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_rgb = frame[y:y+h, x:x+w]
        face_ir = gray[y:y+h, x:x+w]

        # Preprocess the detected faces
        face_rgb = preprocess_image(face_rgb, is_rgb=True)
        face_ir = preprocess_image(face_ir, is_rgb=False)

        # Extract features
        features = extract_features(face_rgb, face_ir)

        # Recognize face
        label = svm_model.predict([features])[0]
        name = label_encoder.inverse_transform([label])[0]

        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Video Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''

# Accuracy 214%
'''
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib

# Load pre-trained FaceNet models for RGB and IR
facenet_rgb = InceptionResnetV1(pretrained='vggface2').eval()
facenet_ir = InceptionResnetV1(pretrained='casia-webface').eval()

# Load the face detection model (DNN-based)
prototxt = 'deploy.prototxt'
caffemodel = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

# Load the trained model and label encoder
clf = joblib.load('face_fusion_model.pkl')
le = joblib.load('label_encoder.pkl')

def preprocess_image(img, is_rgb=True):
    if is_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (160, 160))
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img

def extract_features(img_rgb, img_ir):
    img_rgb = preprocess_image(img_rgb, is_rgb=True)
    img_ir = preprocess_image(img_ir, is_rgb=False)
    
    with torch.no_grad():
        embedding_rgb = facenet_rgb(img_rgb).numpy().flatten()
        embedding_ir = facenet_ir(img_ir).numpy().flatten()
    
    return np.concatenate((embedding_rgb, embedding_ir))

def detect_faces(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))
    return faces

# Real-time recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)
    for (x, y, w, h) in faces:
        face_rgb = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_ir = gray[y:y+h, x:x+w]

        features = extract_features(face_rgb, face_ir)
        prediction = clf.predict([features])
        confidence = clf.decision_function([features])[0]

        # Handle low-confidence predictions
        if confidence.max() < 0.5:
            predicted_label = 'Unknown'
            accuracy = confidence.max() * 100
        else:
            predicted_label = le.inverse_transform(prediction)[0]
            accuracy = confidence.max() * 100

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        text = f"{predicted_label}: {accuracy:.2f}%"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Real-Time Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'''

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load pre-trained FaceNet models for RGB and IR
facenet_rgb = InceptionResnetV1(pretrained='vggface2').eval()
facenet_ir = InceptionResnetV1(pretrained='casia-webface').eval()

# Load the face detection model (MTCNN)
mtcnn = MTCNN(keep_all=True)

# Load the trained model and label encoder
clf = joblib.load('models/face_fusion_model.pkl')
le = joblib.load('models/label_encoder.pkl')

# Initialize MinMaxScaler for confidence normalization
scaler = MinMaxScaler()

def preprocess_image(img, is_rgb=True):
    if is_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (160, 160))
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img

def extract_features(img_rgb, img_ir):
    img_rgb = preprocess_image(img_rgb, is_rgb=True)
    img_ir = preprocess_image(img_ir, is_rgb=False)
    
    with torch.no_grad():
        embedding_rgb = facenet_rgb(img_rgb).numpy().flatten()
        embedding_ir = facenet_ir(img_ir).numpy().flatten()
    
    return np.concatenate((embedding_rgb, embedding_ir))

# Real-time recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using MTCNN
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            face_rgb = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_ir = gray[y1:y2, x1:x2]

            features = extract_features(face_rgb, face_ir)
            prediction = clf.predict([features])
            decision_values = clf.decision_function([features])
            
            # Normalize confidence scores
            confidence_normalized = scaler.fit_transform(decision_values.reshape(-1, 1)).flatten()

            # Get the max confidence and corresponding label
            max_confidence = confidence_normalized.max()
            if max_confidence < 0.5:
                predicted_label = 'Unknown'
            else:
                predicted_label = le.inverse_transform(prediction)[0]

            accuracy = max_confidence * 100  # Convert to percentage

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f"{predicted_label}: {accuracy:.2f}%"
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Real-Time Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Face rec on cctv processed frames (Pygame)

'''import cv2
import pygame
from pygame.locals import *
import threading
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from joblib import load
import time

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained FaceNet models for RGB and IR
facenet_rgb = InceptionResnetV1(pretrained='vggface2').eval()
facenet_ir = InceptionResnetV1(pretrained='casia-webface').eval()

# Load the SVM model and label encoder
svm_model = load('models/face_fusion_model.pkl')
label_encoder = load('models/label_encoder.pkl')

# Initialize Pygame
pygame.init()

# Create a Pygame window
screen = pygame.display.set_mode((640, 480))

# Define the RTSP URL or video file path
# rtsp_url = 0  # For webcam
rtsp_url = 'video.mkv'  # Replace with your video file path
cap = cv2.VideoCapture(rtsp_url)

def preprocess_image(img, is_rgb=True):
    if is_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to 3-channel image
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if not already
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to 3-channel image
    img = cv2.resize(img, (160, 160))
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img

def extract_features(img_rgb, img_ir):
    # Process RGB image
    with torch.no_grad():
        embedding_rgb = facenet_rgb(img_rgb).numpy().flatten()

    # Process IR image
    with torch.no_grad():
        embedding_ir = facenet_ir(img_ir).numpy().flatten()
    
    return np.concatenate((embedding_rgb, embedding_ir))

def capture_video():
    frame_count = 0  # To count frames
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        # Process every nth frame for face recognition to reduce load
        if frame_count % 10 == 0:  # Adjust the frequency as needed
            for (x, y, w, h) in faces:
                face_rgb = frame[y:y+h, x:x+w]
                face_ir = gray[y:y+h, x:x+w]

                face_rgb = preprocess_image(face_rgb, is_rgb=True)
                face_ir = preprocess_image(face_ir, is_rgb=False)

                # Extract features
                features = extract_features(face_rgb, face_ir)

                # Recognize face
                label = svm_model.predict([features])[0]
                name = label_encoder.inverse_transform([label])[0]

                # Draw rectangle and label on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Convert the frame to a format suitable for Pygame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)

        # Display the frame in the Pygame window
        screen.blit(frame, (0, 0))
        pygame.display.update()
        
        frame_count += 1  # Increment frame count

# Create a thread for video capture
video_thread = threading.Thread(target=capture_video)
video_thread.start()

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            video_thread.join()  # Wait for the video thread to finish
            exit()
'''
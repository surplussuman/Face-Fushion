'''import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.neural_network import MLPClassifier
import joblib

# Load the MTCNN model for face detection
mtcnn = MTCNN(keep_all=False)

# Load pre-trained FaceNet models
facenet_rgb = InceptionResnetV1(pretrained='vggface2').eval()
facenet_ir = InceptionResnetV1(pretrained='casia-webface').eval()

# Load the trained classifier
clf = joblib.load('face_fusion_model.pkl')

# Function to get embeddings from an image
def get_embeddings(img_rgb, img_ir):
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (160, 160))
    img_rgb = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    img_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)
    img_ir = cv2.resize(img_ir, (160, 160))
    img_ir = torch.tensor(img_ir).unsqueeze(0).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        embedding_rgb = facenet_rgb(img_rgb).numpy().flatten()
        embedding_ir = facenet_ir(img_ir).numpy().flatten()

    return np.concatenate((embedding_rgb, embedding_ir))

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face_rgb = frame[y1:y2, x1:x2]
            face_ir = cv2.applyColorMap(cv2.cvtColor(face_rgb, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)  # Simulated IR

            if face_rgb.size == 0 or face_ir.size == 0:
                continue

            # Get embeddings
            embedding = get_embeddings(face_rgb, face_ir)

            # Predict label
            label = clf.predict([embedding])[0]
            confidence = clf.predict_proba([embedding])[0].max() * 100

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} ({confidence:.2f}%)', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''

'''import torch
from facenet_pytorch import InceptionResnetV1
import os
import numpy as np
import cv2
import joblib

# Load pre-trained FaceNet models for RGB and IR
facenet_rgb = InceptionResnetV1(pretrained='vggface2').eval()
facenet_ir = InceptionResnetV1(pretrained='casia-webface').eval()

# Load the trained SVM model and label encoder
svm = joblib.load('face_fusion_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to extract embeddings for a single image
def get_embeddings(rgb_image_path, ir_image_path):
    # Process RGB image
    img_rgb = cv2.imread(rgb_image_path)
    if img_rgb is None:
        print(f"Error reading RGB image: {rgb_image_path}")
        return None
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (160, 160))
    img_rgb = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        embedding_rgb = facenet_rgb(img_rgb).numpy().flatten()

    # Process IR image
    img_ir = cv2.imread(ir_image_path, cv2.IMREAD_GRAYSCALE)
    if img_ir is None:
        print(f"Error reading IR image: {ir_image_path}")
        return None
    img_ir = cv2.resize(img_ir, (160, 160))
    img_ir = cv2.cvtColor(img_ir, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel image
    img_ir = torch.tensor(img_ir).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        embedding_ir = facenet_ir(img_ir).numpy().flatten()

    # Fuse RGB and IR embeddings (concatenate them)
    fused_embedding = np.concatenate((embedding_rgb, embedding_ir))

    return fused_embedding

# Function to recognize a person given RGB and IR image paths
def recognize_person(rgb_image_path, ir_image_path):
    embedding = get_embeddings(rgb_image_path, ir_image_path)
    if embedding is None:
        return None

    # Predict the label using the trained SVM model
    prediction = svm.predict([embedding])
    label = label_encoder.inverse_transform(prediction)

    return label[0]

# Example usage
rgb_image_path = 'preprocessed_rgb/Suman/1.png'
ir_image_path = 'preprocessed_ir/Suman/1.png'

label = recognize_person(rgb_image_path, ir_image_path)
if label:
    print(f"Recognized as: {label}")
else:
    print("Recognition failed")
'''

import torch
from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
import joblib

# Load pre-trained FaceNet models for RGB and IR
facenet_rgb = InceptionResnetV1(pretrained='vggface2').eval()
facenet_ir = InceptionResnetV1(pretrained='casia-webface').eval()

# Load the trained SVM model and label encoder
svm = joblib.load('face_fusion_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to extract embeddings from a face image
def get_face_embedding(img_rgb, img_ir):
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (160, 160))
    img_rgb = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        embedding_rgb = facenet_rgb(img_rgb).numpy().flatten()

    img_ir = cv2.resize(img_ir, (160, 160))
    img_ir = cv2.cvtColor(img_ir, cv2.COLOR_GRAY2RGB)
    img_ir = torch.tensor(img_ir).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        embedding_ir = facenet_ir(img_ir).numpy().flatten()

    # Fuse RGB and IR embeddings
    fused_embedding = np.concatenate((embedding_rgb, embedding_ir))

    return fused_embedding

# Function to recognize person from embeddings
def recognize_person_from_embedding(embedding):
    prediction = svm.predict([embedding])
    label = label_encoder.inverse_transform(prediction)
    return label[0]

# Initialize video capture (0 for default camera)
cap = cv2.VideoCapture(0)

# Face detection using OpenCV's pre-trained Haar cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for IR processing (simulating IR in real-time)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face regions
        face_rgb = frame[y:y+h, x:x+w]
        face_ir = gray[y:y+h, x:x+w]

        # Get the face embedding
        embedding = get_face_embedding(face_rgb, face_ir)

        # Recognize the person
        label = recognize_person_from_embedding(embedding)

        # Draw the face bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with bounding box and label
    cv2.imshow('Real-Time Face Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

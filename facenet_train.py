import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import os
import pickle
import face_recognition

# Initialize the FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Path to the dataset
dataset_path = 'dataset'

# Initialize lists to hold face embeddings and corresponding labels
embeddings = []
labels = []

# Function to get face embeddings and labels from the dataset
def get_embeddings_and_labels(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                label = os.path.basename(root)
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_img)
                for face_location in face_locations:
                    top, right, bottom, left = face_location
                    face = rgb_img[top:bottom, left:right]
                    face = cv2.resize(face, (160, 160))
                    face = np.transpose(face, (2, 0, 1))
                    face = torch.tensor(face, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        embedding = model(face).numpy().flatten()
                    embeddings.append(embedding)
                    labels.append(label)
    return embeddings, labels

# Load the embeddings and labels
embeddings, labels = get_embeddings_and_labels(dataset_path)

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Train an SVM classifier
classifier = SVC(kernel='linear', probability=True)
classifier.fit(embeddings, labels)

# Save the trained model and label encoder
with open('models/facenet_recognition_model.pkl', 'wb') as f:
    pickle.dump((classifier, label_encoder), f)

print("Model trained and saved as 'face_recognition_model.pkl'")

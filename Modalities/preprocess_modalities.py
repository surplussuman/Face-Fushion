import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import face_recognition
import os

# Initialize the FaceNet model for RGB images
facenet_rgb = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize another model for IR images (you can use the same model or another pretrained model)
facenet_ir = InceptionResnetV1(pretrained='vggface2').eval()

# Path to the dataset
dataset_path_rgb = 'dataset_rgb'
dataset_path_ir = 'dataset_ir'

# Initialize lists to hold face embeddings and corresponding labels
embeddings_rgb = []
embeddings_ir = []
labels = []

# Function to get face embeddings and labels from the dataset
def get_embeddings_and_labels(path_rgb, path_ir):
    for root, dirs, files in os.walk(path_rgb):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                label = os.path.basename(root)
                img_path_rgb = os.path.join(root, file)
                img_path_ir = os.path.join(path_ir, label, file)
                
                # Load and preprocess RGB image
                img_rgb = cv2.imread(img_path_rgb)
                rgb_img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                face_locations_rgb = face_recognition.face_locations(rgb_img)
                
                # Load and preprocess IR image
                img_ir = cv2.imread(img_path_ir, cv2.IMREAD_GRAYSCALE)
                face_locations_ir = face_recognition.face_locations(img_ir)
                
                if face_locations_rgb and face_locations_ir:
                    for face_location_rgb, face_location_ir in zip(face_locations_rgb, face_locations_ir):
                        top, right, bottom, left = face_location_rgb
                        face_rgb = rgb_img[top:bottom, left:right]
                        face_rgb = cv2.resize(face_rgb, (160, 160))
                        face_rgb = np.transpose(face_rgb, (2, 0, 1))
                        face_rgb = torch.tensor(face_rgb, dtype=torch.float32).unsqueeze(0)
                        
                        top, right, bottom, left = face_location_ir
                        face_ir = img_ir[top:bottom, left:right]
                        face_ir = cv2.resize(face_ir, (160, 160))
                        face_ir = np.expand_dims(face_ir, axis=0)
                        face_ir = np.expand_dims(face_ir, axis=0)
                        face_ir = torch.tensor(face_ir, dtype=torch.float32).unsqueeze(0)
                        
                        with torch.no_grad():
                            embedding_rgb = facenet_rgb(face_rgb).numpy().flatten()
                            embedding_ir = facenet_ir(face_ir).numpy().flatten()
                        
                        embeddings_rgb.append(embedding_rgb)
                        embeddings_ir.append(embedding_ir)
                        labels.append(label)
    return embeddings_rgb, embeddings_ir, labels

# Load the embeddings and labels
embeddings_rgb, embeddings_ir, labels = get_embeddings_and_labels(dataset_path_rgb, dataset_path_ir)

# Concatenate RGB and IR embeddings
embeddings = [np.concatenate((e_rgb, e_ir)) for e_rgb, e_ir in zip(embeddings_rgb, embeddings_ir)]

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Train an SVM classifier
classifier = SVC(kernel='linear', probability=True)
classifier.fit(embeddings, labels)

# Save the trained model and label encoder
with open('models/multimodal_face_recognition_model.pkl', 'wb') as f:
    pickle.dump((classifier, label_encoder), f)

print("Multimodal model trained and saved as 'multimodal_face_recognition_model.pkl'")

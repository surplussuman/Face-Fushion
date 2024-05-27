import face_recognition
import cv2
import os
import numpy as np
import pickle

# Path to the dataset
dataset_path = 'dataset'

# Initialize lists to hold face encodings and corresponding labels
encodings = []
labels = []

# Function to read images and labels from the dataset
def get_encodings_and_labels(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                label = os.path.basename(root)
                img_path = os.path.join(root, file)
                img = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(img)
                face_encodings = face_recognition.face_encodings(img, face_locations)
                for encoding in face_encodings:
                    encodings.append(encoding)
                    labels.append(label)
    return encodings, labels

# Load the images and labels
encodings, labels = get_encodings_and_labels(dataset_path)

# Save the encodings and labels
data = {"encodings": encodings, "labels": labels}
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Model trained and saved as 'trained_model.pkl'")

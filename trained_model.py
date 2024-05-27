import cv2
import os
import numpy as np

# Path to the dataset
dataset_path = 'dataset'

# Initialize the face detector and face recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialize lists to hold face samples and corresponding labels
faces = []
labels = []

# Function to read images and labels from the dataset
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    for image_path in image_paths:
        if os.path.isdir(image_path):
            label = os.path.basename(image_path)
            for img_name in os.listdir(image_path):
                img_path = os.path.join(image_path, img_name)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces_detected:
                    faces.append(gray[y:y+h, x:x+w])
                    labels.append(int(label))
    return faces, labels

# Load the images and labels
faces, labels = get_images_and_labels(dataset_path)

# Train the face recognizer
recognizer.train(faces, np.array(labels))

# Save the trained model
recognizer.save('trained_model.yml')

print("Model trained and saved as 'trained_model.yml'")

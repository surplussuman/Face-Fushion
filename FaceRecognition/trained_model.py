import face_recognition
import cv2
import os
import numpy as np
import pickle

# Path to the dataset
dataset_path = 'dataset/'

# Initialize lists to hold face encodings and corresponding labels
encodings = []
labels = []

# import os
print(os.getcwd())


print("Started encoding")
# Function to read images and labels from the dataset
def get_encodings_and_labels(path):
    print("Into function")
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                label = os.path.basename(root)
                img_path = os.path.join(root, file)
                img = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(img)
                face_encodings = face_recognition.face_encodings(img, face_locations)

                print(f"Processing {img_path}")
                # print("Face locations:", face_locations)
                # print("Face encodings:", face_encodings)

                for encoding in face_encodings:
                    encodings.append(encoding)
                    labels.append(label)
    return encodings, labels


print("Done encodeing")
# Load the images and labels
encodings, labels = get_encodings_and_labels(dataset_path)

print("Writing the model")
# Save the encodings and labels
data = {"encodings": encodings, "labels": labels}
with open('models/trained_model_cnn.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Model trained and saved as 'trained_model.pkl'")

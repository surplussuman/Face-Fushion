'''import cv2
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
'''
'''import os
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face

# Load the pre-trained face detection and recognition models
mtcnn = MTCNN(keep_all=True)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Path to the folder containing RGB images
rgb_folder_path = 'dataset_rgb'
ir_folder_path = 'dataset_ir'

# Function to process a folder of images
def process_images(rgb_folder_path, ir_folder_path):
    for label_folder in os.listdir(rgb_folder_path):
        for root, _, files in os.walk(os.path.join(rgb_folder_path, label_folder)):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Load the RGB image
                        rgb_image_path = os.path.join(root, file)
                        rgb_image = cv2.imread(rgb_image_path)
                        if rgb_image is None:
                            print(f'Failed to load RGB image: {rgb_image_path}')
                            continue

                        # Load the IR image
                        ir_image_path = os.path.join(ir_folder_path, label_folder, file)
                        ir_image = cv2.imread(ir_image_path, cv2.IMREAD_GRAYSCALE)
                        if ir_image is None:
                            print(f'Failed to load IR image: {ir_image_path}')
                            continue

                        # Detect faces in the images
                        faces_rgb = mtcnn(rgb_image)
                        faces_ir = mtcnn(ir_image)

                        # Check if any faces were detected
                        if faces_rgb is None or len(faces_rgb) == 0 or faces_ir is None or len(faces_ir) == 0:
                            print(f'No faces detected in {file}')
                            continue

                        # Ensure the faces have the correct dimensions
                        faces_rgb = faces_rgb.permute(0, 3, 1, 2) if faces_rgb.ndim == 4 else faces_rgb.unsqueeze(0)
                        faces_ir = faces_ir.unsqueeze(0).unsqueeze(0)

                        # Get the embeddings for the RGB and IR faces
                        embedding_rgb = facenet(faces_rgb.float()).detach().numpy().flatten()
                        embedding_ir = facenet(faces_ir.float()).detach().numpy().flatten()

                        # Process the embeddings (e.g., save to a file, perform matching, etc.)
                        print(f'Processed {file}: RGB Embedding Shape: {embedding_rgb.shape}, IR Embedding Shape: {embedding_ir.shape}')

                    except Exception as e:
                        print(f'Error processing {file}: {e}')

# Process the images in the folders
process_images(rgb_folder_path, ir_folder_path)
'''
'''import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN, extract_face

# Load the pre-trained face detection and recognition models
mtcnn = MTCNN(keep_all=True)
facenet_rgb = InceptionResnetV1(pretrained='vggface2').eval()
facenet_ir = InceptionResnetV1(pretrained='casia-webface').eval()

# Path to the folder containing RGB images
rgb_folder = 'dataset_rgb'

# Path to the folder containing IR images
ir_folder = 'dataset_ir'

# Function to process a single image
def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Detect faces in the image
    faces = mtcnn(image)
    
    if faces is not None:
        # Extract the first face
        face = extract_face(image, faces[0])
        
        # Preprocess the face for the RGB model
        face_rgb = cv2.resize(face, (160, 160))
        face_rgb = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0).float()
        
        # Preprocess the face for the IR model
        face_ir = cv2.resize(face, (160, 160))
        face_ir = torch.from_numpy(face_ir).unsqueeze(0).unsqueeze(0).float()
        
        # Get the embeddings for the RGB and IR faces
        embedding_rgb = facenet_rgb(face_rgb).detach().numpy().flatten()
        embedding_ir = facenet_ir(face_ir).detach().numpy().flatten()
        
        return embedding_rgb, embedding_ir
    else:
        return None, None

# Function to process all images in a folder
def process_folder(folder):
    embeddings_rgb = []
    embeddings_ir = []
    labels = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder):
        # Process only image files
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Get the full path to the image
            image_path = os.path.join(folder, filename)
            
            # Process the image
            embedding_rgb, embedding_ir = process_image(image_path)
            
            # If embeddings are obtained, append them to the list
            if embedding_rgb is not None and embedding_ir is not None:
                embeddings_rgb.append(embedding_rgb)
                embeddings_ir.append(embedding_ir)
                labels.append(filename.split('.')[0])  # Extract label from filename
    
    return embeddings_rgb, embeddings_ir, labels

# Process the RGB folder
embeddings_rgb, embeddings_ir, labels = process_folder(rgb_folder)

# Process the IR folder
embeddings_rgb_ir, embeddings_ir_ir, labels_ir = process_folder(ir_folder)

# Concatenate the embeddings and labels
embeddings_rgb.extend(embeddings_rgb_ir)
embeddings_ir.extend(embeddings_ir_ir)
labels.extend(labels_ir)

# Convert lists to arrays
embeddings_rgb = np.array(embeddings_rgb)
embeddings_ir = np.array(embeddings_ir)
labels = np.array(labels)

# Print the shapes of the arrays
print(embeddings_rgb.shape, embeddings_ir.shape, labels.shape)
'''

import os
import cv2
from facenet_pytorch import MTCNN

# Load the MTCNN model for face detection
mtcnn = MTCNN(keep_all=False)  # Set keep_all=False to detect a single face

# Function to preprocess images
def preprocess_images(input_folder, output_folder, max_images_per_label=100):
    for label in os.listdir(input_folder):
        label_input_path = os.path.join(input_folder, label)
        label_output_path = os.path.join(output_folder, label)
        os.makedirs(label_output_path, exist_ok=True)

        image_count = 0
        for file in os.listdir(label_input_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(label_input_path, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Detect face
                boxes, _ = mtcnn.detect(img)

                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        face = img[y1:y2, x1:x2]

                        if face.size == 0:  # Check if the cropped face is empty
                            continue

                        # Align and resize the face
                        face = cv2.resize(face, (160, 160))

                        # Save the preprocessed face
                        save_path_img = os.path.join(label_output_path, file)
                        cv2.imwrite(save_path_img, face)

                        image_count += 1
                        if image_count >= max_images_per_label:
                            break

            if image_count >= max_images_per_label:
                break

# Example usage
preprocess_images('dataset_rgb', 'preprocessed_rgb', max_images_per_label=100)
preprocess_images('dataset_ir', 'preprocessed_ir', max_images_per_label=100)

'''import torch
from facenet_pytorch import InceptionResnetV1
import os
import numpy as np
import cv2

# Load pre-trained FaceNet models for RGB and IR
facenet_rgb = InceptionResnetV1(pretrained='vggface2').eval()
facenet_ir = InceptionResnetV1(pretrained='casia-webface').eval()

# Function to extract embeddings
def get_embeddings_and_labels(rgb_folder, ir_folder):
    embeddings_rgb = []
    embeddings_ir = []
    labels = []

    for label in os.listdir(rgb_folder):
        rgb_label_path = os.path.join(rgb_folder, label)
        ir_label_path = os.path.join(ir_folder, label)

        if not os.path.exists(ir_label_path):
            continue

        for file in os.listdir(rgb_label_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                # Process RGB image
                img_path_rgb = os.path.join(rgb_label_path, file)
                img_rgb = cv2.imread(img_path_rgb)
                if img_rgb is None:
                    print(f"Error reading RGB image: {img_path_rgb}")
                    continue
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                img_rgb = cv2.resize(img_rgb, (160, 160))
                img_rgb = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    embedding_rgb = facenet_rgb(img_rgb).numpy().flatten()

                # Process IR image
                img_path_ir = os.path.join(ir_label_path, file)
                img_ir = cv2.imread(img_path_ir, cv2.IMREAD_GRAYSCALE)
                if img_ir is None:
                    print(f"Error reading IR image: {img_path_ir}")
                    continue
                img_ir = cv2.resize(img_ir, (160, 160))
                img_ir = cv2.cvtColor(img_ir, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel image
                img_ir = torch.tensor(img_ir).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    embedding_ir = facenet_ir(img_ir).numpy().flatten()

                embeddings_rgb.append(embedding_rgb)
                embeddings_ir.append(embedding_ir)
                labels.append(label)

    return np.array(embeddings_rgb), np.array(embeddings_ir), np.array(labels)

# Example usage
dataset_path_rgb = 'preprocessed_rgb'
dataset_path_ir = 'preprocessed_ir'

embeddings_rgb, embeddings_ir, labels = get_embeddings_and_labels(dataset_path_rgb, dataset_path_ir)

# Save embeddings and labels
np.save('embeddings_rgb.npy', embeddings_rgb)
np.save('embeddings_ir.npy', embeddings_ir)
np.save('labels.npy', labels)'''


import torch
from facenet_pytorch import InceptionResnetV1
import os
import numpy as np
import cv2

# Load pre-trained FaceNet models for RGB and IR
facenet_rgb = InceptionResnetV1(pretrained='vggface2').eval()
facenet_ir = InceptionResnetV1(pretrained='casia-webface').eval()

# Function to extract embeddings
def get_embeddings_and_labels(rgb_folder, ir_folder):
    embeddings_rgb = []
    embeddings_ir = []
    labels = []

    for label in os.listdir(rgb_folder):
        rgb_label_path = os.path.join(rgb_folder, label)
        ir_label_path = os.path.join(ir_folder, label)

        if not os.path.exists(ir_label_path):
            continue

        for file in os.listdir(rgb_label_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                # Process RGB image
                img_path_rgb = os.path.join(rgb_label_path, file)
                img_rgb = cv2.imread(img_path_rgb)
                if img_rgb is None:
                    print(f"Error reading RGB image: {img_path_rgb}")
                    continue
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                img_rgb = cv2.resize(img_rgb, (160, 160))
                img_rgb = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    embedding_rgb = facenet_rgb(img_rgb).numpy().flatten()

                # Process IR image
                img_path_ir = os.path.join(ir_label_path, file)
                img_ir = cv2.imread(img_path_ir, cv2.IMREAD_GRAYSCALE)
                if img_ir is None:
                    print(f"Error reading IR image: {img_path_ir}")
                    continue
                img_ir = cv2.resize(img_ir, (160, 160))
                img_ir = cv2.cvtColor(img_ir, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel image
                img_ir = torch.tensor(img_ir).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    embedding_ir = facenet_ir(img_ir).numpy().flatten()

                embeddings_rgb.append(embedding_rgb)
                embeddings_ir.append(embedding_ir)
                labels.append(label)

    return np.array(embeddings_rgb), np.array(embeddings_ir), np.array(labels)

# Example usage
dataset_path_rgb = 'preprocessed_rgb'
dataset_path_ir = 'preprocessed_ir'

embeddings_rgb, embeddings_ir, labels = get_embeddings_and_labels(dataset_path_rgb, dataset_path_ir)

# Save embeddings and labels
np.save('embeddings_rgb.npy', embeddings_rgb)
np.save('embeddings_ir.npy', embeddings_ir)
np.save('labels.npy', labels)

'''import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from enhancement import enhance_image

# Load pre-trained FaceNet models for RGB and IR
facenet_rgb = InceptionResnetV1(pretrained='vggface2').eval()
facenet_ir = InceptionResnetV1(pretrained='casia-webface').eval()

print("Started..")
def get_embeddings_and_labels(rgb_folder, ir_folder):
    print("Into function")
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
                enhanced_rgb = enhance_image(img_path_rgb, 'models/EDSR_x4.pb')  # Enhance RGB image
                if enhanced_rgb is None:
                    print(f"Error enhancing RGB image: {img_path_rgb}")
                    continue
                
                img_rgb = cv2.cvtColor(enhanced_rgb, cv2.COLOR_BGR2RGB)
                img_rgb = cv2.resize(img_rgb, (160, 160))
                img_rgb = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    embedding_rgb = facenet_rgb(img_rgb).numpy().flatten()

                # Process IR image
                img_path_ir = os.path.join(ir_label_path, file)
                enhanced_ir = enhance_image(img_path_ir, 'models/EDSR_x4.pb')  # Enhance IR image
                if enhanced_ir is None:
                    print(f"Error enhancing IR image: {img_path_ir}")
                    continue
                
                img_ir = cv2.cvtColor(enhanced_ir, cv2.COLOR_BGR2RGB)  # Convert to 3-channel image
                img_ir = cv2.resize(img_ir, (160, 160))
                img_ir = torch.tensor(img_ir).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    embedding_ir = facenet_ir(img_ir).numpy().flatten()

                embeddings_rgb.append(embedding_rgb)
                embeddings_ir.append(embedding_ir)
                labels.append(label)

                print("Done with functions")

    return np.array(embeddings_rgb), np.array(embeddings_ir), np.array(labels)

# Example usage
dataset_path_rgb = 'data/preprocessed_rgb'
dataset_path_ir = 'data/preprocessed_ir'

print("Calling Function..")
embeddings_rgb, embeddings_ir, labels = get_embeddings_and_labels(dataset_path_rgb, dataset_path_ir)

# Save embeddings and labels
print("Saving...")
np.save('embeddings/embeddings_rgb.npy', embeddings_rgb)
np.save('embeddings/embeddings_ir.npy', embeddings_ir)
np.save('embeddings/labels.npy', labels)
'''

import torch
from facenet_pytorch import InceptionResnetV1
import os
import numpy as np
import cv2
import datetime

# Load pre-trained FaceNet models for RGB and IR
facenet_rgb = InceptionResnetV1(pretrained='vggface2').eval()
facenet_ir = InceptionResnetV1(pretrained='casia-webface').eval()

now = datetime.datetime.now()

print("Started...")
print(now.time())

# Function to extract embeddings in batches
def get_embeddings_and_labels(rgb_folder, ir_folder, batch_size=32):
    print("Into the function")
    embeddings_rgb = []
    embeddings_ir = []
    labels = []

    # Preparing data in batches
    rgb_images = []
    ir_images = []
    label_list = []
    i=0

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
                rgb_images.append(img_rgb)
                
                # Process IR image
                img_path_ir = os.path.join(ir_label_path, file)
                img_ir = cv2.imread(img_path_ir, cv2.IMREAD_GRAYSCALE)
                if img_ir is None:
                    print(f"Error reading IR image: {img_path_ir}")
                    continue
                img_ir = cv2.resize(img_ir, (160, 160))
                img_ir = cv2.cvtColor(img_ir, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel image
                ir_images.append(img_ir)
                
                label_list.append(label)
                
                if len(rgb_images) >= batch_size:
                    # Convert to tensor and normalize
                    rgb_tensor = torch.tensor(np.array(rgb_images)).permute(0, 3, 1, 2).float() / 255.0
                    ir_tensor = torch.tensor(np.array(ir_images)).permute(0, 3, 1, 2).float() / 255.0

                    # Extract embeddings in batches
                    with torch.no_grad():
                        embedding_rgb_batch = facenet_rgb(rgb_tensor).numpy()
                        embedding_ir_batch = facenet_ir(ir_tensor).numpy()
                    
                    embeddings_rgb.extend(embedding_rgb_batch)
                    embeddings_ir.extend(embedding_ir_batch)
                    labels.extend(label_list)
                    
                    # Clear batch lists
                    rgb_images = []
                    ir_images = []
                    label_list = []
                    print(i,"Done")
                    i+=1

    return np.array(embeddings_rgb), np.array(embeddings_ir), np.array(labels)

print("Saving...")
# Example usage
dataset_path_rgb = 'data/dataset_rgb_agum'
dataset_path_ir = 'data/dataset_ir_agum'

embeddings_rgb, embeddings_ir, labels = get_embeddings_and_labels(dataset_path_rgb, dataset_path_ir)

# Save embeddings and labels
np.save('embeddings/embeddings_rgb.npy', embeddings_rgb)
np.save('embeddings/embeddings_ir.npy', embeddings_ir)
np.save('embeddings/labels.npy', labels)

print(f'Saved {datetime.datetime.now()}')
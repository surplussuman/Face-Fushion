import cv2
import imgaug.augmenters as iaa
import os

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def augment_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # Horizontal flip
        iaa.Crop(percent=(0, 0.1)),  # Random crops
        iaa.LinearContrast((0.75, 1.5)),  # Improve or worsen the contrast
        iaa.Multiply((0.8, 1.2)),  # Change brightness
        iaa.Affine(
            rotate=(-25, 25),  # Rotate
            scale=(0.8, 1.2)  # Scale
        )
    ])

    for label in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label)
        output_label_path = os.path.join(output_dir, label)
        
        if os.path.exists(output_label_path):
            break

        else:

            if not os.path.exists(output_label_path):
                os.makedirs(output_label_path)
        
            for filename in os.listdir(label_path):
                print(f'Doing {label}')
                image = cv2.imread(os.path.join(label_path, filename))
                image = preprocess_image(image)
                images_aug = seq(images=[image] * 5)  # Generate 5 augmented images
                for i, img in enumerate(images_aug):
                    cv2.imwrite(os.path.join(output_label_path, f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"), img)

input_dir_rgb = "data/dataset_rgb"
output_dir_rgb = "data/dataset_rgb_agum"
input_dir_ir = 'data/dataset_ir'
output_dir_ir = "data/dataset_ir_agum"


augment_images(input_dir_rgb, output_dir_rgb)
augment_images(input_dir_ir, output_dir_ir)


# Data augmentation while collecting images 

'''
import cv2
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import imgaug.augmenters as iaa

# Paths to your dataset
rgb_save_path = 'data/dataset_rgb'
ir_save_path = 'data/dataset_ir'

# Ensure directories exist
os.makedirs(rgb_save_path, exist_ok=True)
os.makedirs(ir_save_path, exist_ok=True)

# Data augmentation function
def augment_image(image):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # Horizontal flip
        iaa.Crop(percent=(0, 0.1)),  # Random crops
        iaa.LinearContrast((0.75, 1.5)),  # Improve or worsen the contrast
        iaa.Multiply((0.8, 1.2)),  # Change brightness
        iaa.Affine(
            rotate=(-25, 25),  # Rotate
            scale=(0.8, 1.2)  # Scale
        )
    ])
    image_aug = seq(image=image * 5)
    return image_aug

# Capture images and augment them
def capture_and_save_images(label, num_images=100):
    cap = cv2.VideoCapture(0)
    count = 0

    label_rgb_path = os.path.join(rgb_save_path, label)
    label_ir_path = os.path.join(ir_save_path, label)

    os.makedirs(label_rgb_path, exist_ok=True)
    os.makedirs(label_ir_path, exist_ok=True)

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            continue

        # Detect face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_rgb = cv2.resize(face, (160, 160))

            # Convert to PIL Image for augmentation
            face_rgb = augment_image(face_rgb)

            # Convert RGB to simulated IR (grayscale)
            face_ir = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2GRAY)
            face_ir = cv2.cvtColor(face_ir, cv2.COLOR_GRAY2RGB)

            # Apply augmentation to IR image
            face_ir = augment_image(face_ir)

            # Save images
            cv2.imwrite(os.path.join(label_rgb_path, f'{count}.png'), face_rgb)
            cv2.imwrite(os.path.join(label_ir_path, f'{count}.png'), face_ir)

            count += 1
            if count >= num_images:
                break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
label = 'demo'
capture_and_save_images(label, num_images=100)

'''
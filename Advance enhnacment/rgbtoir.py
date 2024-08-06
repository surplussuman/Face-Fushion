import cv2
import os
import numpy as np
from PIL import Image
from torchvision import transforms

# Paths to your dataset
rgb_save_path = 'data/dataset_rgb'
ir_save_path = 'data/dataset_ir'

# Ensure directories exist
os.makedirs(rgb_save_path, exist_ok=True)
os.makedirs(ir_save_path, exist_ok=True)

# Data augmentation function
def augment_image(image):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop((160, 160), scale=(0.8, 1.0)),
    ])
    return transform(image)

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
            face_rgb_pil = Image.fromarray(cv2.cvtColor(face_rgb, cv2.COLOR_BGR2RGB))
            face_rgb_pil = augment_image(face_rgb_pil)
            face_rgb = np.array(face_rgb_pil)

            # Convert RGB to simulated IR (grayscale)
            face_ir = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2GRAY)
            face_ir = cv2.cvtColor(face_ir, cv2.COLOR_GRAY2RGB)

            # Convert to PIL Image for augmentation
            face_ir_pil = Image.fromarray(face_ir)
            face_ir_pil = augment_image(face_ir_pil)
            face_ir = np.array(face_ir_pil)

            # cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.imshow('Face Detection', frame)

            # Save images
            cv2.imwrite(os.path.join(label_rgb_path, f'{count}.png'), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(label_ir_path, f'{count}.png'), cv2.cvtColor(face_ir, cv2.COLOR_RGB2BGR))

            count += 1
            if count >= num_images:
                break

    cap.release()
    cv2.destroyAllWindows()


label = input ("Enter Name: ")
capture_and_save_images(label, num_images=200)

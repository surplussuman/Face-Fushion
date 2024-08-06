import os
import cv2

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def preprocess_and_save_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for label in os.listdir(input_folder):
        label_path = os.path.join(input_folder, label)
        output_label_path = os.path.join(output_folder, label)
        
        if not os.path.exists(output_label_path):
            os.makedirs(output_label_path)
        
        for file in os.listdir(label_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(label_path, file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Error reading image: {img_path}")
                    continue
                preprocessed_img = preprocess_image(img)
                output_path = os.path.join(output_label_path, file)
                cv2.imwrite(output_path, preprocessed_img)

# Define your input and output directories
raw_rgb_folder = 'data/dataset_rgb'
raw_ir_folder = 'data/dataset_ir'
preprocessed_rgb_folder = 'data/preprocessed_rgb'
preprocessed_ir_folder = 'data/preprocessed_ir'

# Preprocess and save images
preprocess_and_save_images(raw_rgb_folder, preprocessed_rgb_folder)
preprocess_and_save_images(raw_ir_folder, preprocessed_ir_folder)

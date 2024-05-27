import cv2
import os

# Define paths to save RGB and simulated IR images
rgb_save_path = 'dataset_rgb'
ir_save_path = 'dataset_ir'

# Ensure the directory exists
os.makedirs(ir_save_path, exist_ok=True)

# Function to convert RGB images to simulated IR images
def convert_rgb_to_ir(label):
    for root, dirs, files in os.walk(os.path.join(rgb_save_path, label)):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                # Read the RGB image
                img_path_rgb = os.path.join(root, file)
                img_rgb = cv2.imread(img_path_rgb)

                # Convert to grayscale
                img_ir = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

                # Optionally, apply filters to enhance features
                img_ir = cv2.GaussianBlur(img_ir, (5, 5), 0)

                # Save the simulated IR image
                ir_filename = os.path.join(ir_save_path, label, file)
                os.makedirs(os.path.dirname(ir_filename), exist_ok=True)
                cv2.imwrite(ir_filename, img_ir)

# Convert RGB images to simulated IR images for the given label
label = input("Enter person to conert to IR : ")
convert_rgb_to_ir(label)

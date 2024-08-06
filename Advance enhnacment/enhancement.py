'''import cv2

# Load super resolution model
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("models/EDSR_x4.pb")  # Example with EDSR model
sr.setModel("edsr", 4)

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(img):
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Histogram equalization for contrast enhancement
    img = cv2.equalizeHist(img)
    
    # Denoising
    img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
    
    return img

def apply_super_resolution(img):
    result = sr.upsample(img)
    return result

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces
def enhance_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image reading was successful
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    # Preprocess image if necessary (e.g., convert to grayscale, enhance contrast, denoise)
    img = preprocess_image(img)
    
    # Apply super-resolution
    img = apply_super_resolution(img)

    # Convert to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img
'''
'''
def enhance_image(image_path, model_path):
    # Load the pre-trained model
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", 4)

    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Convert the image to RGB if it is in grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Enhance the image
    result = sr.upsample(image)
    return result
'''

'''import cv2

# Load super resolution model
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("models/EDSR_x4.pb")  # Example with EDSR model
sr.setModel("edsr", 4)

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def preprocess_image(img):
#     # Convert to grayscale if not already
#     if len(img.shape) == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Histogram equalization for contrast enhancement
#     img = cv2.equalizeHist(img)
    
#     # Denoising
#     img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
    
#     return img

def preprocess_image(img, is_rgb=True):
    if is_rgb:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
    else:
        img_gray = img
    
    # Histogram equalization for grayscale images
    img_gray = cv2.equalizeHist(img_gray)
    
    # Convert back to 3 channels if necessary for super resolution
    if is_rgb:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    # Denoising
    img_denoised = cv2.fastNlMeansDenoising(img_gray, None, 30, 7, 21)
    
    return img_denoised



# def apply_super_resolution(img):
#     result = sr.upsample(img)
#     return result

def apply_super_resolution(img):
    # Ensure the image has 3 channels (required by the super-resolution model)
    if len(img.shape) == 2:  # if grayscale, convert to 3 channels
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result = sr.upsample(img)
    return result


def enhance_image(image_path, model_path):
    # Load the pre-trained model
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", 4)

    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Convert the image to RGB if it is in grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Enhance the image
    result = sr.upsample(image)
    return result

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces
'''

import cv2

# Load super resolution model
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("models/EDSR_x4.pb")  # Example with EDSR model
sr.setModel("edsr", 4)

def preprocess_image(img, is_rgb=True):
    if is_rgb:
        if len(img.shape) == 2:  # If the image is grayscale, convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        if len(img.shape) == 3 and img.shape[2] == 3:  # If the image is RGB, convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histogram equalization for grayscale images
    if len(img.shape) == 2:
        img = cv2.equalizeHist(img)
    
    # Convert back to 3 channels if necessary for super resolution
    if is_rgb and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Denoising
    img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)

    return img

def apply_super_resolution(img):
    if len(img.shape) == 2:  # If grayscale, convert to 3 channels
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result = sr.upsample(img)
    return result

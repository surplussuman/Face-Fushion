import cv2
import numpy as np
from torchvision import transforms
import os
from PIL import Image



def generate_dataset(label):
    
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def augment_image(image):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop((160, 160), scale=(0.8, 1.0)),
        ])
        return transform(image)

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
         
        if faces is ():
            return None
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h,x:x+w]
        return cropped_face
     
    cap = cv2.VideoCapture(0)
    img_id = 0

    path_rgb = 'dataset/'
    label_rgb_path = os.path.join(path_rgb, label)
    os.makedirs(label_rgb_path, exist_ok= True)
     
    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id+=1
            face = cv2.resize(face_cropped(frame), (200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Covert to PIL image 
            face_rgb_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_rgb_pil = augment_image(face_rgb_pil)
            face_rgb = np.array(face_rgb_pil)

            file_name_path = os.path.join(label_rgb_path, f'{img_id}.png')
            # file_name_path = "data/dataset_rgb"+(label)+str(img_id)+".jpg"
            # file_name_path = "Images for visualization/"+str(img_id)+'.jpg'
            cv2.imwrite(file_name_path, face_rgb)
            cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2 )
             
            cv2.imshow("Cropped_Face", face)
            if cv2.waitKey(1)==13 or int(img_id)==400:
                break
                 
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed !!!")

label = input("Enter name: ")
generate_dataset(label)
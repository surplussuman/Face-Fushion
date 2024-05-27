import cv2
import os
import imutils

# Initialize the webcam
#cap = cv2.VideoCapture(0)

# Create a directory to save the images
save_dir = 'dataset_rgb'

# Maximum number of images to capture for each person
count = 0

# Wait for the user to enter the name or label (press any key to continue without entering)
#label = input('Enter name or label (press Enter to skip): ')

def create_dataset(label):
    id = label
    # if not os.path.exists('dataset_rgb/{}/'.format(id)):
    #     os.makedirs('dataset_rgb/{}/'.format(id))
    # directory = 'dataset_rgb/{}/'.format(id)

    print("[INFO] Initializing Video stream")
    cap = cv2.VideoCapture(0)
    sampleNum = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use a face detection algorithm to detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Crop the face region
            face = frame[y:y+h, x:x+w]


            # cv2.imwrite(directory + '/' + str(sampleNum) + '.jpg', face)
            # face_aligned = imutils.resize(face, width=400)


            # Display instructions to enter the name or label of the person
            cv2.putText(frame, 'Enter name or label:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Face Detection', frame)


            if label:
                # Create a directory for the person if it doesn't exist
                person_dir = os.path.join(save_dir, label)
                os.makedirs(person_dir, exist_ok=True)

                sampleNum = sampleNum+1

                # Save the cropped face image
                filename = os.path.join(person_dir, f'{len(os.listdir(person_dir))}.png')
                cv2.imwrite(filename, face)

                print(f'Image {sampleNum} saved!!!')


        # Break the loop if 'q' is pressed
        if sampleNum > 100:
            break


    # Release the webcam and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

label = input("Enter name: ")
create_dataset(label)

'''import cv2

cap = cv2.VideoCapture(3)  # Try 0, then 1, 2, etc.

if not cap.isOpened():
    print("Camera not found")
else:
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
'''
import cv2

# index = 0
# while True:
#     cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
#     if not cap.read()[0]:
#         break
#     else:
#         print(f"Camera index {index} is available.")
#     cap.release()
#     index += 1


for i in range(3, 10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Testing camera index: {i}")
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:  # Ensure the frame is valid
            print(f"Valid frame captured from index: {i}")
            print(f"Frame dimensions: {frame.shape}")
            cv2.imshow(f"Camera {i}", frame)

            cv2.waitKey(5000)

            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()
        else:
            print(f"No valid frame from camera index: {i}")
        cap.release()


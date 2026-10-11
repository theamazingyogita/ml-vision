
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")

print("Press SPACE to capture an image. Press Q to quit.")

captured_image = None

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read from webcam")
        break

    cv2.imshow("Webcam - Press SPACE to capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        captured_image = frame.copy()
        break
    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

if captured_image is not None:
    print("Captured image shape:", captured_image.shape)
    cv2.imshow("Captured Image", captured_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

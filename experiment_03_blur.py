
import cv2

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Could not open webcam")

captured_image = None
print("Press SPACE to capture; Q to quit.")

while True:
    success, frame = camera.read()
    if not success:
        break
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        captured_image = frame.copy()
        break
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

if captured_image is None:
    raise SystemExit("No image captured.")

gray = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

blur_3 = cv2.GaussianBlur(gray, (3, 3), 0)
blur_7 = cv2.GaussianBlur(gray, (7, 7), 0)
blur_15 = cv2.GaussianBlur(gray, (15, 15), 0)

cv2.imshow("01 - Grayscale", gray)
cv2.imshow("02 - Blur 3x3", blur_3)
cv2.imshow("03 - Blur 7x7", blur_7)
cv2.imshow("04 - Blur 15x15", blur_15)

cv2.waitKey(0)
cv2.destroyAllWindows()

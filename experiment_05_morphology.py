
import cv2
import numpy as np

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
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
_, binary = cv2.threshold(
    blurred, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
kernel = np.ones((5, 5), np.uint8)

opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
eroded = cv2.erode(binary, kernel, iterations=1)
dilated = cv2.dilate(binary, kernel, iterations=1)

cv2.imshow("01 - Binary", binary)
cv2.imshow("02 - Opening", opened)
cv2.imshow("03 - Closing", closed)
cv2.imshow("04 - Erosion", eroded)
cv2.imshow("05 - Dilation", dilated)

cv2.waitKey(0)
cv2.destroyAllWindows()

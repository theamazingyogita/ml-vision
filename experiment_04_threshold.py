
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
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

_, threshold_100 = cv2.threshold(
    blurred, 100, 255, cv2.THRESH_BINARY
)
_, threshold_180 = cv2.threshold(
    blurred, 180, 255, cv2.THRESH_BINARY
)
otsu_value, threshold_otsu = cv2.threshold(
    blurred, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print("Otsu's automatically selected threshold:", otsu_value)

cv2.imshow("01 - Blurred grayscale", blurred)
cv2.imshow("02 - Threshold 100", threshold_100)
cv2.imshow("03 - Threshold 180", threshold_180)
cv2.imshow("04 - Otsu threshold", threshold_otsu)

cv2.waitKey(0)
cv2.destroyAllWindows()

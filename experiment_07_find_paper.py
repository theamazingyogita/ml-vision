
import cv2
import numpy as np

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Could not open webcam")

captured_image = None
print("Position the paper, then press SPACE to capture; Q to quit.")

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

kernel = np.ones((15, 15), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(
    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

result = captured_image.copy()
quad = None

if contours:
    largest = max(contours, key=cv2.contourArea)
    frame_area = captured_image.shape[0] * captured_image.shape[1]

    if cv2.contourArea(largest) >= frame_area * 0.10:
        hull = cv2.convexHull(largest)
        perimeter = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)

        if len(approx) == 4:
            quad = approx.reshape(4, 2)

if quad is not None:
    cv2.polylines(result, [quad.astype(np.int32)], True, (0, 255, 0), 3)

    for index, point in enumerate(quad.astype(int)):
        cv2.circle(result, tuple(point), 7, (0, 0, 255), -1)
        cv2.putText(
            result, str(index), tuple(point + [8, -8]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
        )
    print("Detected four corners:\n", quad)
else:
    print("Paper not detected. Try a contrasting background and better lighting.")

cv2.imshow("Thresholded image", binary)
cv2.imshow("Paper detection", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

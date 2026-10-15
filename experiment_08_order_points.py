
import cv2
import numpy as np

def order_points(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32
    )

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Could not open webcam")

captured_image = None
print("Show the entire paper; SPACE captures, Q quits.")

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
binary = cv2.morphologyEx(
    binary, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)
)

contours, _ = cv2.findContours(
    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

if not contours:
    raise SystemExit("No contours found.")

largest = max(contours, key=cv2.contourArea)
hull = cv2.convexHull(largest)
approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)

if len(approx) != 4:
    raise SystemExit("No four-corner contour found. Improve paper contrast.")

unordered = approx.reshape(4, 2)
ordered = order_points(unordered[[2, 0, 3, 1]])

print("Ordered corners [TL, TR, BR, BL]:\n", ordered)

result = captured_image.copy()
labels = ["TL", "TR", "BR", "BL"]

for label, point in zip(labels, ordered.astype(int)):
    cv2.circle(result, tuple(point), 7, (0, 0, 255), -1)
    cv2.putText(
        result, label, tuple(point + [8, -8]),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

cv2.imshow("Ordered corners", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

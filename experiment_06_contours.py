
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
_, binary = cv2.threshold(
    blurred, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

contours, hierarchy = cv2.findContours(
    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

print("Number of external contours:", len(contours))

result = captured_image.copy()
cv2.drawContours(result, contours, -1, (0, 0, 255), 2)

for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.putText(
        result, f"{area:.0f}", (x, max(y - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
    )

cv2.imshow("Binary image", binary)
cv2.imshow("Contours and areas", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

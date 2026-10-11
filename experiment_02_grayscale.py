
# Here we are using the experiment 1 image captured code and then use this rgb image and grayscale it
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")

captured_image = None

print("Press SPACE to capture. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        break

    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        captured_image = frame.copy()
        break
    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

if captured_image is None:
    raise SystemExit("No image captured.")

# Step 2: Convert the captured image to grayscale
gray_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

# Step 3: Inspect the data
print("Original image shape:", captured_image.shape)
print("Grayscale image shape:", gray_image.shape)
print("Original pixel (BGR):", captured_image[100, 100])
print("Grayscale pixel:", gray_image[100, 100])

# Step 4: Compare the imagesS
cv2.imshow("Original - BGR", captured_image)
cv2.imshow("Grayscale", gray_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

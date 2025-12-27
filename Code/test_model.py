import cv2
from ultralytics import YOLO

# 1. Load your trained model
# Replace 'best.pt' with the path to your downloaded weight file
model = YOLO("../Model/best1.pt")

# 2. Load the image using OpenCV
image_path = "bat.png"
img = cv2.imread(image_path)

# 3. Run Inference
# The model returns a list of Results objects
results = model(img)

# 4. Plot the results
# results[0].plot() creates a numpy array with the bounding boxes drawn on it
annotated_frame = results[0].plot()

# 5. Display the result using OpenCV
cv2.imshow("YOLOv11 Detection", annotated_frame)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
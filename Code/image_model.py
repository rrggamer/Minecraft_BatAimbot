import cv2
from ultralytics import YOLO

# 1. Load your trained model
model = YOLO("../Model/best1.pt")

# 2. Load the image using OpenCV
image_path = "bat2.png"
img = cv2.imread(image_path)

# 3. Run Inference
results = model(img)

# 4. Get the default annotated frame (boxes and labels)
annotated_frame = results[0].plot()

# 5. Calculate and draw center points
for result in results:
    # result.boxes.xyxy contains [x1, y1, x2, y2]
    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = box
        
        # Calculate center
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Draw a solid red circle at the center
        # (image, center_coordinates, radius, color_bgr, thickness)
        cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)

# 6. Display the result
cv2.imshow("YOLOv11 Detection with Centers", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
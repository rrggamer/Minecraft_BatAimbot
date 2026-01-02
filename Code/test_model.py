import cv2
from ultralytics import YOLO
import mss
import numpy as np

monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
model = YOLO("../Model/best1.pt")

with mss.mss() as sct:
    while True:
        img = sct.grab(monitor)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        results = model(frame, conf=0.5, verbose=False)
        
        # 1. Get the annotated frame (optional, if you still want the boxes)
        annotated_frame = results[0].plot()
        
        # 2. Iterate through detected boxes to find centers
        for result in results:
            boxes = result.boxes.xyxy  # Get boxes in [x1, y1, x2, y2] format
            
            for box in boxes:
                # Calculate center coordinates
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # 3. Draw the center point
                # cv2.circle(image, center_coordinates, radius, color, thickness)
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1) 
                
                # Optional: Draw a label for the coordinates
                cv2.putText(annotated_frame, f"{center_x},{center_y}", (center_x + 10, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("OpenCV Screen Capture", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
cv2.destroyAllWindows()
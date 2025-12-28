import cv2
from ultralytics import YOLO
import mss
import time
import numpy as np

monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

model = YOLO("../Model/best1.pt")

with mss.mss() as sct:
    while True:
        last_time = time.time()
        
        img = sct.grab(monitor)
        
        frame = np.array(img)
        
        frame = cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
        
        results = model(frame, conf = 0.5, verbose=False)
        
        annotated_frame = results[0].plot()
        
        cv2.imshow("OpenCV Screen Capture", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
cv2.destroyAllWindows()


# image_path = "bat.png"
# img = cv2.imread(image_path)


# results = model(img)
# annotated_frame = results[0].plot()

# cv2.imshow("YOLOv11 Detection", annotated_frame)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
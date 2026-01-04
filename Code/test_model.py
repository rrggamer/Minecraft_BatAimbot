import cv2
from ultralytics import YOLO
import mss
import numpy as np
import keyboard
from pynput.mouse import Controller

#--------Settings----------

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
CENTER_X  = SCREEN_WIDTH // 2
CENTER_Y = SCREEN_HEIGHT // 2

#   Detection Zone
#   FOV box size
FOV_SIZE = 400 
monitor = {
    "top": CENTER_Y - FOV_SIZE // 2, 
    "left": CENTER_X - FOV_SIZE // 2, 
    "width": FOV_SIZE, 
    "height": FOV_SIZE
}

AIM_SPEED = 0.4

print("Loading YOLO Model.....")
model = YOLO("../Model/best1.pt")
mouse = Controller()

print("Ready! Hold 'p' to track a BAT!!")

with mss.mss() as sct:
    while True:
        # --- Capture & Preprocess ---
        img = sct.grab(monitor)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # --- Inference ---
        results = model(frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        
        # --- Logic: Find Closest Target ---
        closest_dist = float('inf')
        target_offset = None
        
        
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = box
                # --- Find Center Box ---
                box_center_x = int((x1 + x2) / 2)
                box_center_y = int((y1 + y2) / 2)
                
                # --- ABS Screen Coordiantes ---
                abs_x = box_center_x + monitor["left"]
                abs_y = box_center_y + monitor["top"]
                
                # --- Calculate Distance From Crosshair
                dist_x = abs_x - CENTER_X
                dist_y = abs_y - CENTER_Y
                
                # --- Distance From Crosshair
                current_dist = (dist_x ** 2 + dist_y ** 2 ) ** 0.5
                
                # --- Keep only the closest to the crosshair
                if current_dist < closest_dist:
                    closest_dist = current_dist
                    target_offset = (dist_x, dist_y)
                
        # --- Control: Move Mouse ----
        if keyboard.is_pressed('p') and 
                
                
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1) 
                
                # Optional: Draw a label for the coordinates
                cv2.putText(annotated_frame, f"{center_x},{center_y}", (center_x + 10, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("OpenCV Screen Capture", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
cv2.destroyAllWindows()
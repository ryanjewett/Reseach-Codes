import cv2
import numpy as np


def draw_trapezoidal_roi(video_path, points):
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Unable to open the video.")
        return
    
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        height, width = frame.shape[:2]
        print(f"Frame dimensions - Height: {height}, Width: {width}")
        
       
        # Define the four coordinates of the trapezoid
        points = np.array([[    # width x height
            (730,520),          # top left 
            (965,520),         # top right 
            (1400,770),         # bottom right 
            (380,800)           # bottom left 
        ]], dtype=np.float32)
        
        
        cv2.polylines(frame, [points.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=3)
        
       
        cv2.imshow("Trapezoidal ROI", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    cap.release()
    cv2.destroyAllWindows()

video_path = 'MOVA0005.avi'  
draw_trapezoidal_roi(video_path, None)

import numpy as np
import cv2
from ultralytics import YOLO
import os
import time
from PythonCode.lane import run_time

global model_name
model_name = "yolov8s-seg.pt"
model = YOLO(model_name)

#model = YOLOv10('yolov10l.pt')

def object_detection(frame):
    results = model.predict(frame)
    result = results[0]                                                            
    for box in result.boxes:                                                
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        x_min, y_min, x_max, y_max = cords
        if conf > .4:
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            label = f"{class_id}: {conf}"
            cv2.putText(frame, label, (int(x_min), int(y_min) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

def main() -> None:
    input_video = "MOVA0005.avi"
    output_folder = "SAVE_HERE"
    final_video_name = input("Input final video name:")

    if not os.path.isfile(input_video):
        print("Incorrect Video format")
        exit(1)
    
    if not os.path.isdir(output_folder):
        print("Incorrect output path")
        exit(1)

    if not final_video_name.endswith(".mp4"):  
        print("Incorrect name of final video")
        exit(1)
    
    cap = cv2.VideoCapture(input_video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  
    final_output_file = os.path.join(output_folder, final_video_name) 
    out = cv2.VideoWriter(final_output_file, fourcc, fps, (2*width, 2*height))  


    time_per_frame = 0
    global model_name

    while cap.isOpened():
        
        start_time = time.time()

        ret, frame = cap.read()

        if not ret:
            break
        
        lane_detection_frame = run_time(frame.copy())  
        
        object_detection_frame = object_detection(frame.copy()) 
        
        original_frame = frame.copy()  # Top-left quadrant
        #lane_detection_frame_resized = cv2.resize(lane_detection_frame, (width, height))  # Bottom-left quadrant
        #object_detection_frame_resized = cv2.resize(object_detection_frame, (width, height))  # Top-right quadrant
        #combined_frame = cv2.addWeighted(lane_detection_frame_resized, .5, object_detection_frame_resized, .5, 0)  # Bottom-right quadrant
        combined_frame = cv2.addWeighted(lane_detection_frame, .5, object_detection_frame, .5, 0)  # Bottom-right quadrant

        #top_row = np.hstack((original_frame, object_detection_frame_resized))
        #bottom_row = np.hstack((lane_detection_frame_resized, combined_frame))

        top_row = np.hstack((original_frame, object_detection_frame))
        bottom_row = np.hstack((lane_detection_frame, combined_frame))

        end_time = time.time()
        time_per_frame = end_time - start_time

        final_frame = np.vstack((top_row, bottom_row))

        text = f"Model: {model_name} - Time per frame: {time_per_frame:.6f} s"

        final_frame_with_text = cv2.putText(final_frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)


        out.write(final_frame_with_text)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


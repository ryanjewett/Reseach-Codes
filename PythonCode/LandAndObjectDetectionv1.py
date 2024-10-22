import numpy as np
import cv2
from ultralytics import YOLO
import os
import time

global model_name
#model_name = "yolov8m-world.pt"
model_name="yolov10x"
model = YOLO(model_name)

#model = YOLOv10('yolov10l.pt')

def lane_detection(image):
    def calculate_angle(x1, y1, x2, y2):
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        return angle
    
    def is_within_angle_range(angle, min_angle=20, max_angle=160, vertical_min=85, vertical_max=95):
        abs_angle = abs(angle)
        return min_angle <= abs_angle <= max_angle and not (vertical_min <= abs_angle <= vertical_max)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny edge detector
    edges = cv2.Canny(blur, 40, 160)
    
    # Define region of interest (ROI) polygon
    height, width = edges.shape
    shift_down = height // 2
    shift_right = width // 6

    polygon = np.array([[     #define roi
        (shift_right, height - shift_down),          # top left 
        (width - 2* shift_right, height - shift_down),      # top right 

        (width - 2* shift_right, height ),  # bottom right 
        (shift_right, height )       # bottom left 
    ]], np.int32)
    
    # Create a mask for the ROI
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, polygon, 255)
    
    # Apply mask to edges image
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Perform Hough transform to detect lines
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=70,
        maxLineGap=40
    )
    
    # Initialize an image to draw lines on
    line_image = np.zeros_like(image)
    
    # Variables to track leftmost and rightmost lines
    leftmost_line = None
    rightmost_line = None
    
    # Process detected lines
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = calculate_angle(x1, y1, x2, y2)
                if is_within_angle_range(angle):
                    # Draw the line on the line_image
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
                    
                    # Determine if this line is the leftmost or rightmost
                    if leftmost_line is None or x1 < leftmost_line[0]:
                        leftmost_line = (x1, y1, x2, y2)
                    if rightmost_line is None or x2 > rightmost_line[2]:
                        rightmost_line = (x1, y1, x2, y2)
    
    # Check if both leftmost and rightmost lines were found
    if leftmost_line is not None and rightmost_line is not None:
        # Calculate horizontal distance between leftmost and rightmost lines
        distance = rightmost_line[2] - leftmost_line[0]
        
        # Check if distance is more than 100 pixels
        if distance > 100:
            # Shade the area between leftmost_line and rightmost_line in green
            cv2.fillPoly(line_image, np.array([[
                (leftmost_line[0], leftmost_line[1]),
                (leftmost_line[2], leftmost_line[3]),
                (rightmost_line[2], rightmost_line[3]),
                (rightmost_line[0], rightmost_line[1])
            ]]), (0, 255, 0))
    
    # Combine original image with line_image
    lanes_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    
    return lanes_image

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
    input_video = "MOVA0023.avi"
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
        
        lane_detection_frame = lane_detection(frame.copy())  
        
        object_detection_frame = object_detection(frame.copy()) 
        
        original_frame = frame.copy()  # Top-left quadrant
        lane_detection_frame_resized = cv2.resize(lane_detection_frame, (width, height))  # Bottom-left quadrant
        object_detection_frame_resized = cv2.resize(object_detection_frame, (width, height))  # Top-right quadrant
        combined_frame = cv2.addWeighted(lane_detection_frame_resized, .5, object_detection_frame_resized, .5, 0)  # Bottom-right quadrant

        top_row = np.hstack((original_frame, object_detection_frame_resized))
        bottom_row = np.hstack((lane_detection_frame_resized, combined_frame))

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


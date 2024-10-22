
import ultralytics
import cv2
import os
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time 
import pandas as pd

model1 = YOLO("yolov8m.pt")
model2 = YOLO("yolov5mu.pt")

print("Done")

def process_image(type, image_path,df):
    type = type
    if type == 0:
        model = model1
    else:
        model = model2
    image = image_path
    
    start_time = time.time()
    results = model.predict(image_path)
    end_time = time.time()
    total_time = end_time - start_time

    result = results[0]                                                           #takes the strongest results    
    len(result.boxes)
    box = result.boxes[0]

    pictureData = []
    
    for box in result.boxes:                                                 #get each object detected and puts a box around it with the name and confidence 
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        if(conf >= .5):
            x_min, y_min, x_max, y_max = cords
            if(x_max - x_min >= 10 and y_max - y_min > 10):
                cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                label = f"{class_id}: {conf}"
                cv2.putText(image, label, (int(x_min), int(y_min) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                pictureData.append({'time_total': total_time, 'class_id': class_id, 'cords': cords, 'conf': conf})
    
    df = df.append(pictureData, ignore_index=True)

    conf_avrg = df['conf'].mean()
    df.set_index('frame_number', inplace=True)
    
    return(image,total_time,conf_avrg,df)

def plotInfo(total_time, conf_avrg):
    frame_height = height
    frame_width = width
    conf = conf_avrg
    iteration = iteration + 1

    plt.figure(figsize=(frame_width,frame_height))
    plt.plot(iteration, total_time, label='Total Time')
    plt.plot(iteration, conf, label='Confidence')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Metrics over Iterations')
    plt.legend()
    plt.grid(True)
 
    plt.savefig(plot_img)

    return plot_img

def main():
    input_video = input("Insert the video path:")
    output_folder = input("Insert the save folder:")
    named_video_file = input("Name the new video:")
    global named_excel_file
    #named_excel_file = input("Name of excel file:")
    #named_excel_file = named_excel_file + ".xlsx"

    if not os.path.exists(output_folder):                                          #make sure the folder acutally exists
        print("Not correct folder for saving video")
        exit(0)
    if not os.path.exists(input_video):
        print("No image video found!")
        exit(0)
    
    cap = cv2.VideoCapture(input_video)

    global width
    global height
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = width *2
    height = height*2

    named_video_file = named_video_file + ".mp4"                                   #adds .mp4 to end of inputed file name 
    output_folder = os.path.join(output_folder, named_video_file) 
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  
    out = cv2.VideoWriter(output_folder, fourcc, fps, (width, height))

    df1 = pd.DataFrame(columns=['time_total', 'class_id', 'cords', 'conf'])
    df2 = pd.DataFrame(columns=['time_total', 'class_id', 'cords', 'conf'])
    global frame_count
    frame_count = 1
    type1 = 0
    type2 = 1
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit(0)
    
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        new_frame,total_time,conf_avrg1,df1 = process_image(type1,frame,df1)
        new_frame2,total_time2,conf_avrg2,df2 = process_image(type2,frame,df2)

        graph_left = plotInfo(total_time, conf_avrg1)
        graph_right = plotInfo(total_time2, conf_avrg2)

        left_side = np.vstack([graph_left, new_frame])
        right_side = np.vstack([graph_right, new_frame2])
        
        combined_frame = np.hstack([left_side,right_side])
        frame_count = frame_count + 1
        out.write(combined_frame)

    cap.release()
    out.release()
    end_time = time.time()
    #df1.to_excel(named_excel_file)
    #df2.to_excel(named_excel_file)
    total_time3 = start_time - end_time
    print(f"video is saved at {output_folder} Total is : {total_time3} seconds")
    
if __name__ == "__main__":
    main()



#this code takes in a folder that is full of single frames and creates a video using them

import ultralytics
import cv2
import os
import numpy as np
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt

model = YOLO("yolov8s.pt")


print("Done!")


global total_objects
total_objects = 0
def process_image(image_path):
    image = image_path
    results = model.predict(image_path)
    result = results[0]                                                           #takes the strongest results    
    len(result.boxes)
    box = result.boxes[0]
    for box in result.boxes:
        total_objects = total_objects + 1                                                  #get each object detected and puts a box around it with the name and confidence 
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)

        x_min, y_min, x_max, y_max = cords
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        label = f"{class_id}: {conf}"
        cv2.putText(image, label, (int(x_min), int(y_min) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return(image)
    

def main():
    
    input_folder = input("Enter the path to the folder containing images: ")       #get all folder location  
    output_folder = input("Enter the desired save location for the video file: ")
    named_video_file = input("Enter the name of the completed video file: ")


    if not os.path.exists(output_folder):                                          #make sure the folder acutally exists
        print("Not correct file for saving video")
        exit(0)
    if not os.path.exists(input_folder):
        print("No image folder found!")
        exit(0)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]      #save all images to image_files

    
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))           #get the size of the images inputed. ALL MUST BE SAME SIZE
    height, width, _ = first_image.shape
    width = width*2                                                                #width needs to be doubled if using orignal and new image

    named_video_file = named_video_file + ".mp4"                                   #adds .mp4 to end of inputed file name 
    output_file = os.path.join(output_folder, named_video_file)                    #creates the output video file location
    fourcc = cv2.VideoWriter_fourcc(*'avc1')                                       #avc1 is the only codec that I found that worked
    out = cv2.VideoWriter(output_file, fourcc, 5.0, (width, height))               #changed third input to VideoWriter to adjust frame rate

    
    for image_file in image_files:                         
        img_path = os.path.join(input_folder, image_file)                         #gets the image path for the current image in the folder
        original_img = cv2.imread(img_path)                                       #uses cv2.imread to store the image in original_img

        
        processed_img = process_image(original_img)                               #get image with bounding boxes

        new_org_image = cv2.imread(img_path)                                      #had to create new image variable to save orgianl image

        result_img = cv2.hconcat([new_org_image, processed_img])                  #stacks both images side by side

        
        out.write(result_img)                                                     #save the image to the .avi file to make a video

    
    out.release()
    print(f"Video saved at: {output_folder}\nTotal Objects found: {total_objects}")   


if __name__ == "__main__":
    main()



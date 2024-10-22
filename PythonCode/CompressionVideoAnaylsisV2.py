import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
#from moviepy.editor import VideoFileClip
import datetime 


def new_process_image(model,image_path1, image_path2):  #1 is comp
    model = model
    matches_found_org = 0
    matches_found_comp = 0
    matches_missed_org = 0
    matches_missed_comp = 0
    image1 = image_path1
    image2 = image_path2

    results1 = model.predict(image_path1)
    results2 = model.predict(image_path2)

    fresults1 = results1[0]
    fresults2 = results2[0]

    for box1 in fresults1.boxes:
        class_id1 = str(fresults1.names[box1.cls[0].item()])
        cords1 = box1.xyxy[0].tolist()
        conf1 = round(box1.conf[0].item(), 2)
        x_min1, y_min1, x_max1, y_max1 = [int(round(x)) for x in cords1]
        found_match = False
        for box2 in fresults2.boxes:
            class_id2 = str(fresults2.names[box2.cls[0].item()])
            cords2 = box2.xyxy[0].tolist()
            conf2 = round(box2.conf[0].item(), 2)
            x_min2, y_min2, x_max2, y_max2 = [int(round(x)) for x in cords2]
            #if (class_id1 == class_id2) and abs(x_min1 - x_min2) < 10 and abs(y_min1 - y_min2) < 10:
            if abs(x_min1 - x_min2) < 20 and abs(y_min1 - y_min2) < 20 and abs(x_max1-x_max2) < 20 and abs(y_max1-y_max2) <20:
                found_match = True
                matches_found_org +=1
                cv2.rectangle(image1, (x_min1, y_min1), (x_max1, y_max1), (0, 255, 0), 2)
                label = f"{class_id1}: {conf1}"
                cv2.putText(image1, label, (x_min1, y_min1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.rectangle(image2, (x_min2, y_min2), (x_max2, y_max2), (0, 255, 0), 2)
                label = f"{class_id2}: {conf2}"
                cv2.putText(image2, label, (x_min2, y_min2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                break
        if not found_match:
            matches_missed_org += 1
            cv2.rectangle(image1, (x_min1, y_min1), (x_max1, y_max1), (0, 0, 255), 2)  #bgr
            label = f"{class_id1}: {conf1}"
            cv2.putText(image1, label, (x_min1, y_min1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    for box2 in fresults2.boxes:
        class_id2 = str(fresults2.names[box2.cls[0].item()])
        cords2 = box2.xyxy[0].tolist()
        conf2 = round(box2.conf[0].item(), 2)
        x_min2, y_min2, x_max2, y_max2 = [int(round(x)) for x in cords2]
        found_match = False
        for box1 in fresults1.boxes:
            class_id1 = str(fresults1.names[box1.cls[0].item()])
            cords1 = box1.xyxy[0].tolist()
            conf1 = round(box1.conf[0].item(), 2)
            x_min1, y_min1, x_max1, y_max1 = [int(round(x)) for x in cords1]
            #if (class_id1 == class_id2) and abs(x_min1 - x_min2) < 10 and abs(y_min1 - y_min2) < 10:
            #if abs(x_min1 - x_min2) < 10 and abs(y_min1 - y_min2) < 10:
            if abs(x_min1 - x_min2) < 20 and abs(y_min1 - y_min2) < 20 and abs(x_max1-x_max2) < 20 and abs(y_max1-y_max2) <20:
                found_match = True
                matches_found_comp +=1
                break
        if not found_match:
            matches_missed_comp +=1
            cv2.rectangle(image2, (x_min2, y_min2), (x_max2, y_max2), (0, 0, 255), 2)
            label = f"{class_id2}: {conf2}"
            cv2.putText(image2, label, (x_min2, y_min2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    print(f"Matches missed Org : {matches_missed_org} Mathces Found org :{matches_found_org} Matches found Comp:{matches_found_comp} Matches Missed comp:{matches_missed_comp}")
    return(image1,image2,matches_missed_org,matches_found_org,matches_missed_comp,matches_found_comp)

def create_video_from_images(image_folder,video_name,output_folder,fps=25):
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape
    output_path_name = output_folder
    output_path = os.path.join(output_path_name, video_name)
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    #cv2.destroyAllWindows()
    video.release()

def compress_images(input_folder, output_folder, quality):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(input_folder, file_name), cv2.IMREAD_UNCHANGED)
            compressed_img = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
            output_file_path = os.path.join(output_folder, file_name)
            with open(output_file_path, 'wb') as f:
                f.write(compressed_img)

def compress_video(input_file, output_folder, quality,output_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_capture = cv2.VideoCapture(input_file)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    output_file_path = os.path.join(output_folder,output_name)
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        compressed_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
        frame = cv2.imdecode(compressed_frame, cv2.IMREAD_UNCHANGED)
        video_writer.write(frame)

    video_capture.release()
    video_writer.release()

def delete_folder_contents(folder_path):
    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                delete_folder_contents(item_path)
                os.rmdir(item_path)  
        print(f"Folder '{folder_path}' Cleared")    
    except Exception as e:
        print(f"Error occurred: {e}")

def delete_temp_video(folder_path,video_file_name):
    try:
        file_path = os.path.join(folder_path, video_file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File '{video_file_name}' deleted successfully.")
            return True
        else:
            print(f"File '{video_file_name}' does not exist.")
            return False
    except Exception as e:
        print(f"An error occurred while deleting '{video_file_name}': {str(e)}")
        return False

def plot_data(output_folder,compression_quality,matches_found_org,matches_missed_org,matches_found_comp,matches_missed_comp):
   
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)  
    x_values = range(len(matches_found_org))
    plt.plot(x_values, matches_found_org, label='OriginaL Found')
    plt.plot(x_values, matches_found_comp, label='Compressed Found')
    plt.xlabel('Frame Number')
    plt.ylabel('Count')
    plt.title('Matches Found')
    plt.legend()

    plt.subplot(1, 3, 2)  
    x_values = range(len(matches_missed_org))
    plt.plot(x_values, matches_missed_org, label='Orginal Missed')
    plt.plot(x_values, matches_missed_comp, label='Compressed Missed')
    plt.xlabel('Frame Number')
    plt.ylabel('Count')
    plt.title('Objects Missed')
    plt.legend()

    percentage_missed_comp = []
    for found_org, missed_org, missed_comp in zip(matches_found_org, matches_missed_org, matches_missed_comp):
        if found_org + missed_org == 0:
            percentage_missed_comp.append(0)
        else:
            percentage_missed_comp.append(100 * missed_comp / (found_org + missed_org))

    plt.subplot(1, 3, 3) 
    plt.plot(x_values, percentage_missed_comp, label='Percentage Difference')
    plt.xlabel('Frame Number')
    plt.ylabel('Percentage Difference (%)')
    plt.title('Percentage Difference')
    plt.legend()

    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    plot_name = f'CompressionQuality{compression_quality}_{timestamp}.png'
    output_file_path = os.path.join(output_folder, plot_name)
    plt.savefig(output_file_path)
    print(f"Plot saved at: {output_file_path}")
    
# def avi_to_mp4(input_file, output_file):

#     clip = VideoFileClip(input_file)
#     clip.write_videofile(output_file, codec='libx264')
#     clip.close()

def main():
    input_path = input("Select input folder or video: ")  # Path to the input folder or video file
    analyzed_video_name = input("Name Final Video (no .mp4 required): ")
    compression_quality = int(input("Enter the compression quality (0-10): "))
    model = input("What yolo model (v8 only!) or leave empty for default: ")

    if not model:
        model = YOLO("yolov8m.pt")
        print("Model Set to Default")
    elif model.endswith('.pt'):
        model = YOLO(model)
        print(f"Model Set to {model}")
    else:
        print("inocrrect model")
        exit(1)

    if(compression_quality > 10 or compression_quality < 0):
        print("incorrect compression quality")
        exit(1)
    if(analyzed_video_name.endswith('.mp4')):
        print("incorrect final video name")
        exit(1)
    
    mp4_video_name = analyzed_video_name + '.mp4'
    analyzed_video_name = analyzed_video_name + '.avi'
    output_path_compressed_images = "/Users/ryanjewett/Documents/Reseach2024/TEMP_FOLDER"  # Put a temp folder path here
    output_folder = "/Users/ryanjewett/Documents/Reseach2024/SAVE_HERE"                      #final ouput folder path here
    output_name_preprocessed = (f"TempCompressedVideo{compression_quality}.avi")
    output_video_from_input_folder = "TempVideo.mp4"
    mp4_video_path = os.path.join(output_folder,mp4_video_name)

    if os.path.isfile(input_path):
        if input_path.endswith(('.mp4', '.avi', '.mov')):
            orginal_video = input_path
            print("Orginal Video Created")
            compress_video(input_path, output_folder, compression_quality, output_name_preprocessed)
            print("Video Compressed")
        else:
            print("Unsupported file format. Please provide a video file.")
    elif os.path.isdir(input_path):
        create_video_from_images(input_path, output_video_from_input_folder,output_folder)
        print("Original Video Created")
        compress_images(input_path, output_path_compressed_images, compression_quality)
        print("Images Compressed")
        create_video_from_images(output_path_compressed_images,output_name_preprocessed,output_folder)
        print("Video Compressed from Images Created")
        orginal_video = os.path.join(output_folder, output_video_from_input_folder)
    else:
        print("Invalid input path. Please provide a valid file or folder path.")
    
    compressed_video = os.path.join(output_folder,output_name_preprocessed)

    try:
        cap2 = cv2.VideoCapture(orginal_video)
        cap = cv2.VideoCapture(compressed_video)
    except:
        print("Error with opening compressed video or orginal video")
        exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = width *2
    #fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 5
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    final_output_file = os.path.join(output_folder, analyzed_video_name) 
    out = cv2.VideoWriter(final_output_file, fourcc, fps, (width, height))

    matches_found_org = []
    matches_missed_org = []
    matches_found_comp = []
    matches_missed_comp = []
    
    while cap.isOpened and cap2.isOpened():
        ret1, frame = cap.read()
        ret2, frame2 = cap2.read()
        if not (ret1 or ret2):
            break

        processed_frame_org, processed_frame,missed_org,found_org,missed_comp,found_comp = new_process_image(model,frame,frame2)
        matches_found_org.append(found_org)
        matches_missed_org.append(missed_org)
        matches_found_comp.append(found_comp)
        matches_missed_comp.append(missed_comp)
        combined_frame = cv2.hconcat([processed_frame_org, processed_frame])
        out.write(combined_frame)
       
    cap.release()
    cap2.release()
    out.release()
    plot_data(output_folder,compression_quality,matches_found_org,matches_missed_org,matches_found_comp,matches_missed_comp)
    #avi_to_mp4(final_output_file,mp4_video_path)
    delete_folder_contents(output_path_compressed_images)
    delete_temp_video(output_folder, output_name_preprocessed)
    delete_temp_video(output_folder, analyzed_video_name)
    delete_temp_video(output_folder, output_video_from_input_folder)
    print(f"Analyzed video is saved at {analyzed_video_name} saved in {output_folder}")
    #plt.show()
if __name__ == "__main__":
    main()

#C:\Users\ryanj\Documents\2024ResearchV2\metro_west_test_video.mp4

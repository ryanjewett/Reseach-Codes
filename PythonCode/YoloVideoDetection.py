from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO('/Users/ryanjewett/Documents/Reseach2024/runs/detect/train16/weights/firsttrain.pt')


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



def main():
    inputVideo = '/Users/ryanjewett/Documents/Reseach2024/MOVA0023.avi'
    outputFolder = 'SAVE_HERE'
    outputVideoName = input("name video:")

    cap = cv2.VideoCapture(inputVideo)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  
    final_output_file = os.path.join(outputFolder, outputVideoName) 
    out = cv2.VideoWriter(final_output_file, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        detectedFrame = object_detection(frame)
        

        out.write(detectedFrame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()








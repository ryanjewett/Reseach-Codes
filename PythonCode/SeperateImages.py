import cv2
import os

video_path = 'MOVA0023.avi'

output_dir = '/Users/ryanjewett/Documents/Reseach2024/0023Images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

frame_count = 0
upload_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    if frame_count % 30 == 0:
        upload_count+=1
        frame_filename = os.path.join(output_dir, f'frame_{upload_count:04d}.jpg')
    
    cv2.imwrite(frame_filename, frame)
    
    frame_count += 1
    if upload_count > 74:
        break

cap.release()
print(f"Saved {upload_count} frames to {output_dir}")

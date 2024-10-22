from ultralytics import YOLO

# Load the model
model = YOLO('model20best.pt')  # Replace with your YOLOv8 model checkpoint

# Validate the model
results = model.val(data='/Users/ryanjewett/Documents/Reseach2024/validation2.yaml')

# Print results
print(results)
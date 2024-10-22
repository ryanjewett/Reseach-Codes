import cv2
from PythonCode.lane import run_time


# Function to process and display the image
def process_and_display_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Check if the image is read correctly
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return
    
    # Process the image through the run_time function
    result = run_time(image)
    
   
    
    # Display the image using OpenCV
    cv2.imshow('Image', result)
    
    # Wait for a key press and close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to your JPG image
image_path = 'original_lane_detection_5.jpg'

# Run the function
process_and_display_image(image_path)

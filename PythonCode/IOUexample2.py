import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO('yolov8m.pt')


def load_yolo_label(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        labels.append((class_id, x_center, y_center, width, height))
    return labels


def load_image(image_path):
    image = cv2.imread(image_path)
    return image


def model_prediction(image):
    results = model.predict(image)
    result = results[0]
    predictions = []

    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        predictions.append((class_id, cords, conf))

    return predictions

# Function to calculate IOU
def get_iou(ground_truth, pred):
    # Coordinates of the area of intersection
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])
     
    # Intersection height and width
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
     
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
     
    # Prediction dimensions
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
     
    iou = area_of_intersection / area_of_union
     
    return iou


def draw_bounding_boxes(image, labels, predictions, iou_threshold):
    for label in labels:
        class_id, x_center, y_center, width, height = label
        x1 = int((x_center - width / 2) * image.shape[1])
        y1 = int((y_center - height / 2) * image.shape[0])
        x2 = int((x_center + width / 2) * image.shape[1])
        y2 = int((y_center + height / 2) * image.shape[0])

        label_box = [x1, y1, x2, y2]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        highest_iou = 0
        best_pred_box = None

        for prediction in predictions:
            pclass_id, pred_box, conf = prediction
            iou = get_iou(label_box, pred_box)
            if iou > highest_iou:
                highest_iou = iou
                best_pred_box = pred_box

        if best_pred_box is not None:
            color = (0, 255, 0) if highest_iou >= iou_threshold else (0, 0, 255)
            px1, py1, px2, py2 = best_pred_box
            cv2.rectangle(image, (int(px1), int(py1)), (int(px2), int(py2)), color, 2)

    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


label_path = '/Users/ryanjewett/Documents/Reseach2024/0023Labeled/frame_0002.txt'
image_path = '/Users/ryanjewett/Documents/Reseach2024/0023Images/frame_0002.jpg'

labels = load_yolo_label(label_path)
image = load_image(image_path)


predictions = model_prediction(image)


iou_threshold = float(input("Enter the desired IOU threshold (0 to 1): "))


draw_bounding_boxes(image, labels, predictions, iou_threshold)

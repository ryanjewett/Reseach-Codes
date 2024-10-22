import cv2
from ultralytics import YOLO

from patched_yolo_infer import (
    MakeCropsDetectThem,
    CombineDetections,
    visualize_results,
)

img_path = 'Highwayexample.jpg'
img = cv2.imread(img_path)

element_crops = MakeCropsDetectThem(
    image=img,
    model_path="yolov8m-seg.pt",
    segment=True,
    show_crops=False,
    shape_x=640,
    shape_y=480,
    overlap_x=50,
    overlap_y=50,
    conf=0.2,
    iou=0.7,
    classes_list=[0, 1, 2, 3, 5, 7],
    resize_initial_size=True,
)
result = CombineDetections(element_crops, nms_threshold=0.5)

print('YOLO-Patch-Based-Inference:')
visualize_results(
    img=result.image,
    confidences=result.filtered_confidences,
    boxes=result.filtered_boxes,
    polygons=result.filtered_polygons,
    classes_ids=result.filtered_classes_id,
    classes_names=result.filtered_classes_names,
    segment=True,
    thickness=2,
    fill_mask=True,
    show_boxes=False,
    delta_colors=2,
    show_class=False,
    axis_off=False
)
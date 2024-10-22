from ultralytics import YOLO

pModel = YOLO('yolov8m.pt')
dataSet = 'train2.yaml'

results = pModel.train(data=dataSet, epochs = 5,device = 'cpu')

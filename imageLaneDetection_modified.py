import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

class_names = ['train','hot dog','skis','snowboard', 'sports ball','baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'pizza', 'donut', 'cake','teddy bear', 'hair drier', 'toothbrush']  # Add more class names as per your requirement to remove

def loadModel():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model_path = "models/tusimple_18.pth"
    model_type = ModelType.TUSIMPLE
    use_gpu = False
    lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)
    return model, lane_detector

def makeInference(image):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print("type(img)", type(img))
    output_img = lane_detector.detect_lanes(img)
    image = cv2.cvtColor(output_img , cv2.COLOR_BGR2RGB)
    results = model(image)
    boxes = results.pandas().xyxy[0]
    class_labels = results.pandas().xyxy[0]['name']
    confidences = results.pandas().xyxy[0]['confidence']
    return image, boxes, class_labels, confidences


image_path = "input.jpg"
print("type(image_path)", type(image_path))

model, lane_detector = loadModel()
image, boxes, class_labels, confidences = makeInference(image_path)

for _, row in boxes.iterrows():
    x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
    class_label = row['name']
    confidence = row['confidence']
    if class_label not in class_names:
        confidence_percent = confidence * 100
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        text = f"{class_label}: {confidence_percent:.2f} %"
        cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL) 
cv2.imshow("Detected lanes", image)
cv2.waitKey(0)
cv2.imwrite("output2.jpg",image)

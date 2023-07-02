import cv2
import pixellib

import torch
from torchvision import transforms
from PIL import Image
import numpy as np


#from pixellib.instance import instance_segmentation
#segment_image = instance_segmentation()
#segment_image.load_model("mask_rcnn_coco.h5") 

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define the class names to be removed from YoloV5 model
class_names = ['train','hot dog','skis','snowboard', 'sports ball','baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'pizza', 'donut', 'cake','teddy bear', 'hair drier', 'toothbrush']  # Add more class names as per your requirement to remove


from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = False

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)

while(True):
    ret, frame = cap.read()

    # Detect the lanes
    output_img = lane_detector.detect_lanes(frame)

    image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    #print(image.shape,"here")


    ### Apply Detection
    results = model(image)

    boxes = results.pandas().xyxy[0]
    class_labels = results.pandas().xyxy[0]['name']
    confidences = results.pandas().xyxy[0]['confidence']

    #result=segment_image.segmentFrame(frame,show_bboxes=True)
    #image=result[1]

    for _, row in boxes.iterrows():
        x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        class_label = row['name']
        confidence = row['confidence']
        if class_label not in class_names:

            confidence_percent = confidence * 100

            # Draw the bounding box rectangle
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Write the class name and confidence score on the bounding box
            text = f"{class_label}: {confidence_percent:.2f} %"
            cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    cv2.imshow("Detected lanes", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


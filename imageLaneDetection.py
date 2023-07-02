import cv2
import pixellib

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import time

#Library Added for intel Optimization
import intel_extension_for_pytorch as ipex

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

#intels optimization
model = ipex.optimize(model, dtype=torch.bfloat16)

# Define the class names to be removed from YoloV5 model
class_names = ['train','hot dog','skis','snowboard', 'sports ball','baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'pizza', 'donut', 'cake','teddy bear', 'hair drier', 'toothbrush']  # Add more class names as per your requirement to remove


from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

####### Benchmark Metrics #############
eval_time=[]

model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = False

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

print("EVALUATION STARTED")


### Apply Detection
n_images=1100
for i in range(1,n_images):
    image_path = "bdd100k/images/10k/train/"+str(i)+".jpg"

    # Read RGB images
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Detect the lanes
    start_time = time.time()
    output_img = lane_detector.detect_lanes(img)
    image = cv2.cvtColor(output_img , cv2.COLOR_BGR2RGB)
    
    #Object Detection
    results = model(image)
    end_time = time.time()
    execution_time = end_time - start_time
    eval_time.append(execution_time)
    percentage = (i + 1) / n_images* 100
    print(f"Progress: {percentage:.2f}%")

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
# Draw estimated depth
#cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL) 
#cv2.imwrite("Detected lanes", image)
#cv2.waitKey(0)

cv2.imwrite("output2.jpg",image)


average_et = sum(eval_time[100:]) / len(eval_time[100:])
print("Average execultion time over "+ str(len(eval_time[100:])) +" images with 100 warmup rounds "+str(average_et))
print("Total execultion time is "+str(sum(eval_time[100:])) +" over "+ str(len(eval_time[100:])) +" images")

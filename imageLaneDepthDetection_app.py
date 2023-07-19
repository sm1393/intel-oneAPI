import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
from MidasDepthEstimation.midasDepthEstimator import midasDepthEstimator
import matplotlib.pyplot as plt
import intel_extension_for_pytorch as ipex
import torch.optim as optim

class_names = ['train','hot dog','skis','snowboard', 'sports ball','baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'pizza', 'donut', 'cake','teddy bear', 'hair drier', 'toothbrush']  # Add more class names as per your requirement to remove

@st.cache_resource
def loadModel():
    global model, lane_detector, depthEstimator
    model_path = "models/tusimple_18.pth"
    model_type = ModelType.TUSIMPLE
    use_gpu = False
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)
    depthEstimator = midasDepthEstimator()
    return model, lane_detector, depthEstimator

model, lane_detector, depthEstimator = loadModel()

optimizer = optim.Adam(model.parameters(), lr=0.001)

model = ipex.optimize(model, optimizer=optimizer, dtype=torch.float)
lane_detector = ipex.optimize(lane_detector, optimizer=optimizer, dtype=torch.float)
depthEstimator = ipex.optimize(depthEstimator, optimizer=optimizer, dtype=torch.float)

def makeInference(_image):
    global model, lane_detector, depthEstimator
    st.image(_image, caption='Image')
    image = np.array(_image)
    with torch.no_grad():
        output_img = lane_detector.detect_lanes(image)
        colorDepth = depthEstimator.estimateDepth(image)
        results = model(image)

    combinedImg = cv2.addWeighted(image, 0.7,colorDepth,0.6,0)
    img_out = np.hstack((image, colorDepth, combinedImg))

    image = cv2.cvtColor(output_img , cv2.COLOR_BGR2RGB)
    boxes = results.pandas().xyxy[0]
    class_labels = results.pandas().xyxy[0]['name']
    confidences = results.pandas().xyxy[0]['confidence']
    return image, img_out, boxes, class_labels, confidences

uploaded_file = st.file_uploader(label="Choose a image file", type=["png", "jpg", "jpeg"])

if uploaded_file:
    _image = Image.open(uploaded_file)
    image, depthImage, boxes, class_labels, confidences = makeInference(_image)

    for _, row in boxes.iterrows():
        x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        class_label = row['name']
        confidence = row['confidence']
        if class_label not in class_names:
            confidence_percent = confidence * 100
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            text = f"{class_label}: {confidence_percent:.2f} %"
            cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(image, caption='Obstacles')
    st.image(depthImage, caption='Depth')
    
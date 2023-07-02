import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
from MidasDepthEstimation.midasDepthEstimator import midasDepthEstimator
import matplotlib.pyplot as plt

class_names = ['train','hot dog','skis','snowboard', 'sports ball','baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'pizza', 'donut', 'cake','teddy bear', 'hair drier', 'toothbrush']  # Add more class names as per your requirement to remove

@st.cache_resource
def loadModel():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model_path = "models/tusimple_18.pth"
    model_type = ModelType.TUSIMPLE
    use_gpu = False
    lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)
    depthEstimator = midasDepthEstimator()
    return model, lane_detector, depthEstimator

model, lane_detector, depthEstimator = loadModel()

def makeInference(_image):
    # st.image(_image, caption='Image')
    image = np.array(_image)
    output_img = lane_detector.detect_lanes(image)

    colorDepth = depthEstimator.estimateDepth(image)
    combinedImg = cv2.addWeighted(image, 0.7,colorDepth,0.6,0)
    img_out = np.hstack((image, colorDepth, combinedImg))

    image = cv2.cvtColor(output_img , cv2.COLOR_BGR2RGB)
    results = model(image)
    boxes = results.pandas().xyxy[0]
    class_labels = results.pandas().xyxy[0]['name']
    confidences = results.pandas().xyxy[0]['confidence']
    return image, img_out, boxes, class_labels, confidences

# uploaded_file = st.file_uploader(label="Choose a image file", type=["png", "jpg", "jpeg"])
uploaded_files = st.file_uploader(label="Choose a image file", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

for uploaded_file in uploaded_files:
    col1, col2 = st.columns(2)
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

        with col1:
            st.image(_image, caption='Original Image')
        with col2:
            st.image(image, caption='Inference')
        st.image(depthImage, caption='Depth')
        
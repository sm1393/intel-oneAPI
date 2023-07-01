from streamlit_webrtc import webrtc_streamer
import av

import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
import matplotlib.pyplot as plt

class_names = ['train','hot dog','skis','snowboard', 'sports ball','baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'pizza', 'donut', 'cake','teddy bear', 'hair drier', 'toothbrush']  # Add more class names as per your requirement to remove

@st.cache_resource
def loadModel():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model_path = "models/tusimple_18.pth"
    model_type = ModelType.TUSIMPLE
    use_gpu = False
    lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)
    return model, lane_detector

model, lane_detector = loadModel()

def makeInference(_image):
    st.image(_image, caption='Image')
    image = np.array(_image)
    output_img = lane_detector.detect_lanes(image)
    image = cv2.cvtColor(output_img , cv2.COLOR_BGR2RGB)
    results = model(image)
    boxes = results.pandas().xyxy[0]
    class_labels = results.pandas().xyxy[0]['name']
    confidences = results.pandas().xyxy[0]['confidence']
    return image, boxes, class_labels, confidences

def video_frame_callback(frame):
    image = frame.to_ndarray(format="bgr24")
    image, boxes, class_labels, confidences = makeInference(image)
    for _, row in boxes.iterrows():
        x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        class_label = row['name']
        confidence = row['confidence']
        if class_label not in class_names:
            confidence_percent = confidence * 100
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            text = f"{class_label}: {confidence_percent:.2f} %"
            cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
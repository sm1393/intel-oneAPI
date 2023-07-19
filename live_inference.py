from streamlit_webrtc import webrtc_streamer
import av
import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import time
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
from MidasDepthEstimation.midasDepthEstimator import midasDepthEstimator
import matplotlib.pyplot as plt

class_names = ['train','hot dog','skis','snowboard', 'sports ball','baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'pizza', 'donut', 'cake','teddy bear', 'hair drier', 'toothbrush']  # Add more class names as per your requirement to remove
global _objectDetection

@st.cache_resource
def loadModel():
    objectDetector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    lane_detector = UltrafastLaneDetector("models/tusimple_18.pth", ModelType.TUSIMPLE, False)
    depthEstimator = midasDepthEstimator()
    return objectDetector, lane_detector, depthEstimator

objectDetector, lane_detector, depthEstimator = loadModel()

def detectObject(_image):
    t1 = time.time()
    results = objectDetector(_image)
    t2 = time.time()
    boxes = results.pandas().xyxy[0]
    for _, row in boxes.iterrows():
        x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        class_label = row['name']
        confidence = row['confidence']
        if class_label not in class_names:
            confidence_percent = confidence * 100
            cv2.rectangle(_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            text = f"{class_label}: {confidence_percent:.2f} %"
            cv2.putText(_image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return _image, t2-t1

def detectLane(_image):
    t1 = time.time()
    results = lane_detector.detect_lanes(_image)
    t2 = time.time()
    return cv2.cvtColor(results, cv2.COLOR_BGR2RGB), t2-t1

def estimateDepth(_image):
    t1 = time.time()
    colorDepth = depthEstimator.estimateDepth(_image)
    t2 = time.time()
    combinedImg = cv2.addWeighted(_image,0.7,colorDepth,0.6,0)
    return np.hstack((_image, colorDepth, combinedImg)), t2 - t1

def video_frame_callback(frame):
    image = frame.to_ndarray(format="bgr24")
    ouputImage = image
    if _objectDetection:
        ouputImage, tact_time = detectObject(image)
        fpsText = f'{1 / tact_time:.2f} FPS'
        cv2.putText(ouputImage, fpsText, (0,22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if _laneDetection:
        ouputImage, tact_time = detectLane(image)
        fpsText = f'{1 / tact_time:.2f} FPS'
        cv2.putText(ouputImage, fpsText, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    if _depthEstimation:
        ouputImage, tact_time = estimateDepth(image)
        fpsText = f'{1 / tact_time:.2f} FPS'
        cv2.putText(ouputImage, fpsText, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    return av.VideoFrame.from_ndarray(ouputImage, format="bgr24")

webrtc_streamer(key = "example", video_frame_callback = video_frame_callback)

_objectDetection = st.checkbox("Object detection")
_laneDetection = st.checkbox("Lane detection")
_depthEstimation = st.checkbox("Depth detection")


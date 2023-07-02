from streamlit_webrtc import webrtc_streamer
import av

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

def video_depth_callback(frame):
    image = frame.to_ndarray(format="bgr24")
    colorDepth = depthEstimator.estimateDepth(image)
    combinedImg = cv2.addWeighted(image,0.7,colorDepth,0.6,0)
    img_out = np.hstack((image, colorDepth, combinedImg))
    return av.VideoFrame.from_ndarray(img_out, format="bgr24")

webrtc_streamer(key="example1", video_frame_callback=video_depth_callback)
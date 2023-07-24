from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
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
import intel_extension_for_pytorch as ipex

global class_names, _useOptimization, _objectDetection, _laneDetection, _objectAndLaneDetection, _depthEstimation
class_names = ['train','hot dog','skis','snowboard', 'sports ball','baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'pizza', 'donut', 'cake','teddy bear', 'hair drier', 'toothbrush']  # Add more class names as per your requirement to remove
checks = st.columns(5)
with checks[0]:
    _useOptimization = st.checkbox('Use optimization')
with checks[1]:
    _objectDetection = st.checkbox('Object detection')
with checks[2]:
    _laneDetection = st.checkbox('Lane detection')
with checks[3]:
    _objectAndLaneDetection = st.checkbox('Obstacles and Lane detection')
with checks[4]:
    _depthEstimation = st.checkbox('Depth detection')

class VideoProcessor:
    @st.cache_resource
    def loadModel():
        objectDetector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        lane_detector = UltrafastLaneDetector("models/tusimple_18.pth", ModelType.TUSIMPLE, False, False)
        depthEstimator = midasDepthEstimator()
        return objectDetector, lane_detector, depthEstimator

    @st.cache_resource
    def loadOptimizedModel():
        objectDetector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        objectDetector.eval()
        objectDetector = ipex.optimize(objectDetector)
        lane_detector = UltrafastLaneDetector("models/tusimple_18.pth", ModelType.TUSIMPLE, False, True)
        depthEstimator = midasDepthEstimator()
        return objectDetector, lane_detector, depthEstimator

    objectDetector, lane_detector, depthEstimator = loadModel()
    optimizedObjectDetector, optimizedLane_detector, optimizedDepthEstimator = loadOptimizedModel()

    def detectObject(self, _image):
        if _useOptimization:
            t1 = time.time()
            with torch.no_grad():
                results = self.optimizedObjectDetector(_image)
            t2 = time.time()
        else:
            t1 = time.time()
            results = self.objectDetector(_image)
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

    def detectLane(self, _image):
        if _useOptimization:
            t1 = time.time()
            with torch.no_grad():
                results = self.optimizedLane_detector.detect_lanes(_image)
            t2 = time.time()
        else:
            t1 = time.time()
            results = self.lane_detector.detect_lanes(_image)
            t2 = time.time()
        return cv2.cvtColor(results, cv2.COLOR_BGR2RGB), t2-t1

    def detection(self, _image):
        if _useOptimization:
            t1 = time.time()
            with torch.no_grad():
                objects = self.optimizedObjectDetector(_image)
                lanes = self.optimizedLane_detector.detect_lanes(_image)
            t2 = time.time()
        else:
            t1 = time.time()
            objects = self.objectDetector(_image)
            lanes = self.lane_detector.detect_lanes(_image)
            t2 = time.time()
        boxes = objects.pandas().xyxy[0]
        for _, row in boxes.iterrows():
            x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            class_label = row['name']
            confidence = row['confidence']
            if class_label not in class_names:
                confidence_percent = confidence * 100
                cv2.rectangle(lanes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                text = f"{class_label}: {confidence_percent:.2f} %"
                cv2.putText(lanes, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return lanes, t2-t1

    def estimateDepth(self, _image):
        if _useOptimization:
            t1 = time.time()
            with torch.no_grad():
                colorDepth = self.optimizedDepthEstimator.estimateDepth(_image)
            t2 = time.time()
        else:
            t1 = time.time()
            colorDepth = self.depthEstimator.estimateDepth(_image)
            t2 = time.time()
        combinedImg = cv2.addWeighted(_image,0.7,colorDepth,0.6,0)
        return np.hstack((_image, colorDepth, combinedImg)), t2 - t1

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        ouputImage = image
        if _objectDetection:
            ouputImage, tact_time = self.detectObject(image)
            fpsText = f'{1 / tact_time:.2f} FPS'
            cv2.putText(ouputImage, fpsText, (0,22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if _laneDetection:
            ouputImage, tact_time = self.detectLane(image)
            fpsText = f'{1 / tact_time:.2f} FPS'
            cv2.putText(ouputImage, fpsText, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        if _objectAndLaneDetection:
            ouputImage, tact_time = self.detection(image)
            fpsText = f'{1 / tact_time:.2f} FPS'
            cv2.putText(ouputImage, fpsText, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        if _depthEstimation:
            ouputImage, tact_time = self.estimateDepth(image)
            fpsText = f'{1 / tact_time:.2f} FPS'
            cv2.putText(ouputImage, fpsText, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        return av.VideoFrame.from_ndarray(ouputImage, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)
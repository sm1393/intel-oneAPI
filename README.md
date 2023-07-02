# Autonomous Object and Lane Detection and Depth Estimation Inference Pytorch
For autonomous vehicles, object detection is a crucial task that requires a real-time deep-learning model capable of accurately navigating, considering all possible objects like pedestrians, vehicles, traffic signs, signals, etc., lanes, and identifying distance to nearby objects.​

![!Ultra fast lane detection](https://github.com/ibaiGorordo/Ultrafast-Lane-Detection-Inference-Pytorch-/blob/main/doc/img/detected%20lanes.jpg)
Source: https://www.flickr.com/photos/32413914@N00/1475776461/


# Installation
```
pip install -r requirements

```

# Pretrained model
Download the pretrained model from the [original repository](https://github.com/cfzd/Ultra-Fast-Lane-Detection) and save it into the **[models](https://github.com/ibaiGorordo/Ultrafast-Lane-Detection-Inference-Pytorch-/tree/main/models)** folder. 

# Ultra fast lane detection - TuSimple([link](https://github.com/cfzd/Ultra-Fast-Lane-Detection))

 * **Input**: RGB image of size 1280 x 720 pixels.
 * **Output**: Keypoints for a maximum of 4 lanes (left-most lane, left lane, right lane, and right-most lane).
 
# Object Detection and Lane Detection

 * **Image inference**:
 
 ```
 python imageLaneDetection.py 
 ```
 
  * **Webcam inference**:
 
 ```
 python webcamLaneDetection.py
 ```
 
  * **Video inference**:
 
 ```
 python videoLaneDetection.py
 ```
 # Depth Estimation

 * **Image inference**:
 
 ```
 python imageDepthEstimation.py 
 ```
 
  * **Webcam inference**:
 
 ```
 python webcamDepthDetection.py
 ```
 
  * **Video inference**:
 
 ```
 python videoDepthDetection.py
 ```

 # [Inference video Example](https://youtu.be/0Owf6gef1Ew) 
 ![!Ultrafast lane detection on video](https://github.com/ibaiGorordo/Ultrafast-Lane-Detection-Inference-Pytorch-/blob/main/doc/img/laneDetection.gif)
 
 Original video: https://youtu.be/2CIxM7x-Clc (by Yunfei Guo)
 

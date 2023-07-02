# Autonomous Object and Lane Detection and Depth Estimation Inference Pytorch
For autonomous vehicles, object detection is a crucial task that requires a real-time deep-learning model capable of accurately navigating, considering all possible objects like pedestrians, vehicles, traffic signs, signals, etc., lanes, and identifying distance to nearby objects.​

![!Ultra fast lane detection](https://github.com/ibaiGorordo/Ultrafast-Lane-Detection-Inference-Pytorch-/blob/main/doc/img/detected%20lanes.jpg)
Source: https://www.flickr.com/photos/32413914@N00/1475776461/


### 🍞 Installation
The project was developed with [**Python>=3.7**](https://www.python.org/downloads/) and [**Pytorch>=1.10**](https://pytorch.org/get-started/locally/).
```bash
# Creating Anaconda Virtual Environment Inside Project Folder
conda create -p venv python==3.7.2 -y

# Activating the created Virtual Environment
conda activate venv/

#Installing Dependencies
pip install -r requirements.txt

#Installing Pytorch for CPU
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

#Installing Intel Pytorch Optimisation Ipex Dependency
pip install intel_extension_for_pytorch==1.13.100 -f https://developer.intel.com/ipex-whl-stable-cpu
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
 

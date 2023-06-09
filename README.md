# intel-oneAPI

#### Team Name - Momentum
#### Problem Statement - Object Detection For Autonomous Vehicles
#### Team Leader Email - sudb97@gmail.com


<div align="center">
  
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.10+-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
<br>
</div>



## 📜 Prototype Brief:
  Description:
  The model will be trained on image input which will be temporal in nature. Required data-preprocessing will be done using intel's oneDAL libraries. Further training will be done using HybridNet neural network Architecture with the intel's oneDNN pytorch optimization to perform faster training. Finally the real time inference will be achieved using the intel's oneDNN libraries which will provide mainly three outputs that are, object bounding box, object class and lane detection.This project is part of the Intel OneAPI Hackathon 2023, we have used HybridNet for tackling the object detection and Segmentation Problem. HybridNets is an end2end perception network for multi-tasks. Our work focused on traffic object detection, drivable area segmentation and lane detection.  HybridNets can run real-time on embedded systems, and obtains SOTA Object Detection, Lane Detection on BDD100K Dataset. 
  
![Screenshot from 2023-06-09 03-07-52](https://github.com/sudb97/intel-oneAPI/assets/42773775/80a9e6ed-c410-427b-b8c7-2dbd75232088)
![Screenshot from 2023-06-09 01-45-27](https://github.com/sudb97/intel-oneAPI/assets/42773775/1fe84198-50a8-44f0-8fdb-1304d2bac493)




  
## Tech Stack: 
   List Down all technologies used to Build the prototype **Clearly mentioning Intel® AI Analytics Toolkits, it's libraries and the SYCL/DCP++ Libraries used**
   ![Screenshot from 2023-06-09 01-25-31](https://github.com/sudb97/intel-oneAPI/assets/42773775/f8a02538-83e5-443f-8f68-cfd40f6c5a25)


## 🍞 Project Structure
```bash
HybridNets
│   backbone.py                     # Model configuration
│   export.py                       # UPDATED 10/2022: onnx weight with accompanying .npy anchors
│   hubconf.py                      # Pytorch Hub entrypoint
│   hybridnets_test_images_old.py   # Image inference
│   hybridnets_test_images.py       # Modified hybridnets_test to get inference time for a no of images
│   hybridnets_test_videos_old.py   # Video inference
│   hybridnets_test_videos.py       # Modified hybridnets_test_videos to get inference for different length of videos
│   speedup_test.py                 # Calculate 
│   train.py                        # Train script
│   train_ddp.py                    # DistributedDataParallel training (Multi GPUs)
│   val.py                          # Validate script
│   val_ddp.py                      # DistributedDataParralel validating (Multi GPUs)
│   frameCount_vs_time_plot.py      # Plot framecount vs time taken to infer
│   imageCount_vs_time_plot.py      # Plot imagecount vs time taken to infer
│   speedup_test.py                 # Speedup test for comparison between with and without optimization
│
├───demo                            # Images and videos for testing inference
│
├───demo_results                    # Post processing results on test images and videos to validate the inference
│
├───data                            # Time record for inferences on different conditions
│
├───plots                           # Comparison plots between inferences on different conditions
│
├───encoders                        # https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/encoders
│       ...
│
├───hybridnets
│       autoanchor.py               # Generate new anchors by k-means
│       dataset.py                  # BDD100K dataset
│       loss.py                     # Focal, tversky (dice)
│       model.py                    # Model blocks
│
├───projects
│       bdd100k.yml                 # Project configuration
│
└───utils
│   constants.py
│   plot.py                         # Draw bounding box
│   smp_metrics.py                  # https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/metrics/functional.py
│   utils.py                        # Various helper functions (preprocess, postprocess, eval...)
```

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
 
### 🚩 Project Demo - Step-by-Step Code Execution Instructions:
```bash
# Download end-to-end weights
curl --create-dirs -L -o weights/hybridnets.pth https://github.com/datvuthanh/HybridNets/releases/download/v1.0/hybridnets.pth

# Image inference with Intel Optimisation
python3 hybridnets_test_images.py --source demo/images --output demo_result/images --use_optimization True --enable_postprocessing True

# Image inference without Intel Optimisation
python3 hybridnets_test_images.py --source demo/images --output demo_result/images --use_optimization False --enable_postprocessing True

# Video inference with Intel Optimisation
python3 hybridnets_test_videos.py --source demo/video --output demo_result/video --use_optimization True --enable_postprocessing True

# Video inference without Intel Optimisation
python3 hybridnets_test_videos.py --source demo/video --output demo_result/video --use_optimization False --enable_postprocessing True

# Result is saved in a new folder called demo_result
```

## 🚩 Usage
### Data Preparation
dataset structure:
```bash
HybridNets
└───datasets
    ├───imgs
    │   ├───train
    │   └───val
    ├───det_annot
    │   ├───train
    │   └───val
    ├───da_seg_annot
    │   ├───train
    │   └───val
    └───ll_seg_annot
        ├───train
        └───val
```

For BDD100K: (DataSets Used)
- [imgs](https://bdd-data.berkeley.edu/)
- [det_annot](https://drive.google.com/file/d/1QttvnPI1srmlHp86V-waD3Mn5lT9f4ky/view?usp=sharing)
- [da_seg_annot](https://drive.google.com/file/d/1FDP7ojolsRu_1z1CXoWUousqeqOdmS68/view?usp=sharing)
- [ll_seg_annot](https://drive.google.com/file/d/1jvuSeK-Oofs4OWPL_FiBnTlMYHEAQYUC/view?usp=sharing)

## 🚩 PPT and Demonstration Link

## 🚩 Benchmarking Results
![Screenshot from 2023-06-09 20-32-57](https://github.com/sudb97/intel-oneAPI/assets/42773775/28bc0404-0f66-4975-a19f-a9539bf74b2c)
![Screenshot from 2023-06-09 20-34-19](https://github.com/sudb97/intel-oneAPI/assets/42773775/51b1da23-c512-478f-807e-77673d3ea6c8)


## 📜 References 
> [**HybridNets: End-to-End Perception Network Paper Link**](https://arxiv.org/abs/2203.09035)

## 📜 What I Learned:
  1. Expansion of domain knowledge in deep learning based computer vision techniques like object detection and segmentation.<br>
  2. Usage of End2End Perception Network Hybridnet to do image and video inferencing for simultaneous object detection and segmentation.<br>
  3. Importance of Robustness in Autonomous Driving: Development of a robust algorithm that could handle all weather conditions, night time conditions as evident by the          results.<br>
  4. Incorporation of Intel oneAPI libraries oneDNN libraries.<br>
  5. Learnt about Optimisation techniques for faster inferencing, specifically the libraries developed by Intel.<br>
  
  

# intel-oneAPI

#### Team Name - Momentum
#### Problem Statement - Object Detection For Autonomous Vehicles
#### Team Leader Email - sudb97@gmail.com


<div align="center">
  
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.10+-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
<br>
</div>



##📜Prototype Brief:
  Description:
  The model will be trained on image input which will be temporal in nature. Required data-preprocessing will be done using intel's oneDAL libraries. Further training will be done using HybridNet neural network Architecture with the intel's oneDNN pytorch optimization to perform faster training. Finally the real time inference will be achieved using the intel's oneDNN libraries which will provide mainly three outputs that are, object bounding box, object class and lane detection.This project is part of the Intel OneAPI Hackathon 2023, we have used HybridNet for tackling the object detection and Segmentation Problem. HybridNets is an end2end perception network for multi-tasks. Our work focused on traffic object detection, drivable area segmentation and lane detection.  HybridNets can run real-time on embedded systems, and obtains SOTA Object Detection, Lane Detection on BDD100K Dataset. 
![Screenshot from 2023-06-09 01-05-25](https://github.com/sudb97/intel-oneAPI/assets/42773775/5e42a9ba-d7fe-4141-80cb-569f74e88ec8)


  
## Tech Stack: 
   List Down all technologies used to Build the prototype **Clearly mentioning Intel® AI Analytics Toolkits, it's libraries and the SYCL/DCP++ Libraries used**

### 🍞Project Structure
```bash
HybridNets
│   backbone.py                   # Model configuration
|   export.py                     # UPDATED 10/2022: onnx weight with accompanying .npy anchors
│   hubconf.py                    # Pytorch Hub entrypoint
│   hybridnets_test.py            # Image inference
│   hybridnets_test_videos.py     # Video inference
│   train.py                      # Train script
│   train_ddp.py                  # DistributedDataParallel training (Multi GPUs)
│   val.py                        # Validate script
│   val_ddp.py                    # DistributedDataParralel validating (Multi GPUs)
│
├───encoders                      # https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/encoders
│       ...
│
├───hybridnets
│       autoanchor.py             # Generate new anchors by k-means
│       dataset.py                # BDD100K dataset
│       loss.py                   # Focal, tversky (dice)
│       model.py                  # Model blocks
│
├───projects
│       bdd100k.yml               # Project configuration
│
├───ros                           # C++ ROS Package for path planning
│       ...
│
└───utils
    |   constants.py
    │   plot.py                   # Draw bounding box
    │   smp_metrics.py            # https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/metrics/functional.py
    │   utils.py                  # Various helper functions (preprocess, postprocess, eval...)
```

### 🍞 Installation
The project was developed with [**Python>=3.7**](https://www.python.org/downloads/) and [**Pytorch>=1.10**](https://pytorch.org/get-started/locally/).
```bash
pip install -r requirements.txt
```
 
### 🚩 Project Demo - Step-by-Step Code Execution Instructions:
```bash
# Download end-to-end weights
curl --create-dirs -L -o weights/hybridnets.pth https://github.com/datvuthanh/HybridNets/releases/download/v1.0/hybridnets.pth

# Image inference
python hybridnets_test.py -w weights/hybridnets.pth --source demo/image --output demo_result --imshow False --imwrite True

# Video inference
python hybridnets_test_videos.py -w weights/hybridnets.pth --source demo/video --output demo_result

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

For BDD100K: 
- [imgs](https://bdd-data.berkeley.edu/)
- [det_annot](https://drive.google.com/file/d/1QttvnPI1srmlHp86V-waD3Mn5lT9f4ky/view?usp=sharing)
- [da_seg_annot](https://drive.google.com/file/d/1FDP7ojolsRu_1z1CXoWUousqeqOdmS68/view?usp=sharing)
- [ll_seg_annot](https://drive.google.com/file/d/1jvuSeK-Oofs4OWPL_FiBnTlMYHEAQYUC/view?usp=sharing)


## 📜 References 
> [**HybridNets: End-to-End Perception Network Paper Link**](https://arxiv.org/abs/2203.09035)

## 📜 What I Learned:
  -> Usage of End2End Perception Network Hybridnet to do image and video inferencing for simultaneous object detection and segmentation.
  -> Incorporation of Intel oneAPI libraries oneDNN libraries.

# CSCI-6660-TermProject
# Introduction and Background
Object detection is a critical task in computer vision, pivotal in applications like autonomous vehicles and robotics. Traditional DL approaches, like R-CNN and SSD, are known for their robustness in static environments but may struggle with adaptability in dynamic contexts. DRL methods, such as Deep Q-Networks (DQN), offer the potential for real-time adaptability but face challenges in convergence and generalization. This research aims to explore and compare these methodologies to identify their strengths and limitations across static and dynamic environments.
# Research Question
How do object detection functionality and metrics differ between Deep Learning (DL) approaches and Deep Reinforcement Learning (DRL) approaches, such as DQN and PPO, across various visual contexts? 
# Dataset
We have used coco dataset from kaggle website for our project - https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
# Installation and setup
1. Install jupyter notebook using anaconda navigator.
2. Install python3 version in your system.
3. Install necessary modules like cv2, torchvision, tensorflow using python pip.
4. Use Faster RCNN and SSD by taking weights from ResNet50 and VGG16 respectively as shown below in case of static environment using DL.
faster_rcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
ssd = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT).
5. Use 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28', which is a pretrained model for dynamic environment with RCNN and please ensure that webcam is functioning properly.
6. Use 'MobileNetV2' for SSD, which is a pretrained model for dynamic environment with SSD and even here please ensure that webcam is functioning properly.
# How is Our Work Different from Others
Our work diverges by relying entirely on reinforcement learning through DQN while leveraging COCO annotations. The primary distinction lies in your dynamic agent design, where user-specified object names determine the detection focus, rather than a purely automated pipeline. Our project uses the COCO dataset, which is significantly larger and more diverse than datasets used in earlier RL-based detection work. This introduces challenges related to computation and generalization. While traditional models like YOLO or Faster R-CNN are purely supervised, our approach combines reinforcement learning with deep learning to explore bounding box predictions dynamically. The use of DQN makes our work suitable for scenarios requiring iterative refinement or interaction, such as autonomous agents in complex environments.
# Our Journey in phases
# Phase 1
[CSCI-6660-AIProposal.pptx](https://github.com/user-attachments/files/17878723/CSCI-6660-AIProposal.pptx)
# Phase 2
# Phase 3
[CSCI-6660-AI-FinalPPT.pptx](https://github.com/user-attachments/files/18099758/CSCI-6660-AI-FinalPPT.pptx)

# Youtube Video
https://youtu.be/rfEDKLaf9Ww
# GitHub Repository
https://github.com/CharithaC951/CSCI-6660-TermProject

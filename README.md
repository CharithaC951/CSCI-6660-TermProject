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
# Code

# Results and Conclusion
Case 1 - Single object in single image in static environment
![image](https://github.com/user-attachments/assets/61e4a3a4-b6f8-47b0-9dba-e2465990346d)

During Static environment and in case of single object in an image, SSD 
and FasterRCNN are working very similar and giving good performance.

Case 2 - Multiple objects in single image in static environment
![image](https://github.com/user-attachments/assets/f3c992e1-a9c9-45bf-be28-00f74a891eb2)
![image](https://github.com/user-attachments/assets/b2f38f4f-ed56-400c-a718-7842e3a32fa3)

While working with multiple objects in an image where the objects are 
tiny or overlapping, SSD is giving better results in comparison with FasterRCNN.

Case 3a - Dynamic environemnt using frames in Faster RCNN

Bird

![image](https://github.com/user-attachments/assets/86116ffd-52cd-48b1-8d5a-232131b5371d)

Person

![image](https://github.com/user-attachments/assets/6c47ba29-8f04-4f5c-a214-f9a2c7b10462)

Case 3b - Dynamic environemnt using frames in SSD

Bird

![image](https://github.com/user-attachments/assets/2252690b-5578-4616-9355-d80310222d12)

Person

![image](https://github.com/user-attachments/assets/7cb18a00-ef64-4d9c-be4c-0b66fc7d5183)

As of Dynamic environment, SSD and FasterRCNN are giving similar and promising results but not as good as their performance in static environment. FasterRCNN is performing better on SSD in dynamic environment. Factors such as camera quality, frame rate, lighting and frequently changing external noises in the frames plays key role in decision making.

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

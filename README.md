# CSCI-6660-TermProject
# Introduction and Background
Object detection is a critical task in computer vision, pivotal in applications like autonomous vehicles and robotics. Traditional DL approaches, like R-CNN and SSD, are known for their robustness in static environments but may struggle with adaptability in dynamic contexts. DRL methods, such as Deep Q-Networks (DQN), offer the potential for real-time adaptability but face challenges in convergence and generalization. This research aims to explore and compare these methodologies to identify their strengths and limitations across static and dynamic environments.
# Research Question
How do object detection functionality and metrics differ between Deep Learning (DL) approaches and Deep Reinforcement Learning (DRL) approaches, such as DQN and PPO, across various visual contexts? 
# Dataset
We have used coco dataset from kaggle website for our project - https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
# How is Our Work Different from Others
Our work diverges by relying entirely on reinforcement learning through DQN while leveraging COCO annotations. The primary distinction lies in your dynamic agent design, where user-specified object names determine the detection focus, rather than a purely automated pipeline. Our project uses the COCO dataset, which is significantly larger and more diverse than datasets used in earlier RL-based detection work. This introduces challenges related to computation and generalization. While traditional models like YOLO or Faster R-CNN are purely supervised, our approach combines reinforcement learning with deep learning to explore bounding box predictions dynamically. The use of DQN makes our work suitable for scenarios requiring iterative refinement or interaction, such as autonomous agents in complex environments.
# Our Journey in phases
# Phase 1
[CSCI-6660-AIProposal.pptx](https://github.com/user-attachments/files/17878723/CSCI-6660-AIProposal.pptx)
# Phase 2
# Phase 3
# Youtube Video
	
# GitHub Repository
https://github.com/CharithaC951/CSCI-6660-TermProject

# CSCI-6660-TermProject
# Phase 1
[CSCI-6660-AIProposal.pptx](https://github.com/user-attachments/files/17878723/CSCI-6660-AIProposal.pptx)
# Phase 2
Research Question
How do object detection functionality and metrics differ between Deep Learning (DL) approaches and Deep Reinforcement Learning (DRL) approaches, such as DQN and PPO, across various visual contexts? 
________________________________________
Introduction
Object detection is a critical task in computer vision, pivotal in applications like autonomous vehicles and robotics. Traditional DL approaches, like R-CNN and SSD, are known for their robustness in static environments but may struggle with adaptability in dynamic contexts. DRL methods, such as Deep Q-Networks (DQN), offer the potential for real-time adaptability but face challenges in convergence and generalization. This research aims to explore and compare these methodologies to identify their strengths and limitations across static and dynamic environments.
________________________________________
Related Literature
1.	Deep Reinforcement Learning for Object Detection: Highlights the application of DRL in dynamically learning bounding boxes for object detection Source.
2.	A Survey of Modern Deep Learning based Object Detection Models: Explores the scalability and accuracy improvements of DQN-based models on datasets like COCO Source.
3.	Attentive Layer Separation for Object Classification and Object Localization in Object Detection Source.
   
How is Our Work Different from Others

Our work diverges by relying entirely on reinforcement learning through DQN while leveraging COCO annotations. The primary distinction lies in your dynamic agent design, where user-specified object names determine the detection focus, rather than a purely automated pipeline. Our project uses the COCO dataset, which is significantly larger and more diverse than datasets used in earlier RL-based detection work. This introduces challenges related to computation and generalization. While traditional models like YOLO or Faster R-CNN are purely supervised, our approach combines reinforcement learning with deep learning to explore bounding box predictions dynamically. The use of DQN makes our work suitable for scenarios requiring iterative refinement or interaction, such as autonomous agents in complex environments.
	
Proposed Approach
Our approach differs by introducing a dynamic agent that uses user-specified object targets for detection, improving adaptability and specificity in varying environments. The primary distinction lies in our dynamic agent design, where user-specified object names determine the detection focus, rather than a purely automated pipeline.
________________________________________
Progress and Achievements

Developed initial pipelines for both DL (R-CNN, SSD) and Deep Reinforcement Learning (DQN) approaches using python 3 in Jupyter notebook.
Integrated COCO dataset annotations for supervised learning and reinforcement signal generation.
This has been done on both static and dynamic environments for R-CNN and SSD like testing using  COCO dataset in static and using key frames using web camera in dynamic environments respectively.
Ensured proper functionality of code for the above mentioned cases.
Implemented a reward mechanism based on Intersection over Union (IoU) for DQN to refine object localization iteratively.
Uploaded related progress in Github respectively.
________________________________________
Challenges and Workarounds

Challenges
1.	Lighting, distance and external support for the keyframes plays a good role for better accuracy, when video capturing in real time. May lose some information in this process for RCNN and SSD.
2.	Need highly divergent data set classes for better prediction.
3.	Dataset related challenges: In images with multiple objects, the specified object may occupy a very small region, making detection harder for the model.
4.	Model Related Challenges: If actions correspond to bounding box adjustments, the action space becomes very large, leading to slower learning.
5.	Training Challenges: Training with DQN on high-resolution images and large datasets like COCO is time intensive.
6.	Computational Challenges: Large memory requirements may limit the batch size, slowing down the model’s ability to generalize.
7.	DQN agents often overfit to specific environments. The model may fail to generalize to unseen images or object variations (e.g., occlusions, rotations).
8.	Modifying DQN to handle user-specified objects dynamically requires an additional mechanism for incorporating object-specific state representations.
9.	Proximal policy optimization (PPO) has limited research as of now but has more scope of learning that challenges our current knowledge level.

Workarounds
Working with dataset in different lighting, range and other external support helps in achieving problems of identifying images and adding wide range of diversity for the data when capturing through web camera provided lot of support for better performance. For RDL based approaches, the learning is based on reward signals from comparing predicted bounding boxes with actual objects in the image. This feedback loop helps improve the agent’s performance over time. The reward mechanism (e.g., Intersection over Union (IoU) or other metrics) ensures the agent is optimized for better localization. Exploring object detection with the help of localization using Reinforcement deep learning approaches strengthens our work. Also focusing on PPO gives wide knowledge on overall techniques used for object detection and various conditions.

Future Plan on this project
1.	Expanding our research on Proximal Policy Optimization
2.	Collectively providing a comparative study for Deep Learning and Reinforcement Deep Learning algorithms in both static and dynamic environments
________________________________________
GitHub Repository
https://github.com/CharithaC951/CSCI-6660-TermProject


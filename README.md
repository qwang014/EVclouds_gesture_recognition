# Space-time Event Clouds for Gesture Recognition: from RGB Cameras to Event Cameras
Created by Qinyi Wang, Yexin Zhang, Junsong Yuan, Yilong Lu from Nanyang Technological University and State University of New York at Buffalo.

This repository provides a Tensorflow implementation of this [paper] in WACV19. This implementation is developed based on original [PointNet] and [PointNet++] repositories.

**Introduction

The recently developed event cameras can directly sensethe motion by generating an asynchronous sequence ofevents, i.e., an event stream, where each individual event(x,y,t)corresponds to the space-time location when a pixel sensor captures an intensity change. Compared with RGB cameras, event cameras are frameless but can capture much faster motion, therefore have great potential for rec-ognizing gestures of fast motions. To deal with the unique output of event cameras, previous methods often treat eventstreams as time sequences, thus do not fully explore the space-time sparsity and structure of the event stream data. In this work, we treat the event stream as a set of 3D pointsin space-time, i.e., *space-time event clouds*. To analyze event clouds and recognize gestures, we propose to leverage PointNet, a neural network architecture originally designed for matching and recognizing 3D point clouds. We adapt PointNet to cater to event clouds for real-time gesture recognition. 

   [paper]: https://cse.buffalo.edu/~jsyuan/papers/2019/WACV_2019_Qinyi.pdf
   [PointNet]: https://github.com/charlesq34/pointnet
   [PointNet++]: https://github.com/charlesq34/pointnet2

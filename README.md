# Tracking and Identifying People in Real­Time:
## Classifying team membership based on team uniform’s color

### Table of Contents
**[Short description](#short-description)**  
**[Installation, configuration and running](#installation-configuration-running)**  
**[Results](#results)**  
**[Abstract](#Abstract)**  

### Short description
As part of a semester project at [EPFL](www.epfl.ch),


### Installation configuration running
All information relative to the installation process for this software can be found in the `documentation.pdf` file. The `test/` folder contains a short video and a bash script which should allow you to check that the software is running correclty.

### Results
The image below links to a YouTube video, illustrating the final results achieved. Additional images, and intermediate results can be found in the `report.pdf` file.

[![Video of project results](http://img.youtube.com/vi/pOEk0HC6Kvc/0.jpg)](http://www.youtube.com/watch?v=pOEk0HC6Kvc)

### Abstract
Below you'll find the original abstract from the report of this project. The full report can be found in the `report.pdf` file.

 
*"This project is done within the context of our Bachelor’s project at the Computer Vision Laboratory of the “École Polytechnique Fédérale de Lausanne” and its spin­off company PlayfulVision which specializes in real­time video tracking in sports environments. The goal of the project is to identify the team membership of players in a team­based sport by using the colors of their uniforms. This is performed on videos captured within the setting provided by the Sports Center of the “Université de Lausanne” and PlayfulVision. A single camera view is used at a time and the performance being aimed for is real­time, or as close as possible.

First a literature review was done to assess existing work which could be applied for this project. The conclusion of this review was that our work would be focused around the use of an object detection algorithm which locates a specified object and its parts on an image. More specifically the Deformable Part Models [10] algorithm, abbreviated DPM. In the context of our project it is used to find the players and their respective torso which contains the necessary information to establish their team­membership. Using only such an algorithm unfortunately does not yield the best results, therefore part of our project relies upon optimizing the use of the algorithm. These optimisations take the form of a background subtraction combined with a blob extraction. They allow us to run the DPM algorithm only on regions which are players. This results in a significant speed and correctness improvement.

The DPM algorithm cannot decide the team membership of the player but it allows us to filter out the irrelevant information on a frame, so that we only have the torsos, in our case the jerseys of the players. This specific information is then extracted to be used as a feature. Finally, every player’s feature is given to a classifier to determine the player’s team membership.

We select as features the color histograms of the torsos, more specifically, histograms of the hue values of the pixel colors in the HSV color space. In order to possible team. The template which is the most similar to a given player’s feature determines his or her team membership. Correlation is the similarity metric we use for these comparisons.

The results of this implementation are extremely satisfactory when there is little to no occlusion of players, if the players are standing up straight (not bending over, notcrouching, etc.) and finally if the players are neither too close or too far from the camera.However as soon as the postures are altered, the occlusion increased or the positions changed, the performance decreases significantly.

For future work, it would be interesting to evaluate the results when using multiple camera views to overcome the occlusion and position issues. Another interesting aspect would be to see the impact of using a DPM model trained specifically for different postures.

This report ends with acknowledgments and a synopsis of our personal experience."*

# Image-Segementation By Pytorch
This is a simple Fully Connected Neural Network (FCNN) for two class Image Segemtation task.

# Pre-Process
This file contains the pre-process and how to generate dataloader used in training loop. 
Due to only one pair of image and ground truth is available, one strightforward idea is croping the image into smaller size and adding transforms like rotation, flipping. 

# Train 
This file contains the Module class and training loop.  
Model Structure 

#Predict
This file contain the functions for making predictions. The strategy is croping the large predicited RGB image into small sizes and feed them into the trained FCNN. After predicting for every small size image, combine them back into the mask for  original image. 

#Parameters 
This file contain the used parameters for this task 
| Parameters | Values | 
| ---------  | ------ |
| Batch Size |   64   |
| Train Length | 500  |
| Input Size | (256,256) |
| maximum epochs | 20 |

# Image-Segmentation By Pytorch
This is a simple Fully Connected Neural Network (FCNN) for two class Image Segmentation task.

# Pre-Process.py
This file contains the pre-process and how to generate data-loader used in training loop. 
Due to only one pair of image and ground truth is available, one straightforward idea is cropping the image into smaller size and adding transforms like rotation, flipping,adjusting brightness and contrast to form a data set. 

Example  

![Digraph](images_in_readme\preprocess_example.png)

# Train.py
This file contains the Module class and training loop. 


Model Structure

![Digraph](images_in_readme\Digraph.png)



# Predict.py
This file contain the functions for making predictions. 

The strategy is cropping the large predicted RGB image into small sizes and feed them into the trained FCNN. After predicting for every small size image, combine them back into the mask for  original image. 

Final Result

![Final_Result](images_in_readme\final_result.png)

# Parameters.py
This file contain the used parameters for this task 
| Parameters                                  |          Values           |
| ------------------------------------------- | :-----------------------: |
| Batch Size                                  |            64             |
| Train Length                                |            500            |
| Input Size                                  |         (256,256)         |
| maximum epochs                              |            20             |
| Learning rate                               |           0.005           |
| Kernel size at each conv layer              | [(3,3),(3,3),(3,3),(5,5)] |
| Padding size at each conv layer             |         [1,1,1,2]         |
| number of input channels at each conv layer |      [3,16,32,16,2]       |
| Optimizer                                   |           Adam            |



# Further Improvements  

1. Brightness Problem 

   From the comparison between ground truth and predicted mask, we could observe that the roofs with low brightness or dark color is not recognized well. Based on this problem, the transformation of adjusting brightness on data augmentation has been tried. However, the effects are limited. Therefore, more fined augmentation tricks could be applied to overcome this problem or increases the data size.   

   ![Compare](images_in_readme\compare.jpg)

2. Output all pixels with same value

   During training, it is quite often that the model makes predictions with same-class value among all pixels.

   This is mainly caused by imbalance between two classes. Most of region in given RGB picture is non-roof class. Therefore, at the start of training, the accuracy or performance will increase as long as predicting all the pixels with same value (ie. non-roof class). Afterwards, the model is hard to get rid of this pattern and  cause this problem.

   One improvement idea is balancing the ratio between two classes by cropping the RGB images elaborately and make the input data contain larger region of roof class.    

   
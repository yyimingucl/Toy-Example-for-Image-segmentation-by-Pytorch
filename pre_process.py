# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 20:43:48 2021

@author: 10929
"""

from parameters import params
import torch
import numpy as np
from torchvision import transforms 
from skimage.io import imread
from skimage.color import rgba2rgb,rgb2gray
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from random import random

config = params()
    

def load_image(path):
    image = imread(path)
    if image.shape[-1] == 4:
        image = torch.tensor(rgba2rgb(image),dtype=torch.float32)
    else:
        image = torch.tensor(image,dtype=torch.float32)
    return image.permute(2,0,1)


def load_mask(path):
    mask = imread(path)
    mask = torch.tensor(rgb2gray(mask),dtype=torch.float32)
    return mask



#%% generate data 
Rotate_90 = transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation((90,90)),]),p=0.5)
Rotate_180 = transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation((180,180)),]),p=0.5)
Rotate_270 = transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation((270,270)),]),p=0.5)


transformer = transforms.Compose([transforms.RandomCrop(config.input_size),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.RandomVerticalFlip(p=0.5),
                                  Rotate_90,
                                  Rotate_180,
                                  Rotate_270])


def gen_data(image,mask):
    data =  torch.cat((image, mask.unsqueeze(0)), dim=0)

    data = transformer(data)
    data = torch.split(data, [3, 1], dim=0)
    
    image = data[0]
    if random() > 0.4:
        image = F.adjust_brightness(image,np.random.uniform(0.5,1.5))
    if random() > 0.5:
        image = F.adjust_gamma(image, gamma=np.random.uniform(0.5,1.5))
    if random() > 0.5:
        image = F.adjust_contrast(image, np.random.uniform(0.5,1.5))
        
    mask = data[1]
    mask = mask.type(torch.float32)
    onehot = torch.cat([mask,((mask + 1) % 2)], dim=0)
    return image, onehot




#%% Dataset 

class Segementation_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, rgb_path, gt_path, data_size, augmentation=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image = load_image(rgb_path)
        self.mask = load_mask(gt_path)
        self.augmentation = augmentation
        self.data_size = data_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.image
        mask = self.mask

        if self.augmentation:
            image,mask = self.augmentation(image,mask)
    
        return image,mask
    
    



    

























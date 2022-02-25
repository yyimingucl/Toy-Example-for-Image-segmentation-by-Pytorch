# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 03:05:15 2021

@author: 10929
"""
import argparse
import numpy as np
import random
import torch
import os

root_path = os.path.dirname(os.path.abspath(__file__))

def set_manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
set_manual_seed(2)
print("set random seed : 2")

#6and0.4
def params():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    

    #for training
    add_arg("--epoch_length", default=500, 
            help="number of instances in one epoch")
    add_arg("--batch_size", default = 64, help="batch size", type=int)
    
    add_arg("--lr", default=0.005, help="initial learning rate", type=float)
    
    add_arg("--optimizer", default="adam", help="Optimizer for training",type=str)
    
    add_arg("--max_epoch", default=20, help="number training epochs",type=int)
    
    add_arg("--save_dir", default=os.path.join(root_path), 
            help="Path to save model",type=str)
    
    add_arg("--model_type", default="FCNN", help="Fully Connected Neural Network"
            ,type=str)
    
    add_arg("--kernel_size", default=[(3,3),(3,3),(3,3),(5,5)], 
            help="Kernel Size for each Convolutional Layer ",type=list)
    
    add_arg("--paddings", default=[1,1,1,2], 
            help="padding size for each layer",type=list)
    
    add_arg("--num_channels", default=[3,16,32,16,2],
            help="number of in-channels in each layer",type=list)
    
    add_arg("--input_size", default = (256,256), 
            help="Input dimension (width,height)",type=tuple)
    
    args = parser.parse_args()
    return args
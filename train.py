# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 23:41:33 2021

@author: 10929
"""
# Load Packages
import torch
import torch.nn as nn
from pre_process import Segementation_Dataset,gen_data
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
from parameters import params


config = params()
#%% Define Model

class FCNN(nn.Module):
    """Input->Conv1->ReLU->Conv2->ReLU->MaxPool->Conv3->ReLU->UpSample->Conv4->Output"""
    def __init__(self, channels,kernels,paddings):
        super(FCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1],
                      kernel_size=kernels[0],padding=paddings[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], 
                      kernel_size=kernels[1], padding=paddings[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=channels[2], out_channels=channels[3], 
                      kernel_size=kernels[2], padding=paddings[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=channels[3], out_channels=channels[4], 
                      kernel_size=kernels[3], padding=paddings[3]),
        )
    def forward(self, x):
        x = self.conv(x)
        return torch.sigmoid(x)


#%% Define Accurarcy
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/np.product(y_test.shape)
    acc = torch.round(acc*100)
    
    return acc


#%% Train process
    
def train():
    rgb_path = "./rgb.png"
    gt_path = "./gt.png"
    train_dataset = Segementation_Dataset(rgb_path, gt_path, config.epoch_length, augmentation=gen_data)
    valid_dataset = Segementation_Dataset(rgb_path, gt_path, config.epoch_length, augmentation=gen_data)
    train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=False)

    valid_loader = DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCNN(config.num_channels, config.kernel_size, config.paddings)
    optimizer = Adam(model.parameters(),lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=2, cooldown=2)
    
    train_loss_list = []
    valid_loss_list = []
    valid_loss_min = np.Inf
    for epoch in range(1, config.max_epoch+1):

    # keep track of training and validation loss
        train_acc = []
        valid_acc = []
        train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        model.to(device)
        bar = tqdm.auto.tqdm(train_loader, postfix={"train_loss":0.0})
        for data, target in bar:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            BCE_loss = nn.BCELoss()
            loss = BCE_loss(output, target)
            #print(loss)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            bar.set_postfix(ordered_dict={"train_loss":loss.item()})
            train_acc.append(binary_acc(output, target))
        ######################    
        # validate the model #
        ######################
        model.eval()
        del data, target
        with torch.no_grad():
            bar = tqdm.auto.tqdm(valid_loader, postfix={"valid_loss":0.0})
            for data, target in bar:
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                #IoU = IoULoss()
                #loss = IoU(output,target) 
                CE_loss = nn.BCELoss()
                loss = CE_loss(output, target)
                # update average validation loss 
                valid_loss += loss.item()*data.size(0)
                bar.set_postfix(ordered_dict={"valid_loss":loss.item()})
                valid_acc.append(binary_acc(output, target))
        
        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        scheduler.step(valid_loss)
        
        
        
        # print training/validation statistics 
        print('')
        print('Epoch: {}  Training Loss: {:.6f}  Validation Loss: {:.6f} Training Acc: {:.1f}% Validation Acc: {:.1f}%'.format(
            epoch, train_loss, valid_loss, np.average(train_acc), np.average(valid_acc)))
        print('')
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), os.path.join(config.save_dir,"FCNN.pt"))
            
    
    plt.figure(figsize=(10,10))
    plt.plot(train_loss_list,  marker='o', label="Training Loss")
    plt.plot(valid_loss_list,  marker='o', label="Validation Loss")
    plt.ylabel('loss', fontsize=22)
    plt.legend()
    plt.show()
    
#%%  
if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__)) 
    os.chdir(cur_path)
    train()
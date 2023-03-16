import os
import cv2
import math
import time
import tarfile
import numbers
import threading
import queue as Queue
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split, DataLoader, Dataset
from torchsummary import summary

import mxnet as mx
from mxnet import recordio

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import cuda_device
from utils.cuda_device import to_device, DeviceDataLoader
from utils.dataset import ArcFaceDataset
from utils.network import ResNet50, ResNet101, ResNet152
from utils.losses import CosFace, ArcFace, ElasticCosFace, ElasticArcFace
from utils.plots import plot_accuracies, plot_losses
from utils.display import make_predictions

random_seed = 42
torch.manual_seed(random_seed)
   
device = cuda_device.get_default_device()

root_dir1 = "../MyElasticFace/faces_emore/" #For Ubuntu
root_dir2 = "D:/Face/faces_emore/" #For Windows


dataset = ArcFaceDataset(root_dir1)

#Limiting the dataset for computation purposes
lim_factor = 0.0001 #Set to 1 for using the entire dataset
lim_size = int(lim_factor * len(dataset))
lim_dataset_size = len(dataset) - lim_size
large_ds, lim_ds = random_split(dataset, [lim_dataset_size, lim_size])

test_factor = 0.1
test_size = int(test_factor * len(lim_ds))
train_size = len(lim_ds) - test_size
train_ds, test_ds = random_split(lim_ds, [train_size, test_size])

val_factor = 0.1
val_size = int(val_factor * len(lim_ds))
train_size = len(train_ds) - val_size
train_ds, val_ds = random_split(train_ds, [train_size, val_size])

batch_size=128

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size, num_workers=4, pin_memory=True)

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(test_dl, device)


model = ResNet50()
to_device(model, device)

new_logits = CosFace() #ArcFace() #ElasticArcFace()
to_device(new_logits, device)
#new_logits.to(device)

loss_Function = F.cross_entropy

        
def loss_batch(model, loss_func, xb, yb, opt=None, opt_out=None, metric=None):
    #Generate predictions
    features = model(xb) #F.normalize(model(xb), p=2.0, dim=1)
    preds = new_logits(features, yb)
    
    #Generate probabilities
    #preds = F.softmax(preds, dim=1)
    
    #Calculate loss
    loss = loss_func(preds, yb)
    
    if opt is not None:
        #Compute gradients
        loss.backward()
        
        #Gradient Clipping
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        
        #update parameters
        opt.step()
        opt_out.step()
        
        #Reset Gradients
        opt.zero_grad()
        opt_out.zero_grad()

    metric_result = None
    if metric is not None:
        #compute the metric
        metric_result = metric(preds, yb)
        
    return loss.item(), len(xb), metric_result
 
    
def evaluate(model, loss_fn, valid_dl, metric=None):
    with torch.no_grad():
        #pass each batch through the model
        results = [loss_batch(model, loss_fn, xb, yb, metric=metric) for xb, yb in valid_dl]
        #separate losses, counts and metrics
        losses, nums, metrics = zip(*results)
        #Total size of the dataset
        total = np.sum(nums)
        #Avg. loss across batches
        avg_loss = np.sum(np.multiply(losses, nums))/total
        avg_metric = None
        
        if metric is not None:
            #Avg of metric across batches
            avg_metric = np.sum(np.multiply(metrics, nums)) / total

    return avg_loss, total, avg_metric
    
    
def fit(epochs, model, loss_fn, train_dl, valid_dl, lr=None, lr_func=None, metric=None, opt_fn=None):
    train_losses, train_metrics, val_losses, val_metrics = [], [], [], []
    
    #instantiate the optimizer
    if opt_fn is None: opt_fn = torch.optim.SGD
    opt = opt_fn(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    opt_out = opt_fn(new_logits.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    scheduler_network = torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lr_func)
    scheduler_out = torch.optim.lr_scheduler.LambdaLR(optimizer=opt_out, lr_lambda=lr_func)
    
    for epoch in range(epochs):
        start = time.time()
        ep_train_losses, train_len, ep_train_metrics = [], [], []
        
        #Training
        model.train()
        for xb, yb in train_dl:
            train_loss, len_xb, train_metric = loss_batch(model, loss_fn, xb, yb, 
                                                          opt, opt_out, metric)
            ep_train_losses.append(train_loss)
            train_len.append(len_xb)
            ep_train_metrics.append(train_metric)
            
        scheduler_network.step()
        scheduler_out.step()
        
        total = np.sum(train_len)
        avg_train_loss = np.sum(np.multiply(ep_train_losses, train_len)) / total
        avg_train_metric = None
        if metric is not None:
            avg_train_metric = np.sum(np.multiply(ep_train_metrics, train_len)) / total

        #Evaluation
        model.eval()
        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result

        #Record the loss and metric
        train_losses.append(avg_train_loss)
        train_metrics.append(avg_train_metric)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
        
        #Checkpointing the model - saving every 'n' epochs
        checkpoint_path = "Checkpoints/ResNet50_CosFace_16_Mar_" +str(epoch+1)+".pt"
        
        if ((epoch+1)%2 == 0):
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                #'model_optimizer_state_dict': opt.state_dict(),
                #'fc_optimizer_state_dict': opt_out.state_dict(),
                'new_logits': new_logits,
                'train_acc': avg_train_metric,
                'val_acc': val_metric,
            }, checkpoint_path)
        
        end = time.time()

        #Print progress:
        if metric is None:
            print('Epoch [{}/{}], Train_loss: {:.4f}, Val_loss: {:.4f}'
                  .format(epoch+1, epochs, train_loss, val_loss))
            print(str(round((end - start), 2)) + " secs")
        
        else:
            print('Epoch [{}/{}], Train_loss: {:.4f}, Val_loss: {:.4f}, Train_{}: {:.4f}, Val_{}: {:.4f}'
                  .format(epoch+1, epochs, avg_train_loss, val_loss, metric.__name__,  avg_train_metric, 
                          metric.__name__, val_metric))
            print(str(round((end - start), 2)) + " secs")

    return train_losses, train_metrics, val_losses, val_metrics
    
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)
    
    
val_loss, _, val_acc = evaluate(model, loss_Function, val_dl, metric=accuracy) #metric=None
#print('Loss: {:.4f}'.format(val_loss))
print('Before training ... Val loss: {:.4f}, Val accuracy: {:.4f}'.format(val_loss, val_acc))

opt_func = torch.optim.SGD


def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.5 ** len(
        [m for m in [3,6,8,10,12,14,16,18,20,22,24] if m - 1 <= epoch])
          
#No dynamic updation in LR
def unit_lr(epoch):
    return 1
    
    
num_epochs = 24
lr = 0.01


history = fit(epochs=num_epochs, model=model, loss_fn=loss_Function, 
              train_dl=train_dl, valid_dl=val_dl, lr=lr, lr_func=lr_step_func, 
              metric=accuracy, opt_fn=opt_func)
              
              
train_losses, train_metrics, val_losses, val_metrics = history


# Creating a new data frame
newDataframe = pd.DataFrame()
filename = "ResNet50_CosFace_16_Mar_outputs_data.xlsx"
newDataframe['Train Loss'] = train_losses
newDataframe['Val Loss'] = val_losses
newDataframe['Train Acc.'] = train_metrics
newDataframe['Val Acc.'] = val_metrics
# Converting the data frame to an excel file
newDataframe.to_excel(filename, index = False)


plot_accuracies(num_epochs, train_metrics, val_metrics)
plot_losses(num_epochs, train_losses, val_losses)


result = evaluate(model, loss_Function, test_dl, accuracy)
print('Test_loss: {:.4f}, Test_acc.: {:.4f}'.format(result[0], result[2]))


#Save the latest model

saved_path = "Checkpoints/ResNet50_CosFace_16_Mar_" +str(num_epochs)+".pt"

torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    #'model_optimizer_state_dict': opt.state_dict(),
    #'fc_optimizer_state_dict': opt_out.state_dict(),
    'new_logits': new_logits,
    'train_acc': train_metrics[-1],
    'val_acc': val_metrics[-1],
}, saved_path)


make_predictions(model, new_logits, test_dl, root_dir1, 30)

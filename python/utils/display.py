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
#from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, DataLoader, Dataset
from torchsummary import summary

import mxnet as mx
from mxnet import recordio

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

random_seed = 42
torch.manual_seed(random_seed);

import cuda_device
from cuda_device import to_device, DeviceDataLoader
from utils.dataset import ArcFaceDataset
from network import ResNet50, ResNet101, ResNet152
from losses import CosFace, ArcFace, ElasticCosFace, ElasticArcFace


root_dir1 = "faces_emore/" #For Ubuntu
root_dir2 = "D:/Face/faces_emore/" #For Windows

dataset = ArcFaceDataset(root_dir1)

#Limiting the dataset for computation purposes

lim_factor = 1 #Set to 1 for using the entire dataset
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


model = ResNet152(85742)
to_device(model, device)

new_logits = CosFace() #ArcFace() #ElasticArcFace()
to_device(new_logits, device)


checkpoint = torch.load("Checkpoints/model_06_Mar_18.pt")
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
new_logits = checkpoint['new_logits']
train_acc = checkpoint['train_acc']
val_acc = checkpoint['val_acc']

print('Epoch {}, Train acc.: {:.4f}, Val acc.: {:.4f}'.format(epoch, train_acc, val_acc))


def create_list(root_dir):
    img_list = []
    #%% 1 ~ 5908396, or 0~5908395, #85742 identities
    path_imgidx = os.path.join(root_dir, 'train.idx')
    path_imgrec = os.path.join(root_dir, 'train.rec')
    imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    
    for i in range(5822653):
        header, s = recordio.unpack(imgrec.read_idx(i+1))
        img_list.append(header.label)
        
    lst = np.array(img_list)
    
    return imgrec, lst
    
imgrec, lst = create_list(root_dir1)

def fetch_pred_image(imgrec, lst, idx):
    idxs = np.where(lst==idx)[0]
    
    header, s = recordio.unpack(imgrec.read_idx(idxs[-1]))
    img = mx.image.imdecode(s).asnumpy()
    
    return img
    
def prepare_plot(image, origTarget, predTarget, size=5):
    plt.rcParams["figure.figsize"] = (15*size/25, 15*size/5)
    rand = np.random.randint(0, 128, size)
    
    for i in range(size):
        img_fetch = fetch_pred_image(imgrec, lst, predTarget[rand[i]])
        plt.subplot(size, 3, 3*i+1)
        plt.imshow(image[rand[i]])
        plt.axis('off')
        plt.title('Actual id: ' + str(origTarget[rand[i]]))
        
        plt.subplot(size, 3, 3*i+2)
        plt.imshow(np.real(img_fetch))
        plt.axis('off')
        plt.title('Predicted id: ' + str(predTarget[rand[i]]))
        
        plt.subplot(size, 3, 3*i+3)
        plt.axis('off')
        if(origTarget[rand[i]] == predTarget[rand[i]]):
            plt.title('MATCHED!', fontsize = 12, color='g')
        else:
            plt.title('MIS-MATCHED!', fontsize = 15, color='r')
    
    plt.show()
    
    
def make_predictions(model, dataLoader, size):
    with torch.no_grad():
        #set the model in evaluation mode
        model.eval()
        
        for xb, yb in dataLoader:
            features = model(xb)
            pred = new_logits(features, yb)
            
            softmax = F.softmax(pred, dim=1)
            sum_check = torch.sum(softmax, dim=1)
            #print(sum_check)
            max_value = torch.max(softmax, dim=1)[1]
            ##print("Predicted labels: ", max_value)
            ##print("Actual labels: ", yb)
            
            matches = max_value == yb
            #print(matches)
        
#             print(pred.shape)
#             print(pred[100].shape)
#             print(torch.max(pred[100]))
#             softmax = F.softmax(pred[100], dim=0)
#             max_value = torch.max(softmax)
#             print(max_value)
#             print("Actual label: ", yb[100].item())
#             index = torch.where(softmax == max_value)[0]
#             print("Predicted label: ", index.item())

            invTransform = transforms.Compose(
                [transforms.Normalize(mean=[0., 0., 0.], std=[1/0.5, 1/0.5, 1/0.5]),
                 transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1]),
                ])
            xb = invTransform(xb)
    
            xb = xb.cpu().numpy() #Convert the tensor input into a numpy object
            #print(xb.shape)

            xb = np.transpose(xb, (0, 2, 3, 1)) #Shifting the channel into the 3rd dimension       
            #print(xb.shape)

            yb = yb.cpu().numpy() #Convert the tensor input into a numpy object
            #print(yb[0])

            max_value = max_value.cpu().numpy() #Convert the tensor input into a numpy object
            #print(max_value[0])

            prepare_plot(xb, yb, max_value, size)
            
            break
            
make_predictions(model, test_dl, 30)

import torch 
print(torch.cuda.is_available())
import torch
import os
from torch.utils.data import DataLoader
import cv2
import functools
import torchmetrics
import torchvision
import torchvision.io
from vid_dataset_file import VidDataset, Collate_FN
from slow_model import ResidualBlock, SlowNet
import torch.utils.data.dataloader as Dataloader
torch.cuda.empty_cache()
import matplotlib.pyplot as plt
import cv2 
import numpy as np
from PIL import Image

dict_of_labels = {0:'Hugging', 1:'Fighting', 2: 'Falling'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_folder = '/home/ubuntu/padded_ims/train/'
val_folder = '/home/ubuntu/padded_ims/val/'
train_dataset, val_dataset = VidDataset(train_folder), VidDataset(val_folder)
print(train_dataset[100][0].shape)
train_dataloader = DataLoader(batch_size = 1,dataset= train_dataset,shuffle=True, num_workers=1)
val_dataloader = DataLoader(batch_size = 1,dataset= val_dataset,shuffle=True,num_workers=1)
model = SlowNet(number_of_classes = 3).to(device)
# input_path_to_latest_model = input('input path to latest model')
model = torch.load('/home/ubuntu/giphy_project_files/model_checkpoints/Modelsave4EPOCHEND.pt')

# number_of_example = int(input('how_many_examples?'))
idx = 0
for batch, label, targets in val_dataloader:
    print(batch.shape)
    idx+=1
    if idx > 5: break
    model.eval()
    targets = targets[0].cpu().item()
    batch=batch.permute(0,4,1,2,3)
    print(batch.shape)
    batch = batch.to(device).float()
    outputs = (model(batch))
    best = torch.argmax(outputs,dim=-1)[0].cpu().item()
    print(f'output: {best} {dict_of_labels[best]} targets: {targets} {dict_of_labels[targets]} {label}')

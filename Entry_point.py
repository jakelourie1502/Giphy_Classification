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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_folder = '/home/ubuntu/padded_ims/train/'
val_folder = '/home/ubuntu/padded_ims/val/'
train_dataset, val_dataset = VidDataset(train_folder), VidDataset(val_folder)
train_dataloader = DataLoader(batch_size = 1,dataset= train_dataset,shuffle=True, num_workers=1)
val_dataloader = DataLoader(batch_size = 1,dataset= val_dataset,shuffle=True,num_workers=1)


model = SlowNet(number_of_classes = 3).to(device)
# model = torch.load('/home/ubuntu/giphy_project_files/model_checkpoints/Modelsave0EPOCHEND.pt')
acc = torchmetrics.Accuracy().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)


#pre-saving model
epoch = 'pre_save'
model_save_path = f'/home/ubuntu/new_github_folder/Giphy_Classification/model_checkpoint/Modelsave{epoch}.pt'
torch.save(model, model_save_path)

import time
start_time = time.time()

epochs = 8
for epoch in range(epochs):
  model.train()
  idx = 0
  for batch, labels, targets in train_dataloader:
    idx+=1
    try:
      batch = batch.permute(0,4,1,2,3) #batch, channels, frames, h, w
      batch, targets = batch.to(device), targets.to(device)
      outputs = model(batch)
      loss = criterion(outputs, targets)
      loss.backward()
      batch_acc = acc(outputs, targets)
      #Once every 256, compute and print accuracy for train and optimize step.
      if (idx+1) % 256 == 0:
          optimizer.step(); optimizer.zero_grad()
          accuracy_256_examples = acc.compute()
          print(f'Time: {int(time.time()-start_time)}\nAccuracy for 256_batch number {(idx+1)//100}: {accuracy_256_examples}')
          acc.reset()
          model_save_path = f'/home/ubuntu/new_github_folder/Giphy_Classification/model_checkpoint/Modelsave{epoch}-{idx}.pt'
          torch.save(model, model_save_path)
    except:
      print(f'failed at idx {idx}')
  for batch,_, targets in val_dataloader:
    try: 
      model.eval()
      batch, targets = batch.to(device).float(), targets.to(device).long()
      
      outputs = model(batch)
      acc(outputs, targets)
    except:
      print('failed on a batch for acc')
  accuracy_epoch = acc.compute()
  print(f'End of Epoch Accuracy val: {accuracy_epoch}')
  acc.reset()
  
    ###Model saving

  model_save_path = f'/home/ubuntu/new_github_folder/Giphy_Classification/model_checkpoint/Modelsave{epoch}EPOCHEND.pt'
  torch.save(model, model_save_path)
   
    

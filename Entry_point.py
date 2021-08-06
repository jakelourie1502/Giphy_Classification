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

# filesample = os.listdir(f'{train_folder}Falling/')[0]
# reader = torchvision.io.read_video(f'{train_folder}Falling/{filesample}')[0]
# print(reader.shape)

# a, _,_ = next(iter(train_dataloader))
# print(a.shape)
# b, _,_ = next(iter(val_dataloader))
# print(b.shape)
# quit()

model = SlowNet(number_of_classes = 3).to(device)
acc = torchmetrics.Accuracy().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

'''Not in use'''
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,5,10,15], gamma=0.3)    #not used right now.  
###HOW TO CALL THIS FUNCTION
#if self.lr_scheduler is not None: self.lr_scheduler.step()

#pre-saving model
epoch = 'pre_save'
model_save_path = f'/home/ubuntu/giphy_project_files/model_checkpoints/Modelsave{epoch}.pt'
torch.save(model, model_save_path)

import time
start_time = time.time()

epochs = 5
for epoch in range(1):
  model.train()
  idx = 0
  for batch, _, targets in train_dataloader:
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
          print(f'optimized after {int(time.time()-start_time)}')
          accuracy_256_examples = acc.compute()
          print(f'Time: {int(time.time()-start_time)}\nAccuracy for 256_batch number {(idx+1)//100}: {accuracy_256_examples}')
          acc.reset()
          model_save_path = f'/home/ubuntu/giphy_project_files/model_checkpoints/Modelsave{epoch}-{idx}.pt'
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

  model_save_path = f'/home/ubuntu/giphy_project_files/model_checkpoints/Modelsave{epoch}EPOCHEND.pt'
  torch.save(model, model_save_path)
   
    

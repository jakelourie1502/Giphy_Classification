import torch
import os 
import torchvision 
import numpy as np

#create a dataset class
class VidDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.all_files, self.targets, self.target_idx = self._get_list_of_all_files()

    def _get_list_of_all_files(self):
        '''
        In: Path (in our case, 'padded_ims/train' directory)
        Out: list of all the file names so that they're callabale
        '''
        image_names = []
        targets = []
        target_idx = []
        list_of_folders = os.listdir(self.path)
        for fold_idx, fold in enumerate(list_of_folders):
            if fold != '.DS_Store' and fold != 'Giphy.pem':
                folder = f'{self.path}{fold}'
                indi_file_names = os.listdir(folder)
                image_names += [f'{folder}/{x}' for x in indi_file_names]
                targets += [fold]*len(indi_file_names)
                target_idx += [fold_idx]*len(indi_file_names)
        return image_names, targets, target_idx
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        img_name = self.all_files[idx] 
        reader = torchvision.io.read_video(img_name)
        frames = self.pre_down_sample(reader)        
        frames = (frames/255)
        return frames, self.targets[idx], self.target_idx[idx]

    def pre_down_sample(self,reader):
        '''
        in: reader object that has read a tensor and fps from a video
        out: downsampled tensor up to 100 frames and between 8 and 16 fps
        '''
    
        fps = reader[2]['video_fps']
        vid_array = reader[0]
        length = int(vid_array.shape[0])
        downsample_fps=fps//8
        max_downsample_to_keep_at_least_18_frames = length // 18
        downsample = int(max(1,min(downsample_fps, max_downsample_to_keep_at_least_18_frames)))
        if downsample > 1:
            frames_to_keep = length//downsample
            DS_array = np.linspace(0,(frames_to_keep*downsample)-1, frames_to_keep)
            vid_array = vid_array[DS_array,:,:,:]
        return vid_array[:100] #max 100 frames


class Collate_FN():
    def __call__(self,batch):
        videos, targets, target_idx = zip(*batch)
        # batch = [torch.Tensor(video) for video in videos] #create list of variable size tensors
        batch = torch.nn.utils.rnn.pad_sequence(videos)
        target_tensor = torch.tensor(target_idx)
        return batch[:100], targets, target_tensor #100 is the max video size
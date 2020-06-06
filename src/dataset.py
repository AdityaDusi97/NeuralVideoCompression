import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

test_video = [ 'kitti_2011_09_26_drive_0064_sync', 
      'kitti_2011_09_26_drive_0020_sync', 
      'kitti_2011_09_26_drive_0014_sync', 
      'kitti_2011_09_26_drive_0013_sync', 
      'kitti_2011_09_26_drive_0113_sync', 
      'kitti_2011_09_26_drive_0093_sync', 
      'kitti_2011_09_26_drive_0064_sync' ]
# 0926, 0064: residential
# 0926, 0020: residential
# others: city
# test_video = ['kitti_2011_09_26_drive_0014_sync']

def build_list(root_dir, mode):
    out = []
    videos = os.listdir(root_dir)
    videos = [i for i in videos if os.path.isdir(os.path.join(root_dir, i))]

    if mode is not 'final_test':
        for v in videos:
            if v not in test_video: # skip vidoes in test_videos
                v_path = os.path.join(root_dir, v, mode)
                frame_num = []
                for fn in os.listdir(v_path):
                    if ('pos' in fn) and fn.split('_pos')[0]:
                        frame_num.append(os.path.join(v, mode, fn).split('_pos')[0])
                out.append(frame_num)
    else:
        for v in test_video:
            v_path1 = os.path.join(root_dir, v, 'train')
            v_path2 = os.path.join(root_dir, v, 'test')
            frame_num = []
            for fn in os.listdir(v_path1):
                if ('pos' in fn) and fn.split('_pos')[0]:
                    frame_num.append(os.path.join(v, 'train', fn).split('_pos')[0])
            for fn in os.listdir(v_path2):
                if ('pos' in fn) and fn.split('_pos')[0]:
                    frame_num.append(os.path.join(v, 'test', fn).split('_pos')[0])
            out.append(frame_num)
        
    
    out = [j for i in out for j in i]
    return out

class ResidualDataset(Dataset):

    def __init__(self, root_dir, mode, device):
        """
        Args:   
            root_dir (string): Directory with all the images.
            mode     (string): train/test/eval

            dataset structure:
                data/v1/train/000_pos.png
                             /000_neg.png ...
                       /test /070_pos.png ...
        """
        self.root_dir = root_dir 
        # data
        self.mode = mode
        # train
        self.filelist= build_list(root_dir, mode) 
        print( "dataset mode: {}, length: {}".format(mode, (len(self.filelist))) )
        # v1/train/000

        self.device = device

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir,
                                self.filelist[idx]) # data/v1/train/000
        # print("img_name: {}".format(img_name))
        img_pos = cv2.imread(img_name + '_pos.png').astype(np.float32)
        img_neg = cv2.imread(img_name + '_neg.png').astype(np.float32)
        img = np.transpose(img_pos - img_neg, axes=(2,0,1)) / 255.0

        img = torch.from_numpy(img).to(device=self.device)

        sample = {'image': img, 'name': self.filelist[idx]}

        return sample

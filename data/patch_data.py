import json
import os

import cv2
import os.path as op
import numpy as np
import pandas as pd
import torch
import elasticdeform
from torchvision import transforms

import torch.utils.data as data

from torchvision import transforms
from sklearn.model_selection import train_test_split

class PatchData(data.Dataset):
    def __init__(self, 
                data_dir,
                stage = 'fit',
                class_num = 5,
                train=True,
                repeat_channel = False,
                aug_prob=0.5,
                img_mean=(0.485),
                img_std=(0.229)):
        # self.__dict__.update(locals())
        self.train = train
        self.img_mean = img_mean
        self.img_std = img_std
        self.data_dir = data_dir
        self.stage = stage
        self.aug_prob = aug_prob
        self.class_num = class_num
        self.repeat_channel = repeat_channel
        if self.stage=='fit':
            self.csv_file = data_dir+'/train_meta.csv'
        else:
            self.csv_file = data_dir+'/test_meta.csv'

        self.check_files()

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value. 
        file_list_path = op.join(self.data_dir, self.csv_file)
        df = pd.read_csv(file_list_path)
        if self.stage=='fit':
            fl_train, fl_val = train_test_split(df, test_size=0.1, random_state=32)
            self.data_list = fl_train if self.train else fl_val
        else:
            self.data_list = df

    def to_one_hot(self, idx):
        out = np.zeros(self.class_num , dtype=float)
        out[idx] = 1
        return out

    def __getitem__(self, idx):
        
        # path = os.path.join(self.data_dir.format('img_dir'),'train',self.data_list.iloc[index]['img_id']+'.png')
        img_id = self.data_list.iloc[idx]['patch_id']
        

        if self.stage=='fit':
            img = cv2.imread(os.path.join(self.data_dir,'img_dir/train',self.data_list.iloc[idx]['patch_id']+'.png'),cv2.IMREAD_ANYDEPTH)         
        else:
            img = cv2.imread(os.path.join(self.data_dir,'img_dir/test',self.data_list.iloc[idx]['patch_id']+'.png'),cv2.IMREAD_ANYDEPTH)

        if self.train:
            img = elasticdeform.deform_random_grid(img,sigma=15, points=3,axis=(0,1))

        img = torch.unsqueeze(torch.from_numpy(img/65536), 0).float()

        if self.class_num == 3:
            if str(self.data_list.iloc[idx]['pathology']) == 'NORMAL':
                label = 0
            elif str(self.data_list.iloc[idx]['pathology']) == 'BENIGN' or str(self.data_list.iloc[idx]['pathology']) == 'BENIGN_WITHOUT_CALLBACK':
                label = 1 if str(self.data_list.iloc[idx]['type']) == 'calcification' else 2
            else:
                label = 1 if str(self.data_list.iloc[idx]['type']) == 'calcification' else 2

        if self.class_num ==5:
            if str(self.data_list.iloc[idx]['pathology']) == 'NORMAL':
                label = 0
            elif str(self.data_list.iloc[idx]['pathology']) == 'BENIGN' or str(self.data_list.iloc[idx]['pathology']) == 'BENIGN_WITHOUT_CALLBACK':
                label = 1 if str(self.data_list.iloc[idx]['type']) == 'calcification' else 3
            else:
                label = 1 if str(self.data_list.iloc[idx]['type']) == 'calcification' else 4
                
        trans = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(self.aug_prob),
            transforms.RandomVerticalFlip(self.aug_prob),
            transforms.RandomRotation(10),
            transforms.Normalize(self.img_mean, self.img_std)
        ) if self.train else torch.nn.Sequential(
            transforms.Normalize(self.img_mean, self.img_std)
        )

        img_tensor = trans(img)
        if self.repeat_channel:
            img_tensor = img_tensor.repeat(3,1,1)
        lesion_tensor = torch.tensor(label)

        output = dict(
            input = img_tensor,
            lesion = lesion_tensor,
        )
        return output

    def __len__(self):
        return 10
        return len(self.data_list)  # It's a fake length due to the trick that every loader maintains its own list
        # return self.num_sampleclass
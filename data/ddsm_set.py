import lmdb
import cv2
import os
import random

import numpy as np
import pandas as pd
import torch
from os import path
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from tqdm import tqdm
from lib.preprocess_utils import read_resize_img,segment_breast,crop_img,add_img_margins,get_max_connected_area,get_candidate_patch,draw_rect,overlap_patch_roi
from lib.preprocess_utils import convert_to_8bit, show_img_cv

class DdsmSet(Dataset):
    def __init__(self, 
                data_dir,
                train=True,
                target_size = (896,1152),
                gs_255 = False,
                dataset_stat = False
                ):
        self.img_dir = os.path.join(data_dir,'cbis-ddsm-png/')
        if train:
            csv_file = path.join(data_dir, "csv/train_roi.csv")
        else:
            csv_file = path.join(data_dir, "csv/test_roi.csv")
        self.roi_df = pd.read_csv(csv_file)
        self.target_size = target_size
        self.full_images = self.roi_df['image file path'].copy().drop_duplicates().to_numpy()
        self.gs_255 = gs_255
        if dataset_stat :
            self.dataset_stat = dataset_stat
            self.init_dataset_stat()



    def __len__(self):
        return len(self.full_images)

    def one_hot(self,type,pathology,pathology_display = True):
        if pathology=='MALIGNANT':
            if type == 'calcification':
                return 3 if pathology_display else 1
            if type == 'mass':
                return 4 if pathology_display else 2
        else:
            if type == 'calcification':
                return 1
            if type == 'mass':
                return 2
        return 0

    def init_dataset_stat(self):
        self.env = None
        self.txn = None
        self.psum = 0
        self.psum_sq = 0
        self.pmax = 0
        self.pmin = 65535
    
    def update_dataset_stat(self, full_img_segment):
        self.psum    += full_img_segment.sum()
        self.psum_sq += (full_img_segment ** 2).sum()
        self.pmax = max(self.pmax, full_img_segment.max())
        self.pmin = min(self.pmax, full_img_segment.min())

    def get_dataset_stat(self):
        count = len(self) * self.target_size[0] * self.target_size[1]
        total_mean = self.psum / count
        total_var  = (self.psum_sq / count) - (total_mean ** 2)
        total_std  = np.sqrt(total_var)
        return total_mean, total_var, total_std, self.pmax, self.pmin

    def __getitem__(self, idx):
        full_img_path = self.full_images[idx]
        img_id = full_img_path.split('/')[1][:-4]
        lesions = self.roi_df[self.roi_df['image file path']==full_img_path]
        pathology_global = False

        full_img = read_resize_img(path.join(self.img_dir,full_img_path), target_size=self.target_size,gs_255=self.gs_255)

        full_img_segment,bbox = segment_breast(full_img)

        ann_img = np.zeros(full_img_segment.shape)
        # img_tensor = torch.from_numpy(full_img).type(torch.FloatTensor) 
        # silded_patches = img_tensor.unfold(1,self.patch_size,stride).unfold(0,self.patch_size,stride)
        # silde_shape = silded_patches.shape
        # silded_patches = silded_patches.reshape(silde_shape[0]*silde_shape[1],1,self.patch_size,self.patch_size).detach().numpy()        
        for idx,lesion in lesions.iterrows():
            mask = lesion['ROI mask file path']
            type = lesion['abnormality type']
            pathology = lesion['pathology']

            if pathology == 'MALIGNANT':
                pathology_global = True

            ann_code = self.one_hot(type, pathology, True)

            mask_img =read_resize_img(path.join(self.img_dir,mask), target_size=self.target_size,gs_255=True)
            mask_img = crop_img(mask_img,bbox)
            
            if mask_img.shape != ann_img.shape:
                print(mask_img.shape, ann_img.shape)

            assert mask_img.shape == ann_img.shape
            ann_img = np.where(mask_img>0,ann_code,ann_img)

        

        full_img_segment = cv2.resize(full_img_segment.astype(np.uint16),self.target_size)
        ann_img = cv2.resize(ann_img.astype(np.uint8),self.target_size)
        
        if self.dataset_stat:
            self.update_dataset_stat(full_img_segment)


        # print(np.sum(ann_img))
        return img_id,full_img_segment,ann_img,pathology_global,0


from glob import glob
import os
import cv2
import random

import pydicom
import numpy as np
import pandas as pd
import torch
from os import path
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from tqdm import tqdm
from lib.preprocess_utils import read_resize_img,segment_breast,crop_img,add_img_margins,get_max_connected_area,get_candidate_patch,draw_rect,overlap_patch_roi
from lib.preprocess_utils import convert_to_8bit, show_img_cv

class CsawSet(Dataset):
    def __init__(self, 
                data_dir,
                train=True,
                target_size = (1152,896),
                gs_255 = False
                ):
        self.data_dir = data_dir
        mask_dir = path.join(data_dir, "anon_annotations_nonhidden")
        csv_file = path.join(data_dir, "anon_dataset_nonhidden_211125.csv")
        self.target_size = target_size

        self.mask_files = glob(mask_dir+"/*.png")
        self.csaw_meta = pd.read_csv(csv_file)

        patients = self.csaw_meta['anon_patientid'].unique()

        self.items = []
        for mask_file in self.mask_files:
            mask_file_name = os.path.basename(mask_file)
            mmimg_file_name = mask_file_name.replace('_mask','').replace('png','dcm')
            mmimg_rows = self.csaw_meta[self.csaw_meta.anon_filename==mmimg_file_name]
            if len(mmimg_rows) >0:
                mmimg_row = mmimg_rows.iloc[0]
                mmimg_path = glob(self.data_dir+'/*/'+mmimg_file_name)
                if len(mmimg_path) >0:
                    mmimg_detail = dict(
                        mmimg_file_name = mmimg_file_name,
                        mmimg_file = mmimg_path[0],
                        mask_file = mask_file,
                        pathology = mmimg_row['x_case'],
                        rad_timing = mmimg_row['rad_timing'] 
                    )
                    self.items.append(mmimg_detail)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        mm_img = pydicom.read_file(self.items[idx]['mmimg_file']).pixel_array
        mask_img = cv2.imread(self.items[idx]['mask_file'], cv2.IMREAD_ANYDEPTH)
        pathology = self.items[idx]['pathology']
        rad_timing = self.items[idx]['rad_timing']


        return self.items[idx]['mmimg_file_name'],mm_img,mask_img,pathology,rad_timing
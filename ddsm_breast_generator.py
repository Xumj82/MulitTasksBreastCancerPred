import os
import cv2
import lmdb
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import path
from glob import glob
from random import sample
from argparse import ArgumentParser
from lib.preprocess_utils import read_resize_img, segment_breast, horizontal_flip,convert_to_8bit,convert_to_16bit,crop_borders,clahe,crop_img
from sklearn.model_selection import train_test_split

def main():
    roi_df =pd.read_csv(args.csv_file)
    full_images = roi_df['image file path'].copy().drop_duplicates().to_numpy()
    crop_board_size = (0,0,0.1,0.1)
    max_pixel_val = 65535
    for full_img_path in full_images:
        img_id = full_img_path.split('/')[1][:-4]
        lesions = roi_df[roi_df['image file path']==full_img_path]
        full_img = read_resize_img(path.join(args.img_dir,full_img_path), target_size =args.target_size,gs_255=True)
        full_img = crop_borders(full_img, border_size=crop_board_size)
        full_img = clahe(full_img, max_pixel_val=max_pixel_val)
        full_images = np.clip(full_img,a_min=max_pixel_val*0.05, a_max=max_pixel_val*0.95)
        full_img,bbox,_ = segment_breast(full_img)
        
        full_mask = np.zeros(full_img.shape)
        roi_areas = []

        for idx,lesion in lesions.iterrows():
            mask = lesion['ROI mask file path']
            type = lesion['abnormality type']
            pathology = lesion['pathology']

            roi_area = {}

            mask_img = read_resize_img(path.join(args.img_dir,mask), target_size =args.target_size,gs_255=True)
            mask_img = crop_borders(mask_img, border_size=(0,0,0.1,0.1))
            mask_img = crop_img(mask_img,bbox) 

            full_mask += mask_img

        full_mask = full_mask.astype(np.uint8)
        full_img = convert_to_16bit(full_images).astype(np.uint16)
        return img_id,type,pathology,img,roi_areas,full_mask


if __name__ == '__main__':
   
    parser = ArgumentParser()
    parser.add_argument('--img_dir',
        default='/mnt/nas4/diskl/MMG/Data/MMG-R1/CSAW_1',
        type=str)
    parser.add_argument('--csv_file',
        default='/mnt/nas4/diskl/MMG/Data/MMG-R1/CSAW_1',
        type=str)
    args = parser.parse_args()

    main(args.verbose)
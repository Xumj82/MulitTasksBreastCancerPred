import os
from tabnanny import verbose
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

def one_hot(type,pathology,pathology_display = True):
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

def main():
    roi_df =pd.read_csv(args.csv_file)
    full_images = roi_df['image file path'].copy().drop_duplicates().to_numpy()
    crop_board_size = (0,0,0.1,0.1)
    max_pixel_val = 65535
    target_height,target_width = (1120, 896)
    env = lmdb.open(args.output_dir+'/ddsm_breast',map_size=1099511627776) 
    

    if args.verbose:
        os.makedirs(args.output_dir+'/ddsm_breast_verbose/', exist_ok=True)


    img_list = []

    with tqdm(total=len(full_images)) as pbar:
        for full_img_path in full_images:
            txn = env.begin(write=True) 
            img_id = full_img_path.split('/')[1][:-4]
            lesions = roi_df[roi_df['image file path']==full_img_path]
            pathology_global = False

            full_img = read_resize_img(path.join(args.img_dir,full_img_path),gs_255=True).astype(np.float32)
            full_img = crop_borders(full_img, border_size=crop_board_size)
            full_img,bbox,breast_mask = segment_breast(full_img)
            full_img_org = full_img

            full_img_org = cv2.resize(full_img_org,dsize=(target_width, target_height))
            full_img = cv2.resize(full_img,dsize=(target_width, target_height))

            full_img = clahe(full_img, max_pixel_val=max_pixel_val)


            # full_images = np.clip(full_img,a_min=max_pixel_val*0.05, a_max=max_pixel_val*0.95)
            
            full_mask = np.zeros(full_img.shape)
            roi_areas = []

            for idx,lesion in lesions.iterrows():
                mask = lesion['ROI mask file path']
                type = lesion['abnormality type']
                pathology = lesion['pathology']
                if pathology == 'MALIGNANT':
                    pathology_global = True

                ann_code = one_hot(type, pathology, True)
                roi_area = {}

                mask_img = read_resize_img(path.join(args.img_dir,mask), gs_255=False)
                mask_img = crop_borders(mask_img, border_size=(0,0,0.1,0.1))
                mask_img = crop_img(mask_img,bbox) 
                mask_img = cv2.resize(mask_img,dsize=(target_width, target_height))
                full_mask = np.where(mask_img>0,ann_code,full_mask)  

            if horizontal_flip(breast_mask):
                full_img_org = cv2.flip(full_img_org, 1)
                full_img = cv2.flip(full_img, 1)
                full_mask = cv2.flip(full_mask, 1)

            full_img = convert_to_16bit(full_img).astype(np.uint16)
            full_mask = full_mask.astype(np.uint8)

            txn.put(key = str(img_id).encode(), value = full_img.tobytes(order='C'))
            txn.put(key = str(img_id+'_mask').encode(), value = full_mask.tobytes(order='C'))
            if args.verbose:
                os.makedirs(args.output_dir+'/ddsm_breast_verbose/{}'.format(str(img_id)), exist_ok=True)
                cv2.imwrite(args.output_dir+'/ddsm_breast_verbose/{}/{}.png'.format(str(img_id),'org'),convert_to_8bit(full_img_org).astype(np.uint8))
                cv2.imwrite(args.output_dir+'/ddsm_breast_verbose/{}/{}.png'.format(str(img_id),'img'),convert_to_8bit(full_img).astype(np.uint8))
                cv2.imwrite(args.output_dir+'/ddsm_breast_verbose/{}/{}.png'.format(str(img_id),'mask'), convert_to_8bit(full_mask).astype(np.uint8))

            img_info = [
                img_id, img_id+str('_mask'),pathology_global
            ]
            img_list.append(img_info)
            txn.commit()
            pbar.update(1)
    
    env.close()
    df = pd.DataFrame(img_list,columns =['img_id','mask_id','label'])
    if args.val_size > 0:
        train_set, test_set = train_test_split(df, test_size=args.test_size, random_state=32)
        train_set.to_csv(args.output_dir+'/seq_lv_train_set.csv', index=False)
        test_set.to_csv(args.output_dir+'/seq_lv_val_set.csv', index=False)
    else:
        df.to_csv(args.output_dir+'/seq_lv_test_set.csv', index=False)


if __name__ == '__main__':
   
    parser = ArgumentParser()
    parser.add_argument('--img_dir',
        default='/mnt/f/datasets/cbis-ddsm-png',
        type=str)
    parser.add_argument('--csv_file',
        default='/mnt/f/datasets/csv/train_roi.csv',
        type=str)
    parser.add_argument('--output_dir',
        default='/mnt/h/datasets/ddsm_breast',
        type=str)
    parser.add_argument('--val_size',
        default=0.1,
        type=float)
    parser.add_argument('--verbose',
        default=True,
        type=bool)
    args = parser.parse_args()

    main()
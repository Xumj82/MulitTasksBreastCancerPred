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

RESIZE = (1120, 896)
MAX_PIXEL_VAL= 65535

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

def get_duo_view(p_row):
    cc_views = p_row[p_row['image view']=='CC']
    mlo_views = p_row[p_row['image view']=='MLO']
    if len(cc_views) >0 and len(mlo_views)>0:
        cc_view = cc_views.iloc[0]
        mlo_view = mlo_views.iloc[0]
        return [cc_view['patient_id'],
        cc_view['image file path'].replace('full_mammogram_images/',''),
        mlo_view['image file path'].replace('full_mammogram_images/',''),
        cc_view['abnormality type'],
        cc_view['pathology']]
    return None

def process_pipeline(img_file_path, verbose_folder=None):
    target_height, target_width = RESIZE
    img = read_resize_img(img_file_path, )

    if verbose_folder:
        img_8u = convert_to_8bit(img)
        cv2.imwrite(verbose_folder+'/origin.png',img_8u)

    img = crop_borders(img,border_size=(0,0,0.06,0.06,))

    if verbose_folder:
        img_8u = convert_to_8bit(img)
        cv2.imwrite(verbose_folder+'/crop.png',img_8u)

    img_segment,_,breast_mask = segment_breast(img)

    if verbose_folder:
        img_8u = convert_to_8bit(img_segment)
        cv2.imwrite(verbose_folder+'/segment.png',img_8u)


    img_clahe = clahe(img_resized, max_pixel_val=MAX_PIXEL_VAL)

    if verbose_folder:
        img_8u = convert_to_8bit(img_clahe)
        cv2.imwrite(verbose_folder+'/flip.png',img_8u)

    if horizontal_flip(breast_mask):
        img_filped = cv2.flip(img_segment, 1)
    else:
        img_filped = img_segment

    if verbose_folder:
        img_8u = convert_to_8bit(img_filped)
        cv2.imwrite(verbose_folder+'/flip.png',img_8u)

    img_clahe = convert_to_16bit(img_clahe).astype(np.uint16)

    img_resized = cv2.resize(img_filped,dsize=(target_width, target_height), 
        interpolation=cv2.INTER_CUBIC)
        
    return img_clahe

def save_to_lmdb(df:pd.DataFrame):
    env = lmdb.open(args.output_dir+'/ddsm_breast',map_size=1099511627776) 
    for dix, row in df.iterrows():
        if args.verbose:
            os.makedirs(args.output_dir+'/ddsm_breast_verbose/'+row['patient_id']+'/', exist_ok=True)
        cc_view = row['cc_view']
        mlo_view = row['mlo_view']
        cc_img = process_pipeline(cc_view)
        mlo_img = process_pipeline(mlo_view)

def main():
    roi_df =pd.read_csv(args.csv_file)
    full_images = roi_df['image file path'].copy().drop_duplicates().to_numpy()
    patients = roi_df['patient_id'].unique().tolist()
    duo_view_list = []
    print('re-arrage breast data...')
    with tqdm(total=len(patients)) as pbar:
        for p in patients:
            p_row_right = roi_df[(roi_df['patient_id']==p)&(roi_df['left or right breast']=='RIGHT')]
            p_row_left = roi_df[(roi_df['patient_id']==p)&(roi_df['left or right breast']=='LEFT')]

            rigth_views = get_duo_view(p_row_right)
            if rigth_views:
                duo_view_list.append(rigth_views)
            
            left_views = get_duo_view(p_row_left)
            if left_views:
                duo_view_list.append(left_views)

            pbar.update(1)

    df = pd.DataFrame(duo_view_list, columns =['patient_id', 'cc_view','mlo_view','type','pathology'])

    # df.to_csv(args.output_dir+'/seq_lv_train_set.csv',index=False)

   
    
    if args.verbose:
        os.makedirs(args.output_dir+'/ddsm_breast_verbose/', exist_ok=True)

    if args.val_size > 0:
        train_set, test_set = train_test_split(df, test_size=args.val_size, random_state=32)
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
        default=False,
        type=bool)
    args = parser.parse_args()

    main()
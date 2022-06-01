import os
import cv2
import lmdb
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from random import sample
from argparse import ArgumentParser
from lib.preprocess_utils import read_resize_img, segment_breast, horizontal_flip,convert_to_8bit,convert_to_16bit

SEED = 32
RESIZE = (1120, 896)
random.seed(SEED)
def process_pipeline(img_file_path):
    target_height, target_width = RESIZE
    img = read_resize_img(img_file_path, crop_borders_size=(0.06,0.06,0,0))
    img_segment,_,breast_mask = segment_breast(img)
    img_filped = horizontal_flip(img_segment,breast_mask)
    img_resized = cv2.resize(img_filped,dsize=(target_width, target_height), 
            interpolation=cv2.INTER_CUBIC)
    img_resized = convert_to_16bit(img_resized)
    return img_resized

def main(verbose):
    csaw_df = pd.read_csv(args.csv_file_path)
    patients = csaw_df['anon_patientid'].unique().tolist()
    selected_patients = random.sample(patients,int(args.sample_rate * len(patients)))
    env = lmdb.open(args.output_dir+'/csaw_set',map_size=1099511627776) 
    
    if verbose:
        os.makedirs(args.output_dir+'/csaw_set_verbose', exist_ok=True)

    with tqdm(total=len(selected_patients)) as pbar:
        for p in selected_patients:
            txn = env.begin(write=True) 
            p_rows = csaw_df[csaw_df['anon_patientid']==p]
            for idx,row in p_rows.iterrows():
                img_file_name = row['anon_filename']
                img_file_path = glob(args.img_file_dir+'/*/'+img_file_name)
                if len(img_file_path)<=0:
                    continue
                img = process_pipeline(img_file_path[0])
                txn.put(key = str(img_file_name).encode(), value = img.tobytes(order='C'))
                if verbose:
                    img_8u = convert_to_8bit(img)
                    cv2.imwrite(args.output_dir+'/csaw_set_verbose/'+img_file_name.replace('dcm','png'),img_8u)
                # pd.concat([patients_sample,row])
            txn.commit() 
            pbar.update(1)
    patients_sample = csaw_df[csaw_df['anon_patientid'].isin(selected_patients)]
    patients_sample.to_csv(args.output_dir+'/selected_csaw_set.csv', index=False)
    
    env.close() 
        

if __name__ == '__main__':
   
    parser = ArgumentParser()
    parser.add_argument('--img_file_dir',
        help='data csv file path',
        default='/mnt/nas4/diskl/MMG/Data/MMG-R1/CSAW_1',
        type=str)
    parser.add_argument('--csv_file_path',
        help='data csv file path',
        default='/mnt/nas4/diskl/MMG/Data/MMG-R1/CSAW_1/anon_dataset_nonhidden_211125.csv',
        type=str)
    parser.add_argument('--sample_rate',
        help='from 0 to 1',
        default=0.01,type=int)
    parser.add_argument('--output_dir',default='/mnt/nas4/diskl/MMG/Data/MMG-R1/SAMPLE_DATA',type=str)
    parser.add_argument('--verbose',help='generate png file',default=True,type=bool)
    args = parser.parse_args()

    main(args.verbose)
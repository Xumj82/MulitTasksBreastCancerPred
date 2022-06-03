from cProfile import label
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
from lib.preprocess_utils import read_resize_img, segment_breast, horizontal_flip,convert_to_8bit,convert_to_16bit,crop_borders

SEED = 32
RESIZE = (1120, 896)
random.seed(SEED)
def process_pipeline(img_file_path):
    target_height, target_width = RESIZE
    img = read_resize_img(img_file_path, )
    img = crop_borders(img,border_size=(0,0,0.06,0.06,))
    img_segment,_,breast_mask = segment_breast(img)
    img_filped = horizontal_flip(img_segment,breast_mask)
    img_resized = cv2.resize(img_filped,dsize=(target_width, target_height), 
            interpolation=cv2.INTER_CUBIC)
    img_resized = convert_to_16bit(img_resized).astype(np.uint16)
    return img_resized



def main(verbose):
    csaw_df = pd.read_csv(args.csv_file_path)
    # fill missing data
    csaw_df['x_cancer_laterality'] = csaw_df['x_cancer_laterality'].fillna(value='NULL')
    csaw_df['x_type'] = csaw_df['x_type'].fillna(value=0)
    csaw_df['x_lymphnode_met'] = csaw_df['x_lymphnode_met'].fillna(value=-1)
    csaw_df['rad_timing'] = csaw_df['rad_timing'].fillna(value=4)
    csaw_df['rad_r1'] = csaw_df['rad_r1'].fillna(value=0)
    csaw_df['rad_r2'] = csaw_df['rad_r2'].fillna(value=0)
    csaw_df['rad_recall_type_right'] = csaw_df['rad_recall_type_right'].fillna(value=0)
    csaw_df['rad_recall_type_left'] = csaw_df['rad_recall_type_left'].fillna(value=0)
    
    ## cancer breasts
    # cancer_breasts = csaw_df[csaw_df['x_cancer_laterality']==csaw_df['imagelaterality']]


    patients = csaw_df['anon_patientid'].unique().tolist()
    # selected_patients = random.sample(patients,int(args.sample_rate * len(patients)))
    # env = lmdb.open(args.output_dir+'/csaw_set',map_size=1099511627776) 


    cancer_breasts = [] #['id','cc_view','mlo_view','rad_time']
    healthy_breasts = [] #['id','cc_view','mlo_view','rad_time']

    cancer_exam = [] # ['id','r_cc_view','r_mlo_view','l_cc_view','l_mlo_view','rad_time']
    healthy_exam = [] # ['id','r_cc_view','r_mlo_view','l_cc_view','l_mlo_view','rad_time']

    print('re-arrange annotation file')
    with tqdm(total=len(patients)) as pbar:
        for p in patients:
            p_rows = csaw_df[csaw_df['anon_patientid']==p]

            years = p_rows['exam_year'].unique()
            for y in years:
                y_rows = p_rows[p_rows['exam_year']==y]
                right_cc_row = y_rows[(y_rows['imagelaterality']=='Right')&(y_rows['viewposition']=='CC')]
                right_mlo_row = y_rows[(y_rows['imagelaterality']=='Right')&(y_rows['viewposition']=='MLO')]
                right_gt_label = y_rows['rad_timing'].max() if y_rows['x_cancer_laterality'].iloc[0]=='Right' else 4.0
                right_info = [str(p)+'_'+str(y)+'_'+'Right',
                    right_cc_row.iloc[0]['anon_filename'],
                    right_mlo_row.iloc[0]['anon_filename'],
                    right_gt_label]

                left_cc_row = y_rows[(y_rows['imagelaterality']=='Left')&(y_rows['viewposition']=='CC')]
                left_mlo_row = y_rows[(y_rows['imagelaterality']=='Left')&(y_rows['viewposition']=='MLO')]
                left_gt_label = y_rows['rad_timing'].max() if y_rows['x_cancer_laterality'].iloc[0]=='Left' else 4.0
                left_info = [str(p)+'_'+str(y)+'_'+'Light',
                    left_cc_row.iloc[0]['anon_filename'],
                    left_mlo_row.iloc[0]['anon_filename'],
                    left_gt_label]

                exam_gt_label = y_rows['rad_timing'].max()
                exam_info = [str(p)+'_'+str(y),
                    right_cc_row.iloc[0]['anon_filename'],
                    right_mlo_row.iloc[0]['anon_filename'],
                    left_cc_row.iloc[0]['anon_filename'],
                    left_mlo_row.iloc[0]['anon_filename'],
                    exam_gt_label]

                if right_gt_label == 4.0:
                    healthy_breasts.append(right_info)
                else:
                    cancer_breasts.append(right_info)
                
                if left_gt_label == 4.0:
                    healthy_breasts.append(left_info)
                else:
                    cancer_breasts.append(left_info)

                if exam_gt_label == 4.0:
                    healthy_exam.append(exam_info)
                else:
                    cancer_exam.append(exam_info)
                    # files = [cc_row.iloc[0]['anon_filename'], mlo_row.iloc[0]['anon_filename']]
            pbar.update(1)

    if args.sample_level == 0:
        # total_cases_amount = len(cancer_exam)+len(healthy_exam)
        cancer_cases_amount = int(len(cancer_exam)*args.sample_rate)
        healthy_cases_amount = int(cancer_cases_amount/args.cancer_rate*(1-args.cancer_rate))

        select_cancer_cases = random.sample(cancer_exam,cancer_cases_amount)
        select_healthy_cases = random.sample(healthy_exam,healthy_cases_amount)
        select_case = select_cancer_cases+select_healthy_cases
        output_selected(select_case, 'csaw_exam_lv_set')
        df = pd.DataFrame(select_case,columns =['id','r_cc_view','r_mlo_view','l_cc_view','l_mlo_view','rad_time'])
        df.to_csv(args.output_dir+'/exam_lv_csaw_set.csv', index=False)

    if args.sample_level == 1:
        # total_cases_amount = len(cancer_breasts)+len(healthy_breasts)
        cancer_cases_amount = int(len(cancer_breasts)*args.sample_rate)
        healthy_cases_amount = int(cancer_cases_amount/args.cancer_rate*(1-args.cancer_rate))

        select_cancer_cases = random.sample(cancer_breasts,cancer_cases_amount)
        select_healthy_cases = random.sample(healthy_breasts,healthy_cases_amount)
        select_case = select_cancer_cases+select_healthy_cases
        output_selected(select_case, 'csaw_breast_lv_set')
        df = pd.DataFrame(select_case,columns =['id','cc_view','mlo_view','rad_time'])
        df.to_csv(args.output_dir+'/breast_lv_csaw_set.csv', index=False)
        # for p in selected_patients:
        #     txn = env.begin(write=True) 
        #     p_rows = csaw_df[csaw_df['anon_patientid']==p]
        #     for idx,row in p_rows.iterrows():
                
        #         years = row['exam_year'].unique()

        #         img_file_name = row['anon_filename']
        #         img_file_path = glob(args.img_file_dir+'/*/'+img_file_name)
        #         if len(img_file_path)<=0:
        #             continue
        #         img = process_pipeline(img_file_path[0])
        #         txn.put(key = str(img_file_name).encode(), value = img.tobytes(order='C'))
        #         if verbose:
        #             img_8u = convert_to_8bit(img)
        #             cv2.imwrite(args.output_dir+'/csaw_set_verbose/'+img_file_name.replace('dcm','png'),img_8u)
        #         # pd.concat([patients_sample,row])
        #     txn.commit() 
        #     pbar.update(1)
    # patients_sample = csaw_df[csaw_df['anon_patientid'].isin(selected_patients)]
    # patients_sample.to_csv(args.output_dir+'/selected_csaw_set.csv', index=False)
    
    # env.close() 
def output_selected(select_list, lmdb_name):
    if args.verbose:
        os.makedirs(args.output_dir+'/'+lmdb_name+'_verbose/', exist_ok=True)
    env = lmdb.open(args.output_dir+'/'+lmdb_name,map_size=1099511627776) 
    with tqdm(total=len(select_list)) as pbar:
        for case in select_list:
            txn = env.begin(write=True) 
            id = case[0]
            files = case[1:-1]
            label = case[-1]
            for f in files:
                img_file_path = glob(args.img_file_dir+'/*/'+f)
                if len(img_file_path)<=0:
                    continue
                img = process_pipeline(img_file_path[0])
                txn.put(key = str(id).encode(), value = img.tobytes(order='C'))
                if args.verbose:
                    img_8u = convert_to_8bit(img)
                    cv2.imwrite(args.output_dir+'/'+lmdb_name+'_verbose/'+id+'.png',img_8u)
                # pd.concat([patients_sample,row])
            txn.commit() 
            pbar.update(1)
    
    env.close()
    # df = pd.DataFrame(select_list,columns =['Names'])
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
    parser.add_argument('--sample_level',
        help='exam case level (0) or just breast level(1)',
        default=0,
        type=int)
    parser.add_argument('--sample_rate',
        help='from 0 to 1, how many cancar cases selected from the whole cancer cases',
        default=1,type=int)
    parser.add_argument('--cancer_rate',
        help='from 0 to 1, cancer rate in the whole selected dataset',
        default=0.5,type=int)
    parser.add_argument('--output_dir',default='/mnt/nas4/diskl/MMG/Data/MMG-R1/AMPLE_DATA_EXAM/',type=str)
    parser.add_argument('--verbose',help='generate png file',default=True,type=bool)
    args = parser.parse_args()

    main(args.verbose)
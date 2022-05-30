import os
import csv 
import pandas as pd
import cv2

import inspect
import importlib
from argparse import ArgumentParser

from tqdm import tqdm
from data.csaw_set import CsawSet
from data.ddsm_set import DdsmSet


def main():

    if args.data_name == 'ddsm_set':
        # dataset = DdsmSet(args.data_root,dataset_stat = True)
        # generate_img_mask_pairs(DdsmSet(args.data_root,train=True,dataset_stat = True),args.data_name,train=False)
        generate_img_mask_pairs(DdsmSet(args.data_root,train=False,dataset_stat = True),args.data_name,train=False)
    if args.data_name == 'csaw_set':
        dataset = CsawSet(args.data_root)
        generate_img_mask_pairs(dataset,args.data_name,train=True)
    

def generate_img_mask_pairs(data_set,dataset_name,train = True):

    if train:
        type = 'train'
    else:
        type = 'test'

    # roi_file = pd.read_csv(os.path.join(args.data_root,'csv/'+type+'_roi.csv'))

    # data_set = DDSMSet(os.path.join(args.data_root,'cbis-ddsm-png/'),roi_file,target_size=(2000,3000))
    # data_set_module = load_data_module(dataset_name)
    # data_set = data_set_module(args.data_root)


    if not os.path.exists(os.path.join(args.data_root,dataset_name,'img_dir',type)):
        os.makedirs(os.path.join(args.data_root,dataset_name,'img_dir',type))

    if not os.path.exists(os.path.join(args.data_root,dataset_name,'ann_dir',type)):
        os.makedirs(os.path.join(args.data_root,dataset_name,'ann_dir',type))

    # with open(os.path.join(args.data_root,'ddsm_dataset/img_dir/',type+'_meta.csv'),'a') as fd:
    #     fd.write('img_id,pathology')


 
    fields=['img_id','pathology','rad_time']
    with open(os.path.join(args.data_root,dataset_name,type+'_meta.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    with open(os.path.join(args.data_root,dataset_name,type+'_meta.csv'), 'a') as f:
        writer = csv.writer(f)
        with tqdm(total=len(data_set)) as pbar:
            for i in range(len(data_set)):
                img_id,full_img,ann_img,pathology,rad_time = data_set[i]
                img_path = os.path.join(args.data_root,dataset_name,'img_dir',type,'{}.png').format(img_id)
                ann_path = os.path.join(args.data_root,dataset_name,'ann_dir',type,'{}.png').format(img_id)
                cv2.imwrite(img_path,full_img)
                cv2.imwrite(ann_path,ann_img)
                writer.writerow([img_id,pathology,rad_time])
                # meta_file.write('{},{}\n'.format(img_id,pathology))
                pbar.update(1)

    total_mean, total_var, total_std, pmax, pmin = data_set.get_dataset_stat()
    print("mean:{}, var:{}, std:{}, max:{}, min{}".format(total_mean, total_var, total_std,pmax,pmin))
    f = open(os.path.join(args.data_root,dataset_name,type+'_stat.txt'), "w")
    f.write("mean:{}, var:{}, std:{}, max:{}, min{}".format(total_mean, total_var, total_std,pmax,pmin))
    f.close()

def load_data_module(dataset_name,package = 'data'):
    name = dataset_name
    # Change the `snake_case.py` file name to `CamelCase` class name.
    # Please always name your model file name as `snake_case.py` and
    # class name corresponding `CamelCase`.
    camel_name = ''.join([i.capitalize() for i in name.split('_')])
    try:
       return getattr(importlib.import_module(
            '.'+name, package=package), camel_name)
    except Exception as es:
        raise ValueError(es)
        # raise ValueError(
        #     f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')


if __name__ == '__main__':
   
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--data_name',default='ddsm_set',type=str)
    parser.add_argument('--data_root',default='/media/xumingjie/study/datasets/',type=str)
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = os.environ['data_root']
    print('DATA_ROOT:',args.data_root)
    
    # args.data_root = os.path.expanduser(args.data_root)
    main()
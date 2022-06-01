import random
import pandas as pd
import numpy as np
from argparse import ArgumentParser

def main():
    



if __name__ == '__main__':
   
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--img_dir',default='/home/xumingjie/dataset/ddsm_coco',type=str)
    parser.add_argument('--output_dir',default='/mnt/hdd/datasets/',type=str)
    args = parser.parse_args()
    
    # args.data_root = os.path.expanduser(args.data_root)
    main()


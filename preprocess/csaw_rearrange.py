import os
from os import path
from glob import glob

import pandas as pd
os.chdir('../')
from lib.csaw_utils import get_exam_level_meta, get_patient_level_meta 

CSAW_DIR = '/mnt/nas4/diskl/MMG/Data/MMG-R1/CSAW_1'
META_FILE = 'anon_dataset_nonhidden_211125.csv'

def main():
    csaw_meta = pd.read_csv(path.join(CSAW_DIR, META_FILE)) 


if __name__ == '__main__':
    # parser = ArgumentParser()
    
    main()


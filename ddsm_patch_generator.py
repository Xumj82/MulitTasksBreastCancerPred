import os
import pandas as pd
from argparse import ArgumentParser
from preprocess.patchset import PatchSet
from sklearn.model_selection import train_test_split

def main(args):
    train_roi = pd.read_csv(os.path.join(args.csv_dir, 'train_roi.csv'))
    test_roi = pd.read_csv(os.path.join(args.csv_dir, 'test_roi.csv'))

    # if not os.path.exists(os.path.join(args.output_dir, 'train/')):
    #     os.makedirs(os.path.join(args.output_dir, 'train/'))

    train_set = PatchSet(
       
        args.data_dir+'/cbis-ddsm-png/',
        train_roi,
        target_size=(1120,896),
        out_dir=os.path.join(args.output_dir),
        out_csv=os.path.join(args.output_dir,'train_meta.csv'),
        number_positive = args.number_positive,
        number_negative= args.number_negative,
        number_hard_bkg=args.number_hard_negative, 
        patch_size=args.patch_size
    )

    # if not os.path.exists(os.path.join(args.output_dir, 'test/')):
    #     os.makedirs(os.path.join(args.output_dir, 'test/'))

    test_set = PatchSet( 
        args.data_dir+'/cbis-ddsm-png/',
        test_roi,
        target_size=(1120,896),
        out_dir=os.path.join(args.output_dir),
        out_csv=os.path.join(args.output_dir,'test_meta.csv'),
        number_positive = args.number_positive,
        number_negative= args.number_negative,
        number_hard_bkg=args.number_hard_negative, 
        patch_size=args.patch_size
    )
    
    train_set.get_all_patches(verbose=True)
    test_set.get_all_patches(verbose=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    # parser.add_argument('--ddsm_root', default="/media/xumingjie/study/datasets/cbis-ddsm-png/", type=str)
    parser.add_argument('--data_dir', default='/mnt/f/datasets/', type=str)
    parser.add_argument('--patch_size', default=224, type=int)
    parser.add_argument('--csv_dir', default='/mnt/f/datasets/csv/', type=str)
    # parser.add_argument('--mdb_dir', default='/media/xumingjie/study/datasets/mdb/', type=str)
    parser.add_argument('--output_dir', default='/mnt/c/Users/11351/Desktop/patch_set_1150_224/', type=str)
    # parser.add_argument('--output_csv', default='patch_images', type=str)
    parser.add_argument('--number_positive', default=20, type=int)
    parser.add_argument('--number_negative', default=10, type=int)
    parser.add_argument('--number_hard_negative', default=10, type=int)
    args = parser.parse_args()

    # args.ddsm_root = os.path.expanduser(args.ddsm_root)
    
    main(args)
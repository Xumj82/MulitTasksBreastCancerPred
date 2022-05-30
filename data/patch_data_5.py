import torch
import lmdb
import os.path as op
import pandas as pd
import numpy as np
import pickle as pkl
import elasticdeform
import torch.utils.data as data

from torchvision import transforms
from sklearn.model_selection import train_test_split

# class ElasticDeform(torch.nn.Module):
#     def __init__(self, sigma=15, points=3,axis=(0)):
#         super().__init__()
#         self.sigma = sigma
#         self.point = points
#         self.axis = axis

#     def forward(self, img):
#         img = elasticdeform.deform_random_grid(img,sigma=self.sigma, points=self.point,axis=self.axis)
#         return img

class PatchData5(data.Dataset):
    def __init__(self,csv_file,lmdb_file, 
                 data_dir=r'data/ref',
                 patch_size =112,
                 class_name = ['bkg','mass_b','calcification_b','mass_m','calcification_m'],
                 class_num=5,
                 train=True,
                 no_augment=True,
                 aug_prob=0.5,
                 img_mean=(0.485),
                 img_std=(0.229)):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.aug = train and not no_augment
        self.label_dict = {}
        self.check_files()

    @staticmethod
    def train_test_split_on_patient(data_list ,test_size, random_state):
        patients = []
        for index, row in data_list.iterrows():
            patientid = row['img_id'].split('_')[1] + row['img_id'].split('_')[2]
            patients.append(patientid)
        data_list['PatientID'] = patients

        patientdict = list( dict.fromkeys(patients) )
        traindict, testdict = train_test_split(patientdict, test_size=test_size, random_state=random_state)
        traindict = pd.DataFrame(data=traindict, columns=["PatientID"])
        testdict = pd.DataFrame(data=testdict, columns=["PatientID"])
        data_train = pd.merge(traindict, data_list,  on ='PatientID', how ='left')
        data_test = pd.merge(testdict, data_list,  on ='PatientID', how ='left')
        return data_train,data_test

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value. 
        file_list_path = op.join(self.data_dir, self.csv_file)
        df = pd.read_csv(file_list_path)

        fl_train, fl_val = self.train_test_split_on_patient(
            df, test_size=0.2, random_state=32)
        
        self.data_list = fl_train if self.train else fl_val

        file_lmdb_path = op.join(self.data_dir, self.lmdb_file)
        self.env = lmdb.open(file_lmdb_path, map_size=1099511627776)
        self.txn = self.env.begin(write=True)

        for idx,name in enumerate(self.class_name):
          self.label_dict[name] = idx

    def __len__(self):
        return len(self.data_list)

    def to_one_hot(self, idx):
        out = np.zeros(self.class_num, dtype=float)
        out[idx] = 1
        return out

    def __getitem__(self, idx):
        img_id = self.data_list.iloc[idx]['patch_id']
        img = self.txn.get(img_id.encode())
        img = np.frombuffer(img, dtype=np.uint16).reshape(self.patch_size,self.patch_size).astype(np.float64)
        if self.train:
            img = elasticdeform.deform_random_grid(img,sigma=15, points=3,axis=(0,1))
        img = torch.unsqueeze(torch.from_numpy(img/65536), 0).float()

        if str(self.data_list.iloc[idx]['pathology']) == 'NORMAL':
            labels = self.to_one_hot(0)
        elif str(self.data_list.iloc[idx]['pathology']) == 'BENIGN' or str(self.data_list.iloc[idx]['pathology']) == 'BENIGN_WITHOUT_CALLBACK':
            labels = self.to_one_hot(1) if str(self.data_list.iloc[idx]['type']) == 'calcification' else self.to_one_hot(2)
        else:
            labels = self.to_one_hot(3) if str(self.data_list.iloc[idx]['type']) == 'calcification' else self.to_one_hot(4)

        labels =torch.from_numpy(labels).float()
              

        trans = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(self.aug_prob),
            transforms.RandomVerticalFlip(self.aug_prob),
            transforms.RandomRotation(10),
            transforms.Normalize(self.img_mean, self.img_std)
        ) if self.train else torch.nn.Sequential(
            transforms.Normalize(self.img_mean, self.img_std)
        )

        img_tensor = trans(img)

        return img_tensor, labels, img_id
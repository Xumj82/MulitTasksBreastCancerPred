# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This main entrance of the whole project.
    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from model import Segement_Net
from data import DInterface
from  lib.train_utils import load_model_path_by_args
from lib.config import get_config


def parse_option():
    parser = ArgumentParser('Breast cancer models training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    args, _ = parser.parse_known_args()
    config = get_config(args)
    return config

def load_callbacks(cfg):
    callbacks = []
    # callbacks.append(plc.EarlyStopping(
    #     monitor='val_acc',
    #     mode='max',
    #     patience=10,
    #     min_delta=0.001
    # ))
    callbacks.append(plc.ModelCheckpoint(
        monitor=cfg['acc_monitor'],
        filename='best-{epoch:02d}-{'+cfg['acc_monitor']+':.3f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ))

    # if cfg['lr_scheduler']:
    callbacks.append(plc.LearningRateMonitor(
        logging_interval='step'))
        
    return callbacks

def train(cfg):
    pl.seed_everything(cfg['SEED'])
    load_path = load_model_path_by_args(cfg)
    data_module = DInterface(**cfg['DATA'])

    # if load_path is None:
    #     model = MInterface(**vars(args))
    # else:
    #     model = MInterface(**vars(args)).load_from_checkpoint(load_path)
    if load_path is None:
        model = Segement_Net(**cfg['MODEL'])
    else:
        model = Segement_Net(cfg).load_from_checkpoint(load_path) 

    # # If you want to change the logger's saving folder
    logger = TensorBoardLogger(save_dir='logs', name=cfg['TRAIN']['log_dir'])
    cfg['TRAIN']['Trainer']['callbacks'] = load_callbacks(cfg['TRAIN'])
    cfg['TRAIN']['Trainer']['logger'] = logger

    trainer = Trainer(**cfg['TRAIN']['Trainer'])
    trainer.fit(model, data_module)

def eval(cfg):
    pl.seed_everything(cfg['SEED'])
    load_path = load_model_path_by_args(cfg)
    data_module = DInterface(**cfg['DATA'])

    # if load_path is None:
    #     model = MInterface(**vars(args))
    # else:
    #     model = MInterface(**vars(args)).load_from_checkpoint(load_path)
    if load_path is None:
        model = Segement_Net(**cfg['MODEL'])
    else:
        model = Segement_Net(cfg).load_from_checkpoint(load_path) 

    # # If you want to change the logger's saving folder
    logger = TensorBoardLogger(save_dir='logs', name=cfg['TRAIN']['log_dir'])
    cfg['TRAIN']['Trainer']['callbacks'] = load_callbacks(cfg['TRAIN'])
    cfg['TRAIN']['Trainer']['logger'] = logger

    trainer = Trainer(**cfg['TRAIN']['Trainer'])
    trainer.test(model, data_module)

if __name__ == '__main__':
    config = parse_option()
    if config['MODE'] == 'Train':
        train(config)
    if config['MODE'] == 'Test':
        train(config)
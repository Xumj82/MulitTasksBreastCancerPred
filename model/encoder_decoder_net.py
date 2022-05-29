from cProfile import label
import inspect
import importlib
import mmseg

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


import pytorch_lightning as pl
import torch.optim.lr_scheduler as lrs
import torchmetrics
from sklearn.metrics import accuracy_score
import metrics
from lib.train_utils import load_pretrained
from mmseg.models import losses
from mmcv.runner import BaseModule
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        target = target.long()
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class SegmentationModuleBase(pl.LightningModule):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    @staticmethod
    def pixel_acc(pred, label, ignore_index=-1):
        _, preds = torch.max(pred, dim=1)
        valid = (label != ignore_index).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    @staticmethod
    def part_pixel_acc(pred_part, gt_seg_part, gt_seg_object, object_label, valid):
        mask_object = (gt_seg_object == object_label)
        _, pred = torch.max(pred_part, dim=1)
        acc_sum = mask_object * (pred == gt_seg_part)
        acc_sum = torch.sum(acc_sum.view(acc_sum.size(0), -1), dim=1)
        acc_sum = torch.sum(acc_sum * valid)
        pixel_sum = torch.sum(mask_object.view(mask_object.size(0), -1), dim=1)
        pixel_sum = torch.sum(pixel_sum * valid)
        return acc_sum, pixel_sum 

    @staticmethod
    def part_loss(pred_part, gt_seg_part, gt_seg_object, object_label, valid):
        mask_object = (gt_seg_object == object_label)
        loss = F.nll_loss(pred_part, gt_seg_part * mask_object.long(), reduction='none')
        loss = loss * mask_object.float()
        loss = torch.sum(loss.view(loss.size(0), -1), dim=1)
        nr_pixel = torch.sum(mask_object.view(mask_object.shape[0], -1), dim=1)
        sum_pixel = (nr_pixel * valid).sum()
        loss = (loss * valid.float()).sum() / torch.clamp(sum_pixel, 1).float()
        return loss

class ModelBuilder:
    def __init__(self,**kargs):
        # self.kargs = kargs
        self.__dict__ = kargs
        pass

    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)
    @staticmethod
    def build_net(block,net_cfg):
        net = load_model(block,net_cfg)
        if hasattr(net, 'weights') and net.weights:
            net.init_weights()
        return net

def load_model(package,net_cfg)->BaseModule:
    name = net_cfg['name']
    if 'args' in net_cfg.keys():
        args = net_cfg['args']
    else:
        args = dict()
    # Change the `snake_case.py` file name to `CamelCase` class name.
    # Please always name your model file name as `snake_case.py` and
    # class name corresponding `CamelCase`.
    camel_name = ''.join([i.capitalize() for i in name.split('_')])
    try:
        Model = getattr(importlib.import_module(
            '.'+package+'.'+name, package='model'), camel_name)
    except Exception as e:
        print(e)
        raise ValueError(
            f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
    return instancialize(Model,args)

def instancialize(Model, inkeys):
    """ Instancialize a model using the corresponding parameters
        from self.hparams dictionary. You can also input any args
        to overwrite the corresponding value in self.hparams.
    """
    class_args = inspect.getargspec(Model.__init__).args[1:]
    args1 = {}
    for arg in class_args:
        if arg in inkeys:
            args1[arg] = inkeys[arg]
    return Model(**args1)

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups



class Segement_Net(pl.LightningModule):
    def __init__(self, 

                **kargs
                ):
        
        super().__init__()
        self.save_hyperparameters()
        # self.lr_decay_steps = self.hparams.lr_decay_steps
        # self.lr_decay_rate =self.hparams.lr_decay_rate
        # self.automatic_optimization = False
        builder = ModelBuilder(**kargs)
        # self.nets = dict() 
        
        self.nets = []
        for k in  self.hparams.blocks.keys():
            block = builder.build_net(k, self.hparams.blocks[k])
            self.__setattr__(k, block)
            self.nets.append(k)

        self.metric_funcs = dict()
        self.loss_funcs = nn.ModuleDict()
        self.configure_loss()
        self.configure_metrics()
        # self.net_encoder = builder.build_encoder()
        # self.net_decoder = builder.build_decoder()

    def forward(self,net_input):
        for net in self.nets:
            net_output = self.__getattr__(net)(net_input)
            net_input = net_output
        return net_output

    def training_step(self, batch, batch_idx, optimizer_idx = None):
        input = batch['input']
        out = self(input)

        loss_dict = dict()
        for idx ,k in enumerate(self.loss_funcs.keys()):
            # if k == 'lesion':
            #     loss_dict[k+'_loss'] = self.hparams.loss_scale[k]* self.loss_funcs[k](out[k], batch[k].long())
            #     continue 
            loss_dict[k+'_loss'] = self.hparams.loss_scale[k]* self.loss_funcs[k](out[k], batch[k])
        loss = sum(loss_dict.values())
        self.log_dict(loss_dict,on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input = batch['input']
        out = self(input)

        loss_dict = dict()
        for idx ,k in enumerate(self.loss_funcs.keys()):
            loss_dict['val_'+k+'_loss'] = self.hparams.loss_scale[k]* self.loss_funcs[k](out[k], batch[k])
            self.metric_funcs[k].update(out[k].cpu(), batch[k].cpu())
        loss = sum(loss_dict.values())
        self.log_dict(loss_dict,on_step=True, on_epoch=False, prog_bar=True)

        return loss
        

    def test_step(self, batch, batch_idx):
        self.validation_step(self, batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        metric_dict = dict()
        for k in self.metric_funcs.keys():
            metric_dict['val_'+k+'_acc'] = self.metric_funcs[k].compute()
        self.log_dict(metric_dict,on_step=False, on_epoch=True, prog_bar=False)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_loss(self):
        for k in self.hparams.losses.keys():
            if self.hparams.losses[k] == 'crossentropy':
                self.loss_funcs.add_module(k, nn.CrossEntropyLoss())
            if self.hparams.losses[k] == 'mse':
                self.loss_funcs.add_module(k, nn.MSELoss())
            if self.hparams.losses[k] == 'focal':
                self.loss_funcs.add_module(k, losses.FocalLoss(gamma=2.0, alpha=0.25))

    def configure_metrics(self):
        for k in self.hparams.metrics.keys():
            if self.hparams.metrics[k] == 'f1score':
                self.metric_funcs[k] = torchmetrics.F1Score()
            if self.hparams.metrics[k] == 'accuracy':
                self.metric_funcs[k] = torchmetrics.Accuracy()
            if self.hparams.metrics[k] == 'precision':
                self.metric_funcs[k] = torchmetrics.Precision()
            if self.hparams.metrics[k] == 'cosine':
                self.metric_funcs[k] = torchmetrics.CosineSimilarity()

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        for net_name in self.hparams.blocks:
            net_cfg = self.hparams.blocks[net_name]
            optimizer = None
            scheduler = None
            if 'optimizer' in net_cfg.keys():
                if net_cfg['optimizer'] == 'SGD':
                    # nets =self.nets.items()[self.nets.keys()[0]]
                    optimizer = torch.optim.SGD(
                        self.__getattr__(net_name).parameters(),
                        lr= net_cfg['learning_rate'],
                        momentum=net_cfg['beta1'],
                        weight_decay=net_cfg['weight_decay']
                    )
                if net_cfg['optimizer'] == 'Adam':
                    optimizer = torch.optim.Adam(
                        self.__getattr__(net_name).parameters(),
                        lr= net_cfg['learning_rate'],
                        weight_decay=net_cfg['weight_decay']
                    )           

            if 'scheduler' in net_cfg.keys():
                if net_cfg['scheduler'] == 'step' and optimizer:
                    scheduler = lrs.StepLR( optimizer,
                                            step_size=net_cfg['lr_decay_steps'],
                                            gamma=net_cfg['lr_decay_rate'])
                # if sched == 'cosine':
                #     scheduler = lrs.CosineAnnealingLR(
                #                 optimizer[idx],
                #                 T_max=self.hparams.lr_decay_steps[idx],
                #                 eta_min=self.hparams.lr_decay_min_lr[idx])
            if optimizer:
                optimizers.append(optimizer)
            if scheduler:
                schedulers.append(scheduler)

        # for idx, sched in enumerate(self.hparams.schedulers):
        #     if sched is not None:
        #         if sched == 'step':
        #             scheduler = lrs.StepLR(optimizer[idx],
        #                                     step_size=self.hparams.lr_decay_steps[idx],
        #                                     gamma=self.hparams.lr_decay_rate[idx])
        #         if sched == 'cosine':
        #             scheduler = lrs.CosineAnnealingLR(optimizer[idx],
        #                         T_max=self.hparams.lr_decay_steps[idx],
        #                         eta_min=self.hparams.lr_decay_min_lr[idx])
        #         schedulers.append(scheduler)
        # if self.hparams.lr_encoder:
        #     optimizer_encoder = torch.optim.SGD(
        #         self.net_encoder.parameters(),
        #         # lr = 1e-3,
        #         lr= self.hparams.lr_encoder,
        #         momentum=self.hparams.beta1,
        #         weight_decay=weight_decay)
        #     optimizers.append(optimizer_encoder)

        # if self.hparams.lr_decoder:
        #     optimizer_decoder = torch.optim.SGD(
        #         group_weight(self.net_decoder),
        #         # lr = 1e-3,
        #         lr= self.hparams.lr_encoder,
        #         momentum=self.hparams.beta1,
        #         weight_decay=weight_decay)
        #     optimizers.append(optimizer_decoder)

        # if self.hparams.lr_scheduler == 'step':
        #     if self.hparams.lr_encoder:
        #         scheduler_encoder = lrs.StepLR(optimizer_encoder,
        #                                 # step_size=5,
        #                                 step_size=self.hparams.lr_decay_steps,
        #                                 gamma=self.hparams.lr_decay_rate)
        #         schedulers.append(scheduler_encoder)
        #     if self.hparams.lr_decoder:                            
        #         scheduler_decoder = lrs.StepLR(optimizer_decoder,
        #                                 # step_size=5,
        #                                 step_size=self.hparams.lr_decay_steps,
        #                                 gamma=self.hparams.lr_decay_rate)
        #         schedulers.append(scheduler_decoder)
        # elif self.hparams.lr_scheduler == 'cosine':
        #     if self.hparams.lr_encoder:
        #         scheduler_encoder = lrs.CosineAnnealingLR(optimizer_encoder,
        #                                         T_max=self.hparams.lr_decay_steps,
        #                                         eta_min=self.hparams.lr_decay_min_lr)
        #         schedulers.append(scheduler_encoder)    
        #     if self.hparams.lr_decoder:                              
        #         scheduler_decoder = lrs.CosineAnnealingLR(optimizer_decoder,
        #                                         T_max=self.hparams.lr_decay_steps,
        #                                         eta_min=self.hparams.lr_decay_min_lr)
        #         schedulers.append(scheduler_decoder)                                
        
        return optimizers,schedulers
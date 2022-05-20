
import random

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support,average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import ROC
from torchmetrics import Metric

class RocPloter(Metric):
    def __init__(self, classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class Accuracy(object):
    """Computes and stores the average and current value"""
    def __init__(self, average = 'macro'):
        self.averagetype = average
        self.reset()

    def reset(self):
        self.label = torch.empty((0), dtype=torch.int)
        self.pred = torch.empty((0), dtype=torch.int)
        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def update(self, label, pred, n=1):
        self.label = torch.cat((self.label,label.argmax(dim = -1).cpu()))
        self.pred = torch.cat((self.pred,pred.argmax(dim = -1).cpu()))

    def calculate(self):
        self.precision,self.recall,self.fscore,_=precision_recall_fscore_support(self.label, self.pred, average=self.averagetype)

class BinaryAccuracy(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.label = torch.empty((0), dtype=torch.int)
        self.pred = torch.empty((0), dtype=torch.float)
        self.precision = 0

    def update(self, label, pred, n=1):
        self.label = torch.cat((self.label,label.cpu()))
        self.pred = torch.cat((self.pred,pred.cpu()))

    def calculate(self):
        self.precision =average_precision_score(self.label.detach().numpy(), self.pred.detach().numpy())

class RocPloter(object):
    def __init__(self, classes, fmt=':f'):
        self.classes = classes
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.label = torch.empty((0,len(self.classes)), dtype=torch.int)
        self.output = torch.empty((0,len(self.classes)), dtype=torch.float)

    def update(self, label, output, n=1):
        self.label =  torch.cat((self.label,label.cpu()))
        # sf = nn.Softmax(dim=1)
        # output = sf(output)
        self.output = torch.cat((self.output,output.cpu()))

    def generate_curve(self):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(self.classes)):
            fpr[i], tpr[i], _ = roc_curve(self.label[:, i], self.output[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])   
        return fpr, tpr, roc_auc 

    def plot(self):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        plt.figure()
        for i in range(len(self.classes)):
            fpr[i], tpr[i], _ = roc_curve(self.label[:, i], self.output[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(
                fpr[i],
                tpr[i],
                color=(random.random(),random.random(),random.random()),
                lw=2,
                label="{0} (area = {1:0.2f})".format(self.classes[i], roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend(loc="lower right")
        plt.show()

def pixel_acc(pred, label, ignore_index=-1):
    _, preds = torch.max(pred, dim=1)
    valid = (label != ignore_index).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

def part_pixel_acc(pred_part, gt_seg_part, gt_seg_object, object_label, valid):
    mask_object = (gt_seg_object == object_label)
    _, pred = torch.max(pred_part, dim=1)
    acc_sum = mask_object * (pred == gt_seg_part)
    acc_sum = torch.sum(acc_sum.view(acc_sum.size(0), -1), dim=1)
    acc_sum = torch.sum(acc_sum * valid)
    pixel_sum = torch.sum(mask_object.view(mask_object.size(0), -1), dim=1)
    pixel_sum = torch.sum(pixel_sum * valid)
    return acc_sum, pixel_sum 

def part_loss(pred_part, gt_seg_part, gt_seg_object, object_label, valid):
    mask_object = (gt_seg_object == object_label)
    loss = F.nll_loss(pred_part, gt_seg_part * mask_object.long(), reduction='none')
    loss = loss * mask_object.float()
    loss = torch.sum(loss.view(loss.size(0), -1), dim=1)
    nr_pixel = torch.sum(mask_object.view(mask_object.shape[0], -1), dim=1)
    sum_pixel = (nr_pixel * valid).sum()
    loss = (loss * valid.float()).sum() / torch.clamp(sum_pixel, 1).float()
    return loss
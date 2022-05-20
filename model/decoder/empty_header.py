import torch
from torch import nn
from metrics import pixel_acc
import torchmetrics
class EmptyHeader(nn.Module):
    def __init__(self):
        super().__init__()
        # self.out_channel = out_channel

        # self.crit_dict = nn.ModuleDict()
        # self.crit_dict['total'] = nn.CrossEntropyLoss()
        # self.metric_dict = {}    
        # self.metric_dict['f1_score'] = torchmetrics.F1Score(out_channel)

    def forward(self, x):
        output = dict()
        output['lesion'] = x
        return output

    # def loss(self,pred,feed_dict):
    #     # loss
    #     loss_dict = {}
    #     loss_dict['total'] = self.crit_dict['total'](pred, feed_dict['label']).mean()
    #     return loss_dict

    # def update_metric(self,pred,feed_dict):
    #     self.metric_dict['f1_score'].update(pred.argmax(axis=1).cpu(),feed_dict['label'].cpu())

    # def get_metric(self):
    #     # metric 
    #     acc_dict = {}
    #     for k in self.metric_dict.keys():
    #         acc_dict[k] = self.metric_dict[k].compute()
    #         self.metric_dict[k].reset()
    #     return acc_dict
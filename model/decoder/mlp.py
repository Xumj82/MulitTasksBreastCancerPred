import torch
from torch import nn
from metrics import pixel_acc
class Mlp(nn.Module):
    def __init__(self, fc_dim=3300,freeze=False,pretrained=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(5)
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=-3),
            nn.Linear(fc_dim, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512,1),
        )
        self.crit_dict = nn.ModuleDict()
        self.crit_dict['pathology'] = nn.L1Loss()

        if freeze:
            for param in self.fc.parameters():
                param.requires_grad = False

    def forward(self, x, orig_feed):
        x = self.bn(x)
        x = self.fc(x)
        x = torch.squeeze(x)
        return x

    def loss(self,pred,feed_dict):
        # loss
        loss_dict = {}
        loss_dict['pathology_loss'] = self.crit_dict['pathology'](pred, feed_dict['pathology'].float())
        loss_dict['loss'] = loss_dict['pathology_loss']
        return loss_dict

    def metric(self,pred,feed_dict):
        # metric 
        metric_dict= {}

        metric_dict['pathology_acc'] =self.crit_dict['pathology'](pred, feed_dict['pathology'])

        return metric_dict
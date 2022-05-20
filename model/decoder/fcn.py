import torch
from torch import nn
class Fcn(nn.Module):
    def __init__(self, nr_classes,fc_dim=512):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Conv2d(fc_dim,512,7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(512,512,1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(512,nr_classes,1),
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = torch.squeeze(x)
        return x
import torch
from torchsummary import summary
from torch import nn
class Encoder(nn.Module):
    def __init__(self, in_channel=1, resnet_name='resnet50', freeze=False, pretrained=True):
        super().__init__()
        resnet = torch.hub.load('pytorch/vision:v0.9.0', resnet_name, pretrained=pretrained)
        
        inter_ftrs = resnet.conv1.out_channels
        conv1 = nn.Conv2d(in_channel, inter_ftrs, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.backbone = nn.Sequential(
            conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

    def forward(self, x):
        x = self.backbone(x)
        return x

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = LesionNetV2().to(device)

# summary(model, (1, 512, 224))
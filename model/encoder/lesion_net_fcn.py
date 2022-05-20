import torch
from torch import nn
class Encoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=5, resnet_name='resnet50', freeze=False, pretrained=True):
        super().__init__()
        print(in_channel, out_channel)
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
        num_ftrs = resnet.fc.in_features

        self.fc = nn.Sequential(
            nn.Conv2d(num_ftrs,512,7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(512,512,1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(512,out_channel,1),
        )
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.fc.parameters():
                param.requires_grad = False
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x
        x = self.backbone(x)
        x = self.fc(x)
        x = torch.squeeze(x)
        # x = self.sigmoid(x)
        return x

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = LesionNetV2().to(device)

# summary(model, (1, 512, 224))
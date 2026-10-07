import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from src.models.cbam import CBAM


class CBAMResNet18(nn.Module):

    def __init__(
        self,
        pretrained=True,
        num_classes=4
    ):
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained else None
        )

        backbone = resnet18(weights=weights)

        # Feature extractor (everything except avgpool and fc)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,

            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        # Custom CBAM
        self.cbam = CBAM(
            in_channels=512
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(
            512,
            num_classes
        )

    def forward(self, x):

        x = self.features(x)

        x = self.cbam(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
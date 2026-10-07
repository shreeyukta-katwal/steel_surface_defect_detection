import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from src.models.cbam import CBAM
from src.models.unet_blocks import DecoderBlock


class UNetResNet18(nn.Module):

    def __init__(
        self,
        num_classes=4,
        pretrained=True,
        use_cbam=False
    ):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        backbone = resnet18(weights=weights)

        self.initial = nn.Sequential(

            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )

        self.pool = backbone.maxpool

        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.use_cbam = use_cbam

        if use_cbam:

            self.cbam = CBAM(
                in_channels=512
            )

        self.decoder4 = DecoderBlock(
            512,
            256,
            256
        )

        self.decoder3 = DecoderBlock(
            256,
            128,
            128
        )

        self.decoder2 = DecoderBlock(
            128,
            64,
            64
        )

        self.decoder1 = DecoderBlock(
            64,
            64,
            64
        )

        self.final_up = nn.ConvTranspose2d(
            64,
            64,
            kernel_size=2,
            stride=2
        )

        self.final_conv = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        x0 = self.initial(x)

        x1 = self.pool(x0)

        x1 = self.encoder1(x1)

        x2 = self.encoder2(x1)

        x3 = self.encoder3(x2)

        x4 = self.encoder4(x3)

        if self.use_cbam:
            x4 = self.cbam(x4)

        d4 = self.decoder4(x4, x3)

        d3 = self.decoder3(d4, x2)

        d2 = self.decoder2(d3, x1)

        d1 = self.decoder1(d2, x0)

        out = self.final_up(d1)

        out = self.final_conv(out)

        return out
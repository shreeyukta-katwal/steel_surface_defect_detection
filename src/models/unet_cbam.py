import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from src.models.cbam import CBAM


# ==========================================================
# Decoder Block
# ==========================================================

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = nn.Sequential(

            nn.Conv2d(
                out_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, x, skip):

        x = self.up(x)

        x = torch.cat([x, skip], dim=1)

        return self.conv(x)


# ==========================================================
# U-Net + CBAM
# ==========================================================

class UNetCBAM(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

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

        # ==================================================
        # CBAM Bottleneck
        # ==================================================

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

        self.final = nn.Conv2d(
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

        # ----------------------------------
        # CBAM inserted here
        # ----------------------------------

        x4 = self.cbam(x4)

        d4 = self.decoder4(x4, x3)

        d3 = self.decoder3(d4, x2)

        d2 = self.decoder2(d3, x1)

        d1 = self.decoder1(d2, x0)

        out = self.final(d1)

        out = F.interpolate(
            out,
            size=(256,256),
            mode="bilinear",
            align_corners=False
        )

        return out
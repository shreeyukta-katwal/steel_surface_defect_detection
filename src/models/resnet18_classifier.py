import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Classifier(nn.Module):

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

        self.model = resnet18(weights=weights)

        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)
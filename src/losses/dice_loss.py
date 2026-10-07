import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(probs.size(0), probs.size(1), -1)
        targets = targets.contiguous().view(targets.size(0), targets.size(1), -1)

        intersection = (probs * targets).sum(dim=2)

        union = probs.sum(dim=2) + targets.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (
            union + self.smooth
        )

        loss = 1.0 - dice

        return loss.mean()


class DiceBCELoss(nn.Module):

    def __init__(
        self,
        dice_weight=0.5,
        bce_weight=0.5
    ):
        super().__init__()

        self.dice = DiceLoss()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):

        dice_loss = self.dice(logits, targets)

        bce_loss = self.bce(logits, targets)

        total_loss = (
            self.dice_weight * dice_loss +
            self.bce_weight * bce_loss
        )

        return total_loss
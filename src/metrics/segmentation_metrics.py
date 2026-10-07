import torch


def dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Mean Dice score across all classes.
    """

    probs = torch.sigmoid(logits)

    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), preds.size(1), -1)
    targets = targets.view(targets.size(0), targets.size(1), -1)

    intersection = (preds * targets).sum(dim=2)

    union = preds.sum(dim=2) + targets.sum(dim=2)

    dice = (2 * intersection + smooth) / (union + smooth)

    return dice.mean().item()


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Mean IoU (Jaccard Index).
    """

    probs = torch.sigmoid(logits)

    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), preds.size(1), -1)
    targets = targets.view(targets.size(0), targets.size(1), -1)

    intersection = (preds * targets).sum(dim=2)

    union = preds.sum(dim=2) + targets.sum(dim=2) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.mean().item()


def pixel_accuracy(logits, targets, threshold=0.5):
    """
    Pixel-wise accuracy.
    """

    probs = torch.sigmoid(logits)

    preds = (probs > threshold).float()

    correct = (preds == targets).float().sum()

    total = targets.numel()

    return (correct / total).item()
import torch
from tqdm import tqdm

from src.metrics.segmentation_metrics import (
    dice_score,
    iou_score,
    pixel_accuracy
)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
):

    model.train()

    running_loss = 0.0

    for images, masks in tqdm(loader):

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda"
        ):

            outputs = model(images)

            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    dice_total = 0.0

    iou_total = 0.0

    pixel_total = 0.0

    for images, masks in tqdm(loader):

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda"
        ):

            outputs = model(images)

            loss = criterion(outputs, masks)

        running_loss += loss.item()

        dice_total += dice_score(outputs, masks)
        iou_total += iou_score(outputs, masks)
        pixel_total += pixel_accuracy(outputs, masks)

    return {
        "loss": running_loss / len(loader),
        "dice": dice_total / len(loader),
        "iou": iou_total / len(loader),
        "pixel_acc": pixel_total / len(loader),
    }
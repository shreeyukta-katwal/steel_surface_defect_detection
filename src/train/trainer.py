import time
import numpy as np
import torch
from torch import amp
from tqdm import tqdm

from src.train.metrics import calculate_metrics


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None,
):

    model.train()

    running_loss = 0

    start = time.time()

    for images, labels in tqdm(loader):

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:

            with amp.autocast(device_type=device.type):

                outputs = model(images)

                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

        running_loss += loss.item()

    epoch_time = time.time() - start

    return running_loss / len(loader), epoch_time


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0

    all_outputs = []

    all_targets = []

    for images, labels in loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item()

        all_outputs.append(outputs.cpu().numpy())

        all_targets.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)

    all_targets = np.concatenate(all_targets)

    metrics = calculate_metrics(
        all_outputs,
        all_targets
    )

    return (
        running_loss / len(loader),
        metrics
    )
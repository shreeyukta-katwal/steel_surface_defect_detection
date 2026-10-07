import os
import time
import random

import numpy as np
import pandas as pd

import torch
import albumentations as A

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.datasets.segmentation_dataset import SteelSegmentationDataset

from src.models.unet_resnet18 import UNetResNet18
from src.models.unet_cbam import UNetCBAM

from src.losses.dice_loss import DiceBCELoss

from src.train.segmentation_trainer import (
    train_one_epoch,
    validate
)

# ==========================================================
# CONFIG
# ==========================================================

CONFIG = {

    "seed": 42,

    "image_size": 256,

    "batch_size": 8,

    "epochs": 40,

    "learning_rate": 1e-4,

    "weight_decay": 1e-4,

    "patience": 8,

    "num_workers": 2,

    "pin_memory": True,

    "model_type": "cbam",      # baseline | cbam

    "train_csv":
        "data/train.csv",

    "image_dir":
        "data/train_images",

    "checkpoint_dir":
        "checkpoints",

    "log_dir":
        "logs"
}


# ==========================================================
# Seed
# ==========================================================

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
    

def build_model(model_type):

    if model_type == "baseline":

        return UNetResNet18()

    elif model_type == "cbam":

        return UNetCBAM()

    raise ValueError(f"Unknown model type: {model_type}")

# ==========================================================
# Transforms
# ==========================================================

def get_train_transform():

    return A.Compose([

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),

        A.RandomRotate90(p=0.5)

    ])


def get_val_transform():

    return A.Compose([])

# ==========================================================
# Main
# ==========================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=CONFIG["model_type"],
        choices=["baseline", "cbam"],
        help="Model architecture: baseline or cbam"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG["seed"],
        help="Random seed for split and reproducibility"
    )
    args = parser.parse_args()
    CONFIG["model_type"] = args.model
    CONFIG["seed"] = args.seed

    seed_everything(CONFIG["seed"])

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    print()

    print("Using device :", device)


    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    print()
    print("Configuration")
    print("------------------------------")
    print("Model      :", CONFIG["model_type"])
    print("Image Size :", CONFIG["image_size"])
    print("Batch Size :", CONFIG["batch_size"])
    print("Epochs     :", CONFIG["epochs"])
    print("LR         :", CONFIG["learning_rate"])
    print()

    if device.type == "cuda":

        print(torch.cuda.get_device_name(0))

    os.makedirs(

        CONFIG["checkpoint_dir"],

        exist_ok=True

    )

    os.makedirs(

        CONFIG["log_dir"],

        exist_ok=True

    )
    
    # ==========================================================
    # Load Dataset
    # ==========================================================

    master_df = pd.read_csv(

        CONFIG["train_csv"]

    )

    train_df, val_df = train_test_split(

        master_df,

        test_size=0.2,

        shuffle=True,

        random_state=CONFIG["seed"]

    )
    
    # ==========================================================
    # Datasets
    # ==========================================================

    train_transform = get_train_transform()

    val_transform = get_val_transform()

    train_dataset = SteelSegmentationDataset(

        dataframe=train_df,

        image_dir=CONFIG["image_dir"],

        transform=train_transform,

        image_size=(CONFIG["image_size"], CONFIG["image_size"])

    )

    val_dataset = SteelSegmentationDataset(

        dataframe=val_df,

        image_dir=CONFIG["image_dir"],

        transform=val_transform,

        image_size=(CONFIG["image_size"], CONFIG["image_size"])

    )
    
    # ==========================================================
    # DataLoaders
    # ==========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=CONFIG["batch_size"],

        shuffle=True,

        num_workers=CONFIG["num_workers"],

        pin_memory=CONFIG["pin_memory"]

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=CONFIG["batch_size"],

        shuffle=False,

        num_workers=CONFIG["num_workers"],

        pin_memory=CONFIG["pin_memory"]

    )
    
    print("Everything initialized successfully.")

    print("Train Images :", len(train_dataset))

    print("Validation Images :", len(val_dataset))

    print("Batch Size :", CONFIG["batch_size"])

    print("Epochs :", CONFIG["epochs"])

    print("Learning Rate :", CONFIG["learning_rate"])

    print()
    
    # ==========================================================
    # Model
    # ==========================================================

    model = build_model(

        CONFIG["model_type"]

    )

    model = model.to(device)
    
    
    # ==========================================================
    # Loss
    # ==========================================================

    criterion = DiceBCELoss()
    
    # ==========================================================
    # Optimizer
    # ==========================================================

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=CONFIG["learning_rate"],

        weight_decay=CONFIG["weight_decay"]

    )
    
    # ==========================================================
    # Scheduler
    # ==========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=3

    )
    
    # ==========================================================
    # AMP
    # ==========================================================

    scaler = torch.amp.GradScaler(

        "cuda",

        enabled=device.type == "cuda"

    )
    
    history = []

    best_dice = 0.0

    patience_counter = 0
    
    for epoch in range(CONFIG["epochs"]):

        print()

        print("=" * 60)

        print(f"Epoch {epoch + 1}/{CONFIG['epochs']}")

        print("=" * 60)

        start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step(val_metrics["dice"])

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - start_time

        history.append({

            "epoch": epoch + 1,

            "train_loss": train_loss,

            "val_loss": val_metrics["loss"],

            "dice": val_metrics["dice"],

            "iou": val_metrics["iou"],

            "pixel_acc": val_metrics["pixel_acc"],

            "lr": current_lr
        })
        
        if val_metrics["dice"] > best_dice:

            best_dice = val_metrics["dice"]

            patience_counter = 0

            if CONFIG["seed"] == 42:
                checkpoint_path = os.path.join(
                    CONFIG["checkpoint_dir"],
                    f"{CONFIG['model_type']}_unet_best.pth"
                )
            else:
                checkpoint_path = os.path.join(
                    CONFIG["checkpoint_dir"],
                    f"{CONFIG['model_type']}_unet_seed{CONFIG['seed']}_best.pth"
                )

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print("✓ Best model saved.")

        else:

            patience_counter += 1
            
        print(f"Train Loss : {train_loss:.4f}")

        print(f"Val Loss   : {val_metrics['loss']:.4f}")

        print(f"Dice       : {val_metrics['dice']:.4f}")

        print(f"IoU        : {val_metrics['iou']:.4f}")

        print(f"Pixel Acc  : {val_metrics['pixel_acc']:.4f}")

        print(f"LR         : {current_lr:.6f}")

        print(f"Epoch Time : {epoch_time:.1f} sec")
        
        if patience_counter >= CONFIG["patience"]:

            print()

            print("Early stopping triggered.")

            break
        
        
    history_df = pd.DataFrame(history)

    if CONFIG["seed"] == 42:
        history_path = os.path.join(
            CONFIG["log_dir"],
            f"{CONFIG['model_type']}_history.csv"
        )
    else:
        history_path = os.path.join(
            CONFIG["log_dir"],
            f"{CONFIG['model_type']}_history_seed{CONFIG['seed']}.csv"
        )
    history_df.to_csv(history_path, index=False)

    print()

    print("Training completed successfully.")
    
    
if __name__ == "__main__":

    main()
    
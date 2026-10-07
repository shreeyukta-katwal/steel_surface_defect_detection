import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    multilabel_confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)
from torch.utils.data import DataLoader

from src.datasets.classification_dataset import ClassificationDataset
from src.datasets.transforms import get_val_classification_transform
from src.models.cbam_resnet18 import CBAMResNet18
from src.models.resnet18_classifier import ResNet18Classifier


CLASS_NAMES = ["Class 1", "Class 2", "Class 3", "Class 4"]

CLASSIFICATION_LOG_COLUMNS = {
    "epoch",
    "train_loss",
    "val_loss",
    "precision",
    "recall",
    "f1_macro",
    "f1_micro",
    "roc_auc",
}

SEGMENTATION_LOG_COLUMNS = {
    "epoch",
    "train_loss",
    "val_loss",
    "dice",
    "iou",
    "pixel_acc",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training and validation plots for classification and segmentation."
    )

    parser.add_argument(
        "--baseline-classification-log",
        default="logs/baseline_seed42.csv",
        help="Baseline ResNet18 classification training CSV.",
    )

    parser.add_argument(
        "--cbam-classification-log",
        default="logs/cbam_seed42.csv",
        help="CBAM-ResNet18 classification training CSV.",
    )

    parser.add_argument(
        "--baseline-segmentation-log",
        default="logs/baseline_history.csv",
        help="Baseline U-Net segmentation history CSV.",
    )

    parser.add_argument(
        "--cbam-segmentation-log",
        default="logs/cbam_history.csv",
        help="CBAM U-Net segmentation history CSV.",
    )

    parser.add_argument(
        "--val-csv",
        default="data/processed/val_split.csv",
        help="Validation split used for classifier evaluation plots.",
    )

    parser.add_argument(
        "--image-dir",
        default="data/train_images",
        help="Directory containing training images.",
    )

    parser.add_argument(
        "--baseline-checkpoint",
        default="checkpoints/baseline_resnet18_best.pth",
        help="Baseline ResNet18 checkpoint for PR, ROC, and confusion matrix plots.",
    )

    parser.add_argument(
        "--cbam-checkpoint",
        default="checkpoints/cbam_resnet18_best.pth",
        help="CBAM-ResNet18 checkpoint for PR, ROC, and confusion matrix plots.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Validation batch size for classifier evaluation plots.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Validation DataLoader workers. Zero is safest for Windows plotting runs.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used for multilabel confusion matrices.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/plots",
        help="Directory where plot files will be written.",
    )

    return parser.parse_args()


def ensure_columns(df, path, required_columns):
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing}")


def read_log(path, required_columns):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Log not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Log is empty: {path}")

    ensure_columns(df, path, required_columns)
    return df


def setup_style():
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def save_current_figure(output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_two_model_metric(
    baseline_df,
    cbam_df,
    metric,
    ylabel,
    title,
    output_path,
):
    plt.figure()
    plt.plot(
        baseline_df["epoch"],
        baseline_df[metric],
        marker="o",
        linewidth=1.8,
        markersize=3.5,
        label="Baseline",
    )
    plt.plot(
        cbam_df["epoch"],
        cbam_df[metric],
        marker="s",
        linewidth=1.8,
        markersize=3.5,
        label="CBAM",
    )
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    save_current_figure(output_path)


def plot_classification_curves(baseline_df, cbam_df, output_dir):
    classification_dir = output_dir / "classification"

    plot_two_model_metric(
        baseline_df,
        cbam_df,
        "train_loss",
        "Loss",
        "Classification Training Loss",
        classification_dir / "training_loss.png",
    )

    plot_two_model_metric(
        baseline_df,
        cbam_df,
        "val_loss",
        "Loss",
        "Classification Validation Loss",
        classification_dir / "validation_loss.png",
    )

    plot_two_model_metric(
        baseline_df,
        cbam_df,
        "f1_macro",
        "Macro F1",
        "Classification Macro F1",
        classification_dir / "macro_f1.png",
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    metric_specs = [
        ("train_loss", "Training Loss"),
        ("val_loss", "Validation Loss"),
        ("f1_macro", "Macro F1"),
        ("roc_auc", "ROC AUC"),
    ]

    for ax, (metric, title) in zip(axes.ravel(), metric_specs):
        ax.plot(
            baseline_df["epoch"],
            baseline_df[metric],
            marker="o",
            linewidth=1.6,
            markersize=3,
            label="Baseline",
        )
        ax.plot(
            cbam_df["epoch"],
            cbam_df[metric],
            marker="s",
            linewidth=1.6,
            markersize=3,
            label="CBAM",
        )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)

    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Score")
    axes[1, 1].set_ylabel("Score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Classification Learning Curves", y=1.03)
    save_current_figure(classification_dir / "learning_curves.png")


def plot_segmentation_curves(baseline_df, cbam_df, output_dir):
    segmentation_dir = output_dir / "segmentation"

    plot_two_model_metric(
        baseline_df,
        cbam_df,
        "train_loss",
        "Loss",
        "Segmentation Training Loss",
        segmentation_dir / "training_loss.png",
    )

    plot_two_model_metric(
        baseline_df,
        cbam_df,
        "val_loss",
        "Loss",
        "Segmentation Validation Loss",
        segmentation_dir / "validation_loss.png",
    )

    plot_two_model_metric(
        baseline_df,
        cbam_df,
        "dice",
        "Dice",
        "Segmentation Dice Curve",
        segmentation_dir / "dice_curve.png",
    )

    plot_two_model_metric(
        baseline_df,
        cbam_df,
        "iou",
        "IoU",
        "Segmentation IoU Curve",
        segmentation_dir / "iou_curve.png",
    )

    plot_two_model_metric(
        baseline_df,
        cbam_df,
        "pixel_acc",
        "Pixel Accuracy",
        "Segmentation Pixel Accuracy Curve",
        segmentation_dir / "pixel_accuracy_curve.png",
    )


def build_model(model_name):
    if model_name == "baseline":
        return ResNet18Classifier(pretrained=False, num_classes=4)

    if model_name == "cbam":
        return CBAMResNet18(pretrained=False, num_classes=4)

    raise ValueError(f"Unknown model name: {model_name}")


def load_state_dict(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def predict_classifier(model, loader, device):
    model.eval()
    probabilities = []
    targets = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()

        probabilities.append(probs)
        targets.append(labels.numpy())

    return np.concatenate(targets, axis=0), np.concatenate(probabilities, axis=0)


def get_predictions(args, model_name, checkpoint_path, loader, device):
    model = build_model(model_name)
    model = load_state_dict(model, checkpoint_path, device)
    model = model.to(device)
    return predict_classifier(model, loader, device)


def plot_precision_recall_curves(y_true, predictions_by_model, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)

    for class_idx, ax in enumerate(axes.ravel()):
        for model_label, y_prob in predictions_by_model.items():
            precision, recall, _ = precision_recall_curve(
                y_true[:, class_idx],
                y_prob[:, class_idx],
            )
            ap = average_precision_score(y_true[:, class_idx], y_prob[:, class_idx])
            ax.plot(
                recall,
                precision,
                linewidth=1.8,
                label=f"{model_label} AP={ap:.3f}",
            )

        ax.set_title(CLASS_NAMES[class_idx])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc="lower left")

    fig.suptitle("Precision-Recall Curves", y=1.02)
    save_current_figure(output_path)


def plot_roc_curves(y_true, predictions_by_model, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)

    for class_idx, ax in enumerate(axes.ravel()):
        for model_label, y_prob in predictions_by_model.items():
            fpr, tpr, _ = roc_curve(y_true[:, class_idx], y_prob[:, class_idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                linewidth=1.8,
                label=f"{model_label} AUC={roc_auc:.3f}",
            )

        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
        ax.set_title(CLASS_NAMES[class_idx])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc="lower right")

    fig.suptitle("ROC Curves", y=1.02)
    save_current_figure(output_path)


def plot_confusion_matrices(y_true, y_prob, model_label, threshold, output_path):
    y_pred = (y_prob >= threshold).astype(np.int64)
    matrices = multilabel_confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    labels = np.array([["TN", "FP"], ["FN", "TP"]])

    for class_idx, ax in enumerate(axes.ravel()):
        matrix = matrices[class_idx]
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(CLASS_NAMES[class_idx])
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])

        max_value = matrix.max() if matrix.max() > 0 else 1

        for row_idx in range(2):
            for col_idx in range(2):
                value = matrix[row_idx, col_idx]
                text_color = "white" if value > max_value * 0.55 else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{labels[row_idx, col_idx]}\n{value}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    fig.suptitle(f"{model_label} Multilabel Confusion Matrices", y=1.02)
    save_current_figure(output_path)


def write_plot_manifest(output_dir, generated_files):
    manifest_path = output_dir / "plot_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["plot_file"])

        for path in generated_files:
            writer.writerow([path.as_posix()])

    return manifest_path


def main():
    args = parse_args()
    setup_style()

    output_dir = Path(args.output_dir)
    generated_files = []

    baseline_classification_df = read_log(
        args.baseline_classification_log,
        CLASSIFICATION_LOG_COLUMNS,
    )
    cbam_classification_df = read_log(
        args.cbam_classification_log,
        CLASSIFICATION_LOG_COLUMNS,
    )
    baseline_segmentation_df = read_log(
        args.baseline_segmentation_log,
        SEGMENTATION_LOG_COLUMNS,
    )
    cbam_segmentation_df = read_log(
        args.cbam_segmentation_log,
        SEGMENTATION_LOG_COLUMNS,
    )

    plot_classification_curves(
        baseline_classification_df,
        cbam_classification_df,
        output_dir,
    )
    generated_files.extend(
        [
            output_dir / "classification" / "training_loss.png",
            output_dir / "classification" / "validation_loss.png",
            output_dir / "classification" / "macro_f1.png",
            output_dir / "classification" / "learning_curves.png",
        ]
    )

    plot_segmentation_curves(
        baseline_segmentation_df,
        cbam_segmentation_df,
        output_dir,
    )
    generated_files.extend(
        [
            output_dir / "segmentation" / "training_loss.png",
            output_dir / "segmentation" / "validation_loss.png",
            output_dir / "segmentation" / "dice_curve.png",
            output_dir / "segmentation" / "iou_curve.png",
            output_dir / "segmentation" / "pixel_accuracy_curve.png",
        ]
    )

    val_df = pd.read_csv(args.val_csv)
    val_dataset = ClassificationDataset(
        val_df,
        args.image_dir,
        transform=get_val_classification_transform(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true, baseline_prob = get_predictions(
        args,
        "baseline",
        args.baseline_checkpoint,
        val_loader,
        device,
    )
    cbam_true, cbam_prob = get_predictions(
        args,
        "cbam",
        args.cbam_checkpoint,
        val_loader,
        device,
    )

    if not np.array_equal(y_true, cbam_true):
        raise ValueError("Baseline and CBAM validation targets do not match.")

    classification_dir = output_dir / "classification"
    predictions_by_model = {
        "Baseline": baseline_prob,
        "CBAM": cbam_prob,
    }

    plot_precision_recall_curves(
        y_true,
        predictions_by_model,
        classification_dir / "precision_recall_curve.png",
    )
    plot_roc_curves(
        y_true,
        predictions_by_model,
        classification_dir / "roc_curve.png",
    )
    plot_confusion_matrices(
        y_true,
        baseline_prob,
        "Baseline ResNet18",
        args.threshold,
        classification_dir / "baseline_confusion_matrix.png",
    )
    plot_confusion_matrices(
        y_true,
        cbam_prob,
        "CBAM-ResNet18",
        args.threshold,
        classification_dir / "cbam_confusion_matrix.png",
    )

    generated_files.extend(
        [
            classification_dir / "precision_recall_curve.png",
            classification_dir / "roc_curve.png",
            classification_dir / "baseline_confusion_matrix.png",
            classification_dir / "cbam_confusion_matrix.png",
        ]
    )

    manifest_path = write_plot_manifest(output_dir, generated_files)

    print(f"Wrote {len(generated_files)} plots")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.classification_dataset import ClassificationDataset
from src.datasets.transforms import get_val_classification_transform
from src.models.cbam_resnet18 import CBAMResNet18
from src.models.resnet18_classifier import ResNet18Classifier


CLASS_COLUMNS = ["class_1", "class_2", "class_3", "class_4"]
CLASS_NAMES = ["Class 1", "Class 2", "Class 3", "Class 4"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save per-class classification error examples."
    )
    parser.add_argument("--val-csv", default="data/processed/val_split.csv")
    parser.add_argument("--image-dir", default="data/train_images")
    parser.add_argument("--output-dir", default="results/error_analysis/classification")
    parser.add_argument(
        "--baseline-checkpoint",
        default="checkpoints/baseline_resnet18_best.pth",
    )
    parser.add_argument(
        "--cbam-checkpoint",
        default="checkpoints/cbam_resnet18_best.pth",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--examples-per-class", type=int, default=2)
    return parser.parse_args()


def load_model(model_name, checkpoint_path, device):
    if model_name == "baseline":
        model = ResNet18Classifier(pretrained=False, num_classes=4)
    elif model_name == "cbam":
        model = CBAMResNet18(pretrained=False, num_classes=4)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def collect_probabilities(model, loader, device):
    probabilities = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probabilities.append(torch.sigmoid(logits).cpu())
            targets.append(labels.cpu())

    return torch.cat(probabilities).numpy(), torch.cat(targets).numpy().astype(int)


def read_image(image_dir, image_id):
    image = cv2.imread(str(Path(image_dir) / image_id))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_id}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_example_figure(image, row, output_path):
    plt.figure(figsize=(8, 3.5))
    plt.imshow(image)
    plt.axis("off")
    title = (
        f"{row['model']} | {row['class_name']} {row['error_type']} | "
        f"p={float(row['probability']):.3f}"
    )
    subtitle = f"true={row['true_label']} pred={row['predicted_label']} | {row['image_id']}"
    plt.title(f"{title}\n{subtitle}", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def select_errors(model_label, df, probabilities, targets, threshold, examples_per_class):
    predictions = (probabilities >= threshold).astype(int)
    rows = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_probs = probabilities[:, class_idx]
        class_targets = targets[:, class_idx]
        class_preds = predictions[:, class_idx]

        candidates = [
            ("false_negative", (class_targets == 1) & (class_preds == 0), 1.0 - class_probs),
            ("false_positive", (class_targets == 0) & (class_preds == 1), class_probs),
        ]

        for error_type, mask, sort_scores in candidates:
            indices = mask.nonzero()[0]
            if len(indices) == 0:
                continue

            ranked = sorted(indices, key=lambda idx: sort_scores[idx], reverse=True)
            for rank, idx in enumerate(ranked[:examples_per_class], start=1):
                rows.append(
                    {
                        "model": model_label,
                        "class_name": class_name,
                        "class_index": class_idx + 1,
                        "error_type": error_type,
                        "rank": rank,
                        "image_id": df.iloc[idx]["ImageId"],
                        "true_label": int(class_targets[idx]),
                        "predicted_label": int(class_preds[idx]),
                        "probability": float(class_probs[idx]),
                    }
                )

    return rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.val_csv)
    dataset = ClassificationDataset(
        df,
        args.image_dir,
        transform=get_val_classification_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    manifest_rows = []
    for model_name, model_label, checkpoint_path in [
        ("baseline", "Baseline ResNet18", args.baseline_checkpoint),
        ("cbam", "CBAM-ResNet18", args.cbam_checkpoint),
    ]:
        model = load_model(model_name, checkpoint_path, device)
        probabilities, targets = collect_probabilities(model, loader, device)
        error_rows = select_errors(
            model_label,
            df,
            probabilities,
            targets,
            args.threshold,
            args.examples_per_class,
        )

        for row in error_rows:
            slug = (
                f"{model_name}_class{row['class_index']}_"
                f"{row['error_type']}_{row['rank']}.png"
            )
            figure_path = output_dir / slug
            image = read_image(args.image_dir, row["image_id"])
            save_example_figure(image, row, figure_path)
            row["figure_path"] = str(figure_path).replace("\\", "/")
            manifest_rows.append(row)

    manifest_path = output_dir / "classification_error_manifest.csv"
    fieldnames = [
        "model",
        "class_name",
        "class_index",
        "error_type",
        "rank",
        "image_id",
        "true_label",
        "predicted_label",
        "probability",
        "figure_path",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(manifest_rows)} error examples to {manifest_path}")


if __name__ == "__main__":
    main()

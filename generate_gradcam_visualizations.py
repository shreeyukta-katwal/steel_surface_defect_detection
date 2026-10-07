import argparse
import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.classification_dataset import ClassificationDataset
from src.datasets.transforms import get_val_classification_transform
from src.models.cbam_resnet18 import CBAMResNet18
from src.models.resnet18_classifier import ResNet18Classifier


CLASS_COLUMNS = ["class_1", "class_2", "class_3", "class_4"]
CLASS_NAMES = ["Class 1", "Class 2", "Class 3", "Class 4"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM figures for baseline and CBAM classifiers."
    )

    parser.add_argument(
        "--val-csv",
        default="data/processed/val_split.csv",
        help="Classification validation split CSV.",
    )

    parser.add_argument(
        "--image-dir",
        default="data/train_images",
        help="Directory containing training images.",
    )

    parser.add_argument(
        "--baseline-checkpoint",
        default="checkpoints/baseline_resnet18_best.pth",
        help="Baseline ResNet18 checkpoint.",
    )

    parser.add_argument(
        "--cbam-checkpoint",
        default="checkpoints/cbam_best.pth",
        help="CBAM-ResNet18 checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/gradcam/classification",
        help="Directory where Grad-CAM figures will be written.",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=6,
        help="Number of defective validation examples to visualize.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold shown in figure titles.",
    )

    return parser.parse_args()


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor, class_index):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        score = logits[:, class_index].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)[0].cpu().numpy()
        cam = normalize_cam(cam)
        return cam, logits.detach()

    def close(self):
        for handle in self.handles:
            handle.remove()


def normalize_cam(cam):
    cam = cam.astype(np.float32)
    cam -= cam.min()
    max_value = cam.max()

    if max_value > 0:
        cam /= max_value

    return cam


def load_model(model_name, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if model_name == "baseline":
        model = ResNet18Classifier(pretrained=False, num_classes=4)
    elif model_name == "cbam":
        model = CBAMResNet18(pretrained=False, num_classes=4)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    state_dict = torch.load(checkpoint_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def get_target_layer(model_name, model):
    if model_name == "baseline":
        return model.model.layer4[-1]

    if model_name == "cbam":
        return model.features[7][-1]

    raise ValueError(f"Unknown model name: {model_name}")


def tensor_to_display_image(image_tensor):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * IMAGENET_STD) + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def heatmap_from_cam(cam, image_shape):
    cam_resized = cv2.resize(cam, (image_shape[1], image_shape[0]))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def overlay_cam(image, cam, alpha=0.45):
    heatmap = heatmap_from_cam(cam, image.shape[:2])
    overlay = ((1.0 - alpha) * image.astype(np.float32)) + (
        alpha * heatmap.astype(np.float32)
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def select_examples(val_df, num_examples):
    defective = val_df[val_df["has_defect"] == 1].copy()

    if defective.empty:
        raise ValueError("No defective validation images found.")

    selected_rows = []
    seen_classes = set()

    for _, row in defective.iterrows():
        positive_classes = [
            class_idx
            for class_idx, column in enumerate(CLASS_COLUMNS)
            if int(row[column]) == 1
        ]

        if not positive_classes:
            continue

        if any(class_idx not in seen_classes for class_idx in positive_classes):
            selected_rows.append(row)
            seen_classes.update(positive_classes)

        if len(selected_rows) == num_examples:
            break

    if len(selected_rows) < num_examples:
        selected_ids = {row["ImageId"] for row in selected_rows}

        for _, row in defective.iterrows():
            if row["ImageId"] in selected_ids:
                continue

            selected_rows.append(row)

            if len(selected_rows) == num_examples:
                break

    if len(selected_rows) < num_examples:
        raise ValueError(
            f"Only found {len(selected_rows)} defective examples; requested {num_examples}."
        )

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def choose_target_class(row):
    for class_idx, column in enumerate(CLASS_COLUMNS):
        if int(row[column]) == 1:
            return class_idx

    raise ValueError(f"Selected image has no positive class: {row['ImageId']}")


def run_gradcam_for_row(row, dataset, baseline_cam, cbam_cam, device):
    image_tensor, label_tensor = dataset[0]
    image_batch = image_tensor.unsqueeze(0).to(device)
    target_class = choose_target_class(row)

    baseline_cam_map, baseline_logits = baseline_cam(image_batch, target_class)
    cbam_cam_map, cbam_logits = cbam_cam(image_batch, target_class)

    baseline_probs = torch.sigmoid(baseline_logits)[0].cpu().numpy()
    cbam_probs = torch.sigmoid(cbam_logits)[0].cpu().numpy()

    return {
        "image": tensor_to_display_image(image_tensor),
        "target_class": target_class,
        "baseline_cam": baseline_cam_map,
        "cbam_cam": cbam_cam_map,
        "baseline_probability": baseline_probs[target_class],
        "cbam_probability": cbam_probs[target_class],
        "labels": label_tensor.cpu().numpy(),
    }


def save_gradcam_comparison(result, image_id, output_path):
    image = result["image"]
    target_name = CLASS_NAMES[result["target_class"]]

    baseline_heatmap = heatmap_from_cam(result["baseline_cam"], image.shape[:2])
    cbam_heatmap = heatmap_from_cam(result["cbam_cam"], image.shape[:2])
    baseline_overlay = overlay_cam(image, result["baseline_cam"])
    cbam_overlay = overlay_cam(image, result["cbam_cam"])

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    panels = [
        ("Original Image", image),
        ("Baseline Heatmap", baseline_heatmap),
        ("Baseline Overlay", baseline_overlay),
        ("Original Image", image),
        ("CBAM Heatmap", cbam_heatmap),
        ("CBAM Overlay", cbam_overlay),
    ]

    for ax, (title, panel) in zip(axes.ravel(), panels):
        ax.imshow(panel)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel(
        f"Baseline\np={result['baseline_probability']:.3f}",
        rotation=90,
        labelpad=18,
    )
    axes[1, 0].set_ylabel(
        f"CBAM\np={result['cbam_probability']:.3f}",
        rotation=90,
        labelpad=18,
    )

    fig.suptitle(f"Grad-CAM Comparison - {image_id} - {target_name}", y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_manifest(rows, output_dir):
    manifest_path = output_dir / "gradcam_manifest.csv"

    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "target_class",
                "baseline_probability",
                "cbam_probability",
                "figure_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_csv(args.val_csv)
    selected_df = select_examples(val_df, args.num_examples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = load_model("baseline", args.baseline_checkpoint, device)
    cbam_model = load_model("cbam", args.cbam_checkpoint, device)

    baseline_cam = GradCAM(
        baseline_model,
        get_target_layer("baseline", baseline_model),
    )
    cbam_cam = GradCAM(
        cbam_model,
        get_target_layer("cbam", cbam_model),
    )

    manifest_rows = []

    try:
        for example_number, row in selected_df.iterrows():
            single_row_df = pd.DataFrame([row])
            dataset = ClassificationDataset(
                single_row_df,
                args.image_dir,
                transform=get_val_classification_transform(),
            )
            image_id = row["ImageId"]
            result = run_gradcam_for_row(
                row,
                dataset,
                baseline_cam,
                cbam_cam,
                device,
            )

            figure_path = output_dir / f"example_{example_number + 1:02d}_gradcam.png"
            save_gradcam_comparison(result, image_id, figure_path)

            manifest_rows.append(
                {
                    "image_id": image_id,
                    "target_class": CLASS_NAMES[result["target_class"]],
                    "baseline_probability": f"{result['baseline_probability']:.6f}",
                    "cbam_probability": f"{result['cbam_probability']:.6f}",
                    "figure_path": figure_path.as_posix(),
                }
            )
    finally:
        baseline_cam.close()
        cbam_cam.close()

    manifest_path = write_manifest(manifest_rows, output_dir)

    print(f"Wrote {len(manifest_rows)} Grad-CAM figures")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

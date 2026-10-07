import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.datasets.segmentation_dataset import SteelSegmentationDataset
from src.models.unet_cbam import UNetCBAM
from src.models.unet_resnet18 import UNetResNet18


CLASS_NAMES = ["Class 1", "Class 2", "Class 3", "Class 4"]
CLASS_COLORS = np.array(
    [
        [230, 57, 70],
        [29, 53, 87],
        [42, 157, 143],
        [244, 162, 97],
    ],
    dtype=np.uint8,
)
ERROR_COLORS = {
    "true_positive": np.array([42, 157, 143], dtype=np.uint8),
    "false_negative": np.array([230, 57, 70], dtype=np.uint8),
    "false_positive": np.array([244, 162, 97], dtype=np.uint8),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate segmentation explainability figures with error and difference maps."
    )

    parser.add_argument(
        "--train-csv",
        default="data/train.csv",
        help="Severstal segmentation training CSV.",
    )

    parser.add_argument(
        "--image-dir",
        default="data/train_images",
        help="Directory containing training images.",
    )

    parser.add_argument(
        "--baseline-checkpoint",
        default="checkpoints/baseline_unet_best.pth",
        help="Baseline U-Net checkpoint.",
    )

    parser.add_argument(
        "--cbam-checkpoint",
        default="checkpoints/cbam_unet_best.pth",
        help="CBAM U-Net checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/explainability/segmentation",
        help="Directory where segmentation explainability figures will be written.",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=6,
        help="Number of validation examples to visualize.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square image size used during segmentation training.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for binary mask predictions.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Validation split seed used during segmentation training.",
    )

    parser.add_argument(
        "--min-prediction-pixels",
        type=int,
        default=25,
        help="Minimum predicted positive pixels required in at least one model.",
    )

    return parser.parse_args()


def build_validation_dataset(args):
    train_df = pd.read_csv(args.train_csv)

    _, val_df = train_test_split(
        train_df,
        test_size=0.2,
        shuffle=True,
        random_state=args.seed,
    )

    return SteelSegmentationDataset(
        dataframe=val_df,
        image_dir=args.image_dir,
        transform=None,
        image_size=(args.image_size, args.image_size),
    )


def load_model(model_name, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if model_name == "baseline":
        model = UNetResNet18(num_classes=4, pretrained=False)
    elif model_name == "cbam":
        model = UNetCBAM(num_classes=4)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    state_dict = torch.load(checkpoint_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_mask(model, image_tensor, device, threshold):
    logits = model(image_tensor.unsqueeze(0).to(device))
    probabilities = torch.sigmoid(logits)[0].cpu().numpy()
    return (probabilities >= threshold).astype(np.uint8)


def tensor_to_image(image_tensor):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def tensor_to_mask(mask_tensor):
    return mask_tensor.cpu().numpy().astype(np.uint8)


def combine_mask(mask):
    return (mask.sum(axis=0) > 0).astype(np.uint8)


def colorize_mask(mask):
    color_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)

    for class_idx, color in enumerate(CLASS_COLORS):
        class_pixels = mask[class_idx] > 0
        color_mask[class_pixels] = color

    return color_mask


def make_overlay(image, mask, alpha=0.45):
    color_mask = colorize_mask(mask)
    active_pixels = color_mask.any(axis=2)
    overlay = image.copy()

    blended = (
        (1.0 - alpha) * image[active_pixels].astype(np.float32)
        + alpha * color_mask[active_pixels].astype(np.float32)
    )
    overlay[active_pixels] = np.clip(blended, 0, 255).astype(np.uint8)
    return overlay


def make_error_map(ground_truth, prediction):
    gt_any = combine_mask(ground_truth).astype(bool)
    pred_any = combine_mask(prediction).astype(bool)

    true_positive = gt_any & pred_any
    false_negative = gt_any & ~pred_any
    false_positive = ~gt_any & pred_any

    error_map = np.zeros((gt_any.shape[0], gt_any.shape[1], 3), dtype=np.uint8)
    error_map[true_positive] = ERROR_COLORS["true_positive"]
    error_map[false_negative] = ERROR_COLORS["false_negative"]
    error_map[false_positive] = ERROR_COLORS["false_positive"]

    return error_map


def make_difference_map(ground_truth, prediction):
    difference = np.zeros((ground_truth.shape[1], ground_truth.shape[2]), dtype=np.float32)

    for class_idx in range(ground_truth.shape[0]):
        gt = ground_truth[class_idx].astype(np.float32)
        pred = prediction[class_idx].astype(np.float32)
        difference += pred - gt

    difference = np.clip(difference, -1.0, 1.0)
    return difference


def calculate_pixel_counts(ground_truth, prediction):
    gt_any = combine_mask(ground_truth).astype(bool)
    pred_any = combine_mask(prediction).astype(bool)

    true_positive = int((gt_any & pred_any).sum())
    false_negative = int((gt_any & ~pred_any).sum())
    false_positive = int((~gt_any & pred_any).sum())
    true_negative = int((~gt_any & ~pred_any).sum())

    return {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
    }


def calculate_dice(ground_truth, prediction):
    gt_any = combine_mask(ground_truth).astype(np.float32)
    pred_any = combine_mask(prediction).astype(np.float32)
    intersection = float((gt_any * pred_any).sum())
    union = float(gt_any.sum() + pred_any.sum())

    if union == 0:
        return 1.0

    return (2.0 * intersection) / union


def select_examples(
    dataset,
    baseline_model,
    cbam_model,
    device,
    threshold,
    num_examples,
    min_prediction_pixels,
):
    selected = []
    fallback = []

    for index, image_id in enumerate(dataset.image_ids):
        image_tensor, mask_tensor = dataset[index]
        ground_truth = tensor_to_mask(mask_tensor)

        if ground_truth.sum() == 0:
            continue

        baseline_prediction = predict_mask(
            baseline_model,
            image_tensor,
            device,
            threshold,
        )
        cbam_prediction = predict_mask(
            cbam_model,
            image_tensor,
            device,
            threshold,
        )

        example = {
            "image_id": image_id,
            "image_tensor": image_tensor,
            "ground_truth": ground_truth,
            "baseline_prediction": baseline_prediction,
            "cbam_prediction": cbam_prediction,
        }
        fallback.append(example)

        predicted_pixels = baseline_prediction.sum() + cbam_prediction.sum()

        if predicted_pixels >= min_prediction_pixels:
            selected.append(example)

        if len(selected) == num_examples:
            return selected

    if len(selected) < num_examples:
        selected_ids = {example["image_id"] for example in selected}

        for example in fallback:
            if example["image_id"] in selected_ids:
                continue

            selected.append(example)

            if len(selected) == num_examples:
                break

    if len(selected) < num_examples:
        raise ValueError(
            f"Only found {len(selected)} validation examples with defects; "
            f"requested {num_examples}."
        )

    return selected


def add_error_legend(fig):
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=ERROR_COLORS["true_positive"] / 255.0,
            markersize=8,
            label="True positive",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=ERROR_COLORS["false_negative"] / 255.0,
            markersize=8,
            label="False negative",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=ERROR_COLORS["false_positive"] / 255.0,
            markersize=8,
            label="False positive",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)


def save_model_explainability_figure(
    image,
    ground_truth,
    prediction,
    model_label,
    image_id,
    output_path,
):
    error_map = make_error_map(ground_truth, prediction)
    difference_map = make_difference_map(ground_truth, prediction)
    dice = calculate_dice(ground_truth, prediction)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = [
        ("Original Image", image),
        ("Ground Truth Overlay", make_overlay(image, ground_truth)),
        ("Prediction Overlay", make_overlay(image, prediction)),
        ("Ground Truth Mask", colorize_mask(ground_truth)),
        ("Error Map", error_map),
        ("Difference Map", difference_map),
    ]

    for ax, (title, panel) in zip(axes.ravel(), panels):
        if title == "Difference Map":
            image_plot = ax.imshow(panel, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar(
                image_plot,
                ax=ax,
                fraction=0.046,
                pad=0.04,
                label="Prediction - Ground Truth",
            )
        else:
            ax.imshow(panel)

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{model_label} Explainability - {image_id} - Dice={dice:.3f}", y=0.98)
    add_error_legend(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0.08, 1, 0.94))
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(
    image,
    ground_truth,
    baseline_prediction,
    cbam_prediction,
    image_id,
    output_path,
):
    rows = [
        ("Baseline U-Net", baseline_prediction),
        ("CBAM U-Net", cbam_prediction),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(17, 7))

    for row_idx, (model_label, prediction) in enumerate(rows):
        difference_map = make_difference_map(ground_truth, prediction)
        panels = [
            ("Original", image),
            ("Ground Truth", make_overlay(image, ground_truth)),
            ("Prediction", make_overlay(image, prediction)),
            ("Error Map", make_error_map(ground_truth, prediction)),
            ("Difference", difference_map),
        ]

        for col_idx, (title, panel) in enumerate(panels):
            ax = axes[row_idx, col_idx]

            if title == "Difference":
                image_plot = ax.imshow(panel, cmap="coolwarm", vmin=-1, vmax=1)

                if row_idx == 1:
                    plt.colorbar(
                        image_plot,
                        ax=ax,
                        fraction=0.046,
                        pad=0.04,
                        label="Prediction - Ground Truth",
                    )
            else:
                ax.imshow(panel)

            ax.set_title(title if row_idx == 0 else "")
            ax.set_ylabel(model_label)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"Segmentation Explainability Comparison - {image_id}", y=0.98)
    add_error_legend(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0.08, 1, 0.94))
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_manifest(rows, output_dir):
    manifest_path = output_dir / "segmentation_explainability_manifest.csv"

    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "model",
                "dice",
                "true_positive_pixels",
                "false_negative_pixels",
                "false_positive_pixels",
                "figure_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


def write_explanation(output_dir):
    explanation_path = output_dir / "segmentation_explainability_notes.md"
    lines = [
        "# Segmentation Explainability Notes",
        "",
        "The error map compares the binary union of all four defect channels against the predicted union mask.",
        "",
        "Green pixels are true positives, where the model prediction overlaps the ground truth. Red pixels are false negatives, where a ground-truth defect was missed. Orange pixels are false positives, where the model predicted a defect outside the annotated mask.",
        "",
        "The difference map is computed as prediction minus ground truth for each class and clipped to the range [-1, 1]. Blue regions indicate missed defect pixels, while red regions indicate over-predicted defect pixels.",
        "",
        "These maps are intended to support qualitative discussion of segmentation behavior, including boundary errors, missed fine scratches, and over-segmentation of steel texture.",
        "",
    ]
    explanation_path.write_text("\n".join(lines))
    return explanation_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_validation_dataset(args)
    baseline_model = load_model("baseline", args.baseline_checkpoint, device)
    cbam_model = load_model("cbam", args.cbam_checkpoint, device)

    selected_examples = select_examples(
        dataset,
        baseline_model,
        cbam_model,
        device,
        args.threshold,
        args.num_examples,
        args.min_prediction_pixels,
    )

    manifest_rows = []

    for example_number, example in enumerate(selected_examples, start=1):
        image_id = example["image_id"]
        image = tensor_to_image(example["image_tensor"])
        ground_truth = example["ground_truth"]
        baseline_prediction = example["baseline_prediction"]
        cbam_prediction = example["cbam_prediction"]

        comparison_path = output_dir / f"example_{example_number:02d}_comparison.png"
        baseline_path = output_dir / f"example_{example_number:02d}_baseline.png"
        cbam_path = output_dir / f"example_{example_number:02d}_cbam.png"

        save_comparison_figure(
            image,
            ground_truth,
            baseline_prediction,
            cbam_prediction,
            image_id,
            comparison_path,
        )
        save_model_explainability_figure(
            image,
            ground_truth,
            baseline_prediction,
            "Baseline U-Net",
            image_id,
            baseline_path,
        )
        save_model_explainability_figure(
            image,
            ground_truth,
            cbam_prediction,
            "CBAM U-Net",
            image_id,
            cbam_path,
        )

        for model_label, prediction, figure_path in [
            ("baseline", baseline_prediction, baseline_path),
            ("cbam", cbam_prediction, cbam_path),
        ]:
            counts = calculate_pixel_counts(ground_truth, prediction)
            manifest_rows.append(
                {
                    "image_id": image_id,
                    "model": model_label,
                    "dice": f"{calculate_dice(ground_truth, prediction):.6f}",
                    "true_positive_pixels": counts["true_positive"],
                    "false_negative_pixels": counts["false_negative"],
                    "false_positive_pixels": counts["false_positive"],
                    "figure_path": figure_path.as_posix(),
                }
            )

        manifest_rows.append(
            {
                "image_id": image_id,
                "model": "comparison",
                "dice": "",
                "true_positive_pixels": "",
                "false_negative_pixels": "",
                "false_positive_pixels": "",
                "figure_path": comparison_path.as_posix(),
            }
        )

    manifest_path = write_manifest(manifest_rows, output_dir)
    explanation_path = write_explanation(output_dir)

    print(f"Wrote {len(manifest_rows)} segmentation explainability figures")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {explanation_path}")


if __name__ == "__main__":
    main()

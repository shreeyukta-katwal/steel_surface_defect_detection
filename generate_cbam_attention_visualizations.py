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

from src.datasets.classification_dataset import ClassificationDataset
from src.datasets.transforms import get_val_classification_transform
from src.models.cbam_resnet18 import CBAMResNet18


CLASS_COLUMNS = ["class_1", "class_2", "class_3", "class_4"]
CLASS_NAMES = ["Class 1", "Class 2", "Class 3", "Class 4"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CBAM channel and spatial attention visualizations."
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
        "--cbam-checkpoint",
        default="checkpoints/cbam_best.pth",
        help="CBAM-ResNet18 checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/attention/cbam",
        help="Directory where CBAM attention figures will be written.",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=6,
        help="Number of defective validation examples to visualize.",
    )

    parser.add_argument(
        "--top-channels",
        type=int,
        default=20,
        help="Number of highest-weighted channels to show.",
    )

    return parser.parse_args()


def load_cbam_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = CBAMResNet18(pretrained=False, num_classes=4)
    state_dict = torch.load(checkpoint_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


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


def tensor_to_display_image(image_tensor):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * IMAGENET_STD) + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def normalize_map(attention_map):
    attention_map = attention_map.astype(np.float32)
    attention_map -= attention_map.min()
    max_value = attention_map.max()

    if max_value > 0:
        attention_map /= max_value

    return attention_map


def colorize_attention(attention_map, image_shape):
    resized = cv2.resize(attention_map, (image_shape[1], image_shape[0]))
    heatmap = np.uint8(255 * normalize_map(resized))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def overlay_attention(image, attention_map, alpha=0.45):
    heatmap = colorize_attention(attention_map, image.shape[:2])
    overlay = ((1.0 - alpha) * image.astype(np.float32)) + (
        alpha * heatmap.astype(np.float32)
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


@torch.no_grad()
def extract_cbam_attention(model, image_tensor, device):
    image_batch = image_tensor.unsqueeze(0).to(device)

    features = model.features(image_batch)
    channel_module = model.cbam.channel_attention
    spatial_module = model.cbam.spatial_attention

    avg_out = channel_module.mlp(channel_module.avg_pool(features))
    max_out = channel_module.mlp(channel_module.max_pool(features))
    channel_attention = channel_module.sigmoid(avg_out + max_out)
    channel_refined = features * channel_attention

    spatial_avg = torch.mean(channel_refined, dim=1, keepdim=True)
    spatial_max, _ = torch.max(channel_refined, dim=1, keepdim=True)
    spatial_input = torch.cat([spatial_avg, spatial_max], dim=1)
    spatial_attention = spatial_module.sigmoid(spatial_module.conv(spatial_input))

    refined = channel_refined * spatial_attention
    logits = model.classifier(torch.flatten(model.avgpool(refined), 1))

    return {
        "channel_attention": channel_attention[0, :, 0, 0].cpu().numpy(),
        "spatial_attention": spatial_attention[0, 0].cpu().numpy(),
        "probabilities": torch.sigmoid(logits)[0].cpu().numpy(),
    }


def save_attention_figure(
    image,
    attention,
    image_id,
    target_class,
    output_path,
    top_channels,
):
    channel_attention = attention["channel_attention"]
    spatial_attention = attention["spatial_attention"]
    probabilities = attention["probabilities"]

    top_indices = np.argsort(channel_attention)[-top_channels:][::-1]
    top_values = channel_attention[top_indices]
    low_indices = np.argsort(channel_attention)[:top_channels]
    low_values = channel_attention[low_indices]

    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 4)

    original_ax = fig.add_subplot(grid[0, 0])
    spatial_ax = fig.add_subplot(grid[0, 1])
    overlay_ax = fig.add_subplot(grid[0, 2:])
    top_channel_ax = fig.add_subplot(grid[1, :2])
    low_channel_ax = fig.add_subplot(grid[1, 2:])

    original_ax.imshow(image)
    original_ax.set_title("Original Image")
    original_ax.axis("off")

    spatial_ax.imshow(colorize_attention(spatial_attention, image.shape[:2]))
    spatial_ax.set_title("Spatial Attention")
    spatial_ax.axis("off")

    overlay_ax.imshow(overlay_attention(image, spatial_attention))
    overlay_ax.set_title("Spatial Attention Overlay")
    overlay_ax.axis("off")

    top_channel_labels = [str(index) for index in top_indices]
    top_channel_ax.bar(top_channel_labels, top_values, color="#2a9d8f")
    top_channel_ax.set_ylim(0.0, 1.0)
    top_channel_ax.set_xlabel("Channel Index")
    top_channel_ax.set_ylabel("Attention Weight")
    top_channel_ax.set_title(f"Most Amplified {top_channels} Channels")
    top_channel_ax.grid(axis="y", alpha=0.25)
    top_channel_ax.tick_params(axis="x", labelrotation=45)

    low_channel_labels = [str(index) for index in low_indices]
    low_channel_ax.bar(low_channel_labels, low_values, color="#e76f51")
    low_channel_ax.set_ylim(0.0, 1.0)
    low_channel_ax.set_xlabel("Channel Index")
    low_channel_ax.set_ylabel("Attention Weight")
    low_channel_ax.set_title(f"Most Suppressed {top_channels} Channels")
    low_channel_ax.grid(axis="y", alpha=0.25)
    low_channel_ax.tick_params(axis="x", labelrotation=45)

    target_name = CLASS_NAMES[target_class]
    target_probability = probabilities[target_class]
    fig.suptitle(
        f"CBAM Attention - {image_id} - {target_name} p={target_probability:.3f}",
        y=0.98,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_manifest(rows, output_dir):
    manifest_path = output_dir / "cbam_attention_manifest.csv"

    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "target_class",
                "target_probability",
                "mean_channel_attention",
                "max_channel_attention",
                "mean_spatial_attention",
                "max_spatial_attention",
                "figure_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


def write_explanation(output_dir):
    explanation_path = output_dir / "cbam_attention_explanation.md"

    lines = [
        "# CBAM Attention Visualization Notes",
        "",
        "Channel attention assigns a learned weight to each high-level feature channel after the ResNet18 encoder. Higher channel weights indicate feature detectors that CBAM considered more relevant for the current steel image.",
        "",
        "Spatial attention then compresses the channel-refined feature map using average and max pooling across channels. The resulting heatmap highlights image regions that the model emphasizes before classification.",
        "",
        "In the generated figures, warmer spatial-attention regions indicate stronger CBAM focus. When these regions coincide with visible scratches, stains, edge defects, or texture disruptions, the visualization supports the interpretation that CBAM is directing the classifier toward defect-relevant image areas rather than only background steel texture.",
        "",
        "The channel bar plot should be interpreted as feature importance within the latent encoder representation, not as direct pixel-level localization. The spatial overlay provides the localization cue.",
        "",
    ]

    explanation_path.write_text("\n".join(lines))
    return explanation_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_csv(args.val_csv)
    selected_df = select_examples(val_df, args.num_examples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cbam_model(args.cbam_checkpoint, device)

    manifest_rows = []

    for example_number, row in selected_df.iterrows():
        single_row_df = pd.DataFrame([row])
        dataset = ClassificationDataset(
            single_row_df,
            args.image_dir,
            transform=get_val_classification_transform(),
        )
        image_tensor, _ = dataset[0]
        image = tensor_to_display_image(image_tensor)
        image_id = row["ImageId"]
        target_class = choose_target_class(row)

        attention = extract_cbam_attention(model, image_tensor, device)
        figure_path = output_dir / f"example_{example_number + 1:02d}_cbam_attention.png"

        save_attention_figure(
            image,
            attention,
            image_id,
            target_class,
            figure_path,
            args.top_channels,
        )

        channel_attention = attention["channel_attention"]
        spatial_attention = attention["spatial_attention"]
        probabilities = attention["probabilities"]

        manifest_rows.append(
            {
                "image_id": image_id,
                "target_class": CLASS_NAMES[target_class],
                "target_probability": f"{probabilities[target_class]:.6f}",
                "mean_channel_attention": f"{channel_attention.mean():.6f}",
                "max_channel_attention": f"{channel_attention.max():.6f}",
                "mean_spatial_attention": f"{spatial_attention.mean():.6f}",
                "max_spatial_attention": f"{spatial_attention.max():.6f}",
                "figure_path": figure_path.as_posix(),
            }
        )

    manifest_path = write_manifest(manifest_rows, output_dir)
    explanation_path = write_explanation(output_dir)

    print(f"Wrote {len(manifest_rows)} CBAM attention figures")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {explanation_path}")


if __name__ == "__main__":
    main()

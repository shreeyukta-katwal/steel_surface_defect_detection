# Steel Surface Defect Detection

A prototype for detecting defects in Severstal steel surface images. It compares baseline and CBAM-enhanced models for two tasks:

- **Multilabel classification:** ResNet18 and CBAM-ResNet18 predict the presence of four defect classes in an image.
- **Segmentation:** ResNet18-encoder U-Net and a CBAM variant predict four separate defect-mask channels.

The repository includes the training implementation, a [Jupyter notebook](CET313_steel_defect_prototype.ipynb), and archived validation results. The notebook explains the pipeline and shows the saved histories, figures, and selected predictions without starting a new training run.

## Results at a glance

| Task | Baseline | CBAM variant | Basis |
| --- | ---: | ---: | --- |
| Classification macro F1 | 0.8445 ± 0.0159 | 0.8526 ± 0.0232 | Mean ± sample SD across three seeds; epochs selected by minimum validation loss |
| Segmentation Dice | 0.8372 | 0.8368 | Final validation epoch of one run per model |

These are **validation results**, not independent test results. The logged peak classification macro F1 values are higher, but those epochs were selected retrospectively from the same validation histories; corresponding model weights have not been verified as saved. The segmentation variants also differ in their final upsampling paths, so their difference cannot be attributed to CBAM alone.

## Repository layout

```text
CET313_steel_defect_prototype.ipynb  Notebook and archived results
train_classifier.py                  Classification training entry point
train_segmentation.py                Segmentation training entry point
src/                                 Datasets, transforms, models, losses, and trainers
logs/                                Saved epoch histories
figures/                             Saved result figures
results/                             Selected predictions and visual explanations
```

The exact contents of `figures/` and `results/` depend on the packaged version of this repository. Keep these directories alongside the notebook so its relative image links work.

## Open the notebook

1. Clone or download this repository and open its root folder in VS Code or Jupyter.
2. Select a Python environment with Jupyter support. The archived evidence cells use Python's standard library and IPython display tools; model evaluation additionally needs the libraries imported by the project source, including PyTorch, torchvision, pandas, scikit-learn, Albumentations, and OpenCV.
3. Open `CET313_steel_defect_prototype.ipynb`. Review its saved tables and figures, then run only the setup and archived-evidence cells if you want to audit the included CSV histories.

**Training is disabled by default.** The final training cell requires an explicit opt-in and may take a long time. Do not run it to inspect the existing results.

## Data and model files

The original Severstal images and annotations are obtained separately from the [Severstal Steel Defect Detection competition](https://www.kaggle.com/c/severstal-steel-defect-detection), subject to its access terms. The prepared image splits, original checkpoint files, and exact historical dependency versions were not available in the reviewed package. The notebook's optional model-loading cells therefore report missing prerequisites unless the original files are supplied at the paths shown there.

The archived CSV histories and figures allow inspection of the recorded validation results. They do not, by themselves, reproduce training or provide a fresh evaluation. In particular, the segmentation training script splits annotation rows before grouping by image; image-level separation should be audited before using its validation scores as evidence of generalization.

## Method summary

Classification uses RGB images resized to 256 × 256, four sigmoid outputs, Focal Loss, and a 0.5 prediction threshold. Three seeds (42, 0, and 7) were logged for each classifier. The current training code uses Adam, `ReduceLROnPlateau`, early stopping, and a minimum-validation-loss checkpoint rule.

Segmentation uses four binary mask channels, a ResNet18 encoder, a skip-connected decoder, and a combined soft Dice/BCE loss. One 40-epoch history is available for each segmentation model. The CBAM model adds bottleneck attention, but its final upsampling implementation also differs from the baseline.

## Project status

This is an experimental coursework prototype. The included results describe the recorded validation runs. A stronger comparison would verify disjoint image-level splits, use identical decoder paths for the segmentation ablation, retain complete run configurations and checkpoints, and evaluate frozen models on an untouched test set.

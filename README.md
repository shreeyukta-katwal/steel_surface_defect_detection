# CET313 steel defect detection code submission

Open `CET313_steel_defect_prototype.ipynb` in VS Code or Jupyter with this folder as the working directory. It includes the original prototype training scripts as commented source, archived validation tables, saved figures and selected prediction examples. Saved code-cell outputs are genuine Jupyter kernel results from read-only checks. The training cell is disabled by default.

## Contents

- `train_classifier.py`, `train_segmentation.py`, `dataset_audit.py` and `src/`: original implementation and preprocessing modules.
- `logs/`: eight original classification and segmentation histories, the primary source of notebook metrics.
- `results/`: original CSV summaries/manifests and only the selected prediction images displayed in the notebook.
- `figures/`: only the five historical figures displayed in the notebook.

## Safe use

Cells 2, 4, 6, 8, 16 and 23 (zero-based) perform setup, source/file checks and archived evidence reads. Cell 8 defines a guarded training helper; it does not call it. The final training cell 27 is deliberately unexecuted with `ALLOW_LONG_TRAINING = False`. Do not enable it for this submission. Optional evaluation cells 19 and 21 are unexecuted in this clean copy. They load checkpoints only if the original files are supplied, and any result would be a **new** evaluation rather than a historical result.

Safe evidence cells need Python, Jupyter/IPython and standard-library modules. Optional model evaluation needs compatible PyTorch, torchvision, Albumentations, OpenCV, NumPy, pandas, scikit-learn and tqdm. Replaying the original split script additionally needs `iterative-stratification`; `dataset_audit.py` imports matplotlib. Exact historical dependency versions were not recorded; no package lock is available.

## Original files unavailable

The supplied project contains no `data/train_images/`, `data/train.csv`, `data/processed/master_labels.csv`, `data/processed/train_split.csv`, `data/processed/val_split.csv`, or `data/processed/test_split.csv`. It also contains no `checkpoints/*.pth`: the six classifier seed/model weights and two seed-42 U-Net weights are absent. These files are required for model re-evaluation and independent verification of image counts and splits; they are not required to open this notebook or inspect the archived results. Supply only the exact original run files if reproducibility is required. Do not substitute retrained weights or regenerated splits as historical evidence.

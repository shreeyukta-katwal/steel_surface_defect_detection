import os
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

DATA_PATH = "data/processed/master_labels.csv"
SAVE_DIR = "data/processed"

os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df["ImageId"].values

y = df[
    [
        "class_1",
        "class_2",
        "class_3",
        "class_4",
    ]
].values

# -----------------------------
# Train 70%
# Temp 30%
# -----------------------------

msss = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, temp_idx = next(
    msss.split(X, y)
)

train_df = df.iloc[train_idx]

temp_df = df.iloc[temp_idx]

# -----------------------------
# Validation 15%
# Test 15%
# -----------------------------

X_temp = temp_df["ImageId"].values

y_temp = temp_df[
    [
        "class_1",
        "class_2",
        "class_3",
        "class_4",
    ]
].values

msss2 = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_idx, test_idx = next(
    msss2.split(X_temp, y_temp)
)

val_df = temp_df.iloc[val_idx]

test_df = temp_df.iloc[test_idx]

train_df.to_csv(
    os.path.join(SAVE_DIR, "train_split.csv"),
    index=False,
)

val_df.to_csv(
    os.path.join(SAVE_DIR, "val_split.csv"),
    index=False,
)

test_df.to_csv(
    os.path.join(SAVE_DIR, "test_split.csv"),
    index=False,
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))
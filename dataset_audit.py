import os
import pandas as pd
import matplotlib.pyplot as plt


# Paths

DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "train_images")
SAVE_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================================
# Load positive annotations
# ==========================================================

positive_df = pd.read_csv(TRAIN_CSV)

print("=" * 60)
print("DATASET AUDIT")
print("=" * 60)

print(f"Positive annotation rows : {len(positive_df)}")

# ==========================================================
# Create image-level multi-label table from positives
# ==========================================================

positive_labels = (
    positive_df.assign(value=1)
    .pivot_table(
        index="ImageId",
        columns="ClassId",
        values="value",
        aggfunc="max",
        fill_value=0
    )
)

# Ensure all four defect columns exist
for c in [1, 2, 3, 4]:
    if c not in positive_labels.columns:
        positive_labels[c] = 0

positive_labels = positive_labels[[1, 2, 3, 4]]

positive_labels.columns = [
    "class_1",
    "class_2",
    "class_3",
    "class_4"
]

# ==========================================================
# Get ALL images from disk
# ==========================================================

image_files = sorted([
    f for f in os.listdir(TRAIN_IMAGE_DIR)
    if f.lower().endswith(".jpg")
])

print(f"Images found on disk     : {len(image_files)}")

master = pd.DataFrame({
    "ImageId": image_files
})

# ==========================================================
# Merge labels
# ==========================================================

master = master.merge(
    positive_labels,
    how="left",
    left_on="ImageId",
    right_index=True
)

# Images absent from train.csv are defect-free
master[[
    "class_1",
    "class_2",
    "class_3",
    "class_4"
]] = master[[
    "class_1",
    "class_2",
    "class_3",
    "class_4"
]].fillna(0).astype(int)

# ==========================================================
# Derived columns
# ==========================================================

master["num_defects"] = master[
    ["class_1", "class_2", "class_3", "class_4"]
].sum(axis=1)

master["has_defect"] = (
    master["num_defects"] > 0
).astype(int)

# ==========================================================
# Save
# ==========================================================

save_path = os.path.join(
    SAVE_DIR,
    "master_labels.csv"
)

master.to_csv(save_path, index=False)

# ==========================================================
# Statistics
# ==========================================================

print("\n========== SUMMARY ==========")

print(f"Total images          : {len(master)}")
print(f"Images with defects   : {master['has_defect'].sum()}")
print(f"Images without defects: {(master['has_defect']==0).sum()}")

print("\nPer-class image counts")

for i in range(1,5):
    print(
        f"Class {i}:",
        master[f"class_{i}"].sum()
    )

print("\nImages containing N defect classes")

print(
    master["num_defects"]
    .value_counts()
    .sort_index()
)

# ==========================================================
# Verify completeness
# ==========================================================

assert len(master) == len(image_files), \
    "ERROR: Some images are missing."

assert master["ImageId"].nunique() == len(image_files), \
    "Duplicate ImageIds found."

print("\n✓ Verification passed.")
print("✓ Every training image has a label.")

# ==========================================================
# Plot class counts
# ==========================================================

plt.figure(figsize=(7,4))

counts = [
    master["class_1"].sum(),
    master["class_2"].sum(),
    master["class_3"].sum(),
    master["class_4"].sum()
]

plt.bar(
    ["1","2","3","4"],
    counts
)

plt.title("Images containing each defect class")

plt.xlabel("Defect Class")

plt.ylabel("Images")

plt.tight_layout()

plt.show()

# ==========================================================
# Plot overlap
# ==========================================================

plt.figure(figsize=(7,4))

master["num_defects"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Number of defect classes per image")

plt.xlabel("Classes present")

plt.ylabel("Images")

plt.tight_layout()

plt.show()

print("\nMaster labels saved to:")
print(save_path)
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.rle import rle_decode


class SteelSegmentationDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        transform=None,
        image_size=(256, 256),
    ):

        self.df = dataframe.copy()

        self.image_dir = image_dir

        self.transform = transform

        self.image_size = image_size

        # unique images only
        self.image_ids = sorted(self.df["ImageId"].unique())

        # group rows by image
        self.groups = self.df.groupby("ImageId")

    def __len__(self):

        return len(self.image_ids)

    def __getitem__(self, index):

        image_id = self.image_ids[index]

        image_path = os.path.join(
            self.image_dir,
            image_id
        )

        image = cv2.imread(image_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        original_h, original_w = image.shape[:2]

        mask = np.zeros(
            (
                original_h,
                original_w,
                4
            ),
            dtype=np.uint8
        )

        rows = self.groups.get_group(image_id)

        for _, row in rows.iterrows():

            class_id = int(row["ClassId"])

            rle = row["EncodedPixels"]

            if pd.isna(rle):
                continue

            decoded = rle_decode(
                rle,
                shape=(original_h, original_w)
            )

            mask[:, :, class_id - 1] = decoded

        image = cv2.resize(
            image,
            self.image_size
        )

        resized_mask = np.zeros(
            (
                self.image_size[1],
                self.image_size[0],
                4
            ),
            dtype=np.uint8
        )

        for c in range(4):

            resized_mask[:, :, c] = cv2.resize(
                mask[:, :, c],
                self.image_size,
                interpolation=cv2.INTER_NEAREST
            )

        if self.transform:

            transformed = self.transform(
                image=image,
                mask=resized_mask
            )

            image = transformed["image"]

            resized_mask = transformed["mask"]

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2, 0, 1)

        image = image / 255.0

        mask = torch.tensor(
            resized_mask,
            dtype=torch.float32
        ).permute(2, 0, 1)

        return image, mask
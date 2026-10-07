import os

import cv2

import torch

from torch.utils.data import Dataset


class ClassificationDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        transform=None,
    ):

        self.df = dataframe.reset_index(drop=True)

        self.image_dir = image_dir

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = os.path.join(
            self.image_dir,
            row.ImageId
        )

        image = cv2.imread(img_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        label = row[
            [
                "class_1",
                "class_2",
                "class_3",
                "class_4",
            ]
        ].values.astype("float32")

        if self.transform:

            image = self.transform(
                image=image
            )["image"]

        return image, torch.tensor(label)
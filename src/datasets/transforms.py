import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_SIZE = 256


def get_train_classification_transform():

    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.2),

        A.RandomBrightnessContrast(
            p=0.3
        ),

        A.Normalize(),

        ToTensorV2()
    ])


def get_val_classification_transform():

    return A.Compose([
        A.Resize(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        A.Normalize(),

        ToTensorV2()
    ])


# --------------------------
# Segmentation
# --------------------------

def get_train_segmentation_transform():

    return A.Compose([
        A.Resize(
            256,
            800
        ),

        A.HorizontalFlip(
            p=0.5
        ),

        A.RandomBrightnessContrast(
            p=0.3
        ),

        A.Normalize(),

        ToTensorV2()
    ])


def get_val_segmentation_transform():

    return A.Compose([
        A.Resize(
            256,
            800
        ),

        A.Normalize(),

        ToTensorV2()
    ])
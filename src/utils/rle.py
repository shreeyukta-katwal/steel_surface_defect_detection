import numpy as np


def rle_decode(mask_rle, shape=(256, 1600)):
    """
    Decode a run-length encoded mask.

    Parameters
    ----------
    mask_rle : str or float
        RLE string from train.csv.
        May be NaN if no defect.

    shape : tuple
        (height, width)

    Returns
    -------
    numpy.ndarray
        Binary mask of shape (H, W)
    """

    if mask_rle is np.nan:
        return np.zeros(shape, dtype=np.uint8)

    if not isinstance(mask_rle, str):
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.strip().split()

    starts = np.asarray(s[0::2], dtype=int)
    lengths = np.asarray(s[1::2], dtype=int)

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Kaggle Severstal uses column-major ordering
    return img.reshape(shape[::-1]).T


def rle_encode(mask):
    """
    Encode binary mask into RLE.

    Parameters
    ----------
    mask : ndarray
        Binary mask (H,W)

    Returns
    -------
    str
        RLE string
    """

    pixels = mask.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)
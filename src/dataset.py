
# ============================================================
# dataset.py — PyTorch Dataset for Assam flood patches
# ============================================================

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY


class AssamFloodDataset(Dataset):
    """
    Dataset for Assam flood patches.

    Args:
        images:       NumPy array shaped (N, C, H, W).
        labels:       NumPy array shaped (N, H, W) or (N, 1, H, W).
        district_ids: Optional array shaped (N,).
        augment:      Apply spatial augmentation to training samples.

    Returns:
        image:       Float tensor shaped (C, H, W).
        label:       Float tensor shaped (H, W), containing 0/1 labels.
        district_id: Integer district/grid-cell ID.
    """

    def __init__(
        self,
        images,
        labels,
        district_ids=None,
        augment=False
    ):
        self.images = np.asarray(images, dtype=np.float32)
        self.labels = np.asarray(labels)
        self.augment = augment

        if self.images.ndim != 4:
            raise ValueError(
                "images must have shape (N, C, H, W); "
                f"received {self.images.shape}"
            )

        if self.labels.ndim == 4 and self.labels.shape[1] == 1:
            self.labels = self.labels[:, 0]

        if self.labels.ndim != 3:
            raise ValueError(
                "labels must have shape (N, H, W) or (N, 1, H, W); "
                f"received {self.labels.shape}"
            )

        if len(self.images) != len(self.labels):
            raise ValueError(
                "Number of image patches and label patches differs: "
                f"{len(self.images)} images, {len(self.labels)} labels"
            )

        if self.images.shape[2:] != self.labels.shape[1:]:
            raise ValueError(
                "Image and label patch dimensions do not match: "
                f"{self.images.shape[2:]} vs {self.labels.shape[1:]}"
            )

        if not np.isfinite(self.images).all():
            raise ValueError(
                "Images contain NaN or infinite values. "
                "Clean or impute invalid pixels before creating the dataset."
            )

        if not np.isfinite(self.labels).all():
            raise ValueError(
                "Labels contain NaN or infinite values. "
                "Handle invalid label pixels before creating the dataset."
            )

        unique_labels = np.unique(self.labels)
        if not np.isin(unique_labels, [0, 1]).all():
            raise ValueError(
                "Flood labels must be binary (0 or 1). "
                f"Found values: {unique_labels[:10]}"
            )

        if district_ids is None:
            self.district_ids = np.zeros(
                len(self.images), dtype=np.int64
            )
        else:
            self.district_ids = np.asarray(
                district_ids, dtype=np.int64
            )

            if self.district_ids.shape != (len(self.images),):
                raise ValueError(
                    "district_ids must contain one ID per patch; "
                    f"received shape {self.district_ids.shape}"
                )

        if self.district_ids.size and np.any(self.district_ids < 0):
            raise ValueError("District IDs cannot be negative.")

        # Spatial transforms are applied identically to features and labels.
        # Avoid brightness/color transforms on calibrated satellite features.
        self.transform = (
            A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
            if augment
            else None
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # (C, H, W) -> (H, W, C) for Albumentations.
        image = self.images[idx].transpose(1, 2, 0).copy()
        label = self.labels[idx].astype(np.float32, copy=True)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        # Return contiguous PyTorch tensors.
        image = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        ).float()

        label = torch.from_numpy(
            np.ascontiguousarray(label)
        ).float()

        district_id = int(self.district_ids[idx])

        return image, label, district_id


def assign_district_ids(
    coords,
    assam_shapefile_path=None,
    n_districts=35,
    total_rows=None,
    total_cols=None
):
    """
    Assign grid-cell IDs to patches based on raster coordinates.

    IMPORTANT:
    These are approximate grid-cell IDs, NOT actual Assam
    administrative district assignments. A district shapefile
    and spatial join are required for genuine district IDs.
    """

    if len(coords) == 0:
        return np.empty(0, dtype=np.int64)

    if total_rows is None or total_cols is None:
        return np.zeros(len(coords), dtype=np.int64)

    if total_rows <= 0 or total_cols <= 0:
        raise ValueError("Raster dimensions must be positive.")

    n_row_blocks = 5
    n_col_blocks = max(1, int(np.ceil(n_districts / n_row_blocks)))

    row_size = max(1, int(np.ceil(total_rows / n_row_blocks)))
    col_size = max(1, int(np.ceil(total_cols / n_col_blocks)))

    district_ids = []

    for row, col in coords:
        if row < 0 or col < 0 or row >= total_rows or col >= total_cols:
            raise ValueError(
                f"Patch coordinate {(row, col)} lies outside "
                f"the raster ({total_rows}, {total_cols})."
            )

        r_block = min(row // row_size, n_row_blocks - 1)
        c_block = min(col // col_size, n_col_blocks - 1)

        district_id = r_block * n_col_blocks + c_block
        district_ids.append(min(district_id, n_districts - 1))

    return np.asarray(district_ids, dtype=np.int64)


def get_dataloaders(
    splits,
    district_ids_train=None,
    district_ids_val=None,
    district_ids_test=None
):
    """
    Build DataLoaders from the train/validation/test split dictionary.
    """

    required_keys = [
        "X_train", "y_train",
        "X_val", "y_val",
        "X_test", "y_test"
    ]

    missing_keys = [key for key in required_keys if key not in splits]

    if missing_keys:
        raise KeyError(
            f"Missing keys in splits dictionary: {missing_keys}"
        )

    train_ds = AssamFloodDataset(
        splits["X_train"],
        splits["y_train"],
        district_ids=district_ids_train,
        augment=True
    )

    val_ds = AssamFloodDataset(
        splits["X_val"],
        splits["y_val"],
        district_ids=district_ids_val,
        augment=False
    )

    test_ds = AssamFloodDataset(
        splits["X_test"],
        splits["y_test"],
        district_ids=district_ids_test,
        augment=False
    )

    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
    }

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=False,
        **loader_options
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        **loader_options
    )

    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        drop_last=False,
        **loader_options
    )

    print(
        "DataLoaders created | "
        f"Train: {len(train_loader)} batches | "
        f"Validation: {len(val_loader)} batches | "
        f"Test: {len(test_loader)} batches"
    )

    return train_loader, val_loader, test_loader

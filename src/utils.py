"""
Small project utilities shared across notebooks and modules.

Provides data loading, train/test splitting, feature extraction
(HOG, LBP), PCA, PyTorch dataloader helpers, and Kaggle submission
generation for the Dogs vs. Cats classification pipeline (Part 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm

from src.config import (
    PART1_TRAIN_DIR,
    PART1_TEST_DIR,
    LABEL_MAP,
    IMG_SIZE_CLASSICAL,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_labeled_images(
    img_size: tuple[int, int] = IMG_SIZE_CLASSICAL,
    grayscale: bool = False,
    max_samples: int | None = None,
    return_ids: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
):
    """Load labeled cat/dog images from the Part 1 training directory.

    Parameters
    ----------
    img_size : tuple
        Target (height, width) for resizing.
    grayscale : bool
        If True, load as single-channel grayscale.
    max_samples : int or None
        Cap the number of images per class (useful for quick debugging).
    return_ids : bool
        If True, also return filename stems (e.g. ``cat.12``) in dataset order.

    Returns
    -------
    X : np.ndarray, float32, shape (N, H, W) or (N, H, W, 3), values in [0, 1]
    y : np.ndarray, int, shape (N,) — 0 for cat, 1 for dog
    ids : np.ndarray, optional, shape (N,), dtype object — filename stems without extension
    """
    images: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []

    for class_name, label in LABEL_MAP.items():
        class_dir = PART1_TRAIN_DIR / f"{class_name}s"
        paths = sorted(class_dir.glob("*.jpg"))
        if max_samples is not None:
            paths = paths[:max_samples]

        for p in tqdm(paths, desc=f"Loading {class_name}s"):
            flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            img = cv2.imread(str(p), flag)
            if img is None:
                continue
            img = cv2.resize(img, (img_size[1], img_size[0]))
            if not grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)
            ids.append(p.stem)

    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.int64)
    if return_ids:
        return X, y, np.array(ids, dtype=object)
    return X, y


def load_test_images(
    img_size: tuple[int, int] = IMG_SIZE_CLASSICAL,
    grayscale: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """Load unlabeled Kaggle test images.

    Returns
    -------
    X : np.ndarray, float32, values in [0, 1]
    ids : list[int] — numeric image ids parsed from filenames, sorted ascending
    """
    paths = sorted(PART1_TEST_DIR.glob("*.jpg"), key=lambda p: int(p.stem))
    images: list[np.ndarray] = []
    ids: list[int] = []

    for p in tqdm(paths, desc="Loading test images"):
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(str(p), flag)
        if img is None:
            continue
        img = cv2.resize(img, (img_size[1], img_size[0]))
        if not grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        ids.append(int(p.stem))

    X = np.array(images, dtype=np.float32) / 255.0
    return X, ids


# ---------------------------------------------------------------------------
# Train / test splitting
# ---------------------------------------------------------------------------

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a float32 [0,1] image to uint8 grayscale."""
    if img.ndim == 3:
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return (img * 255).astype(np.uint8)


def extract_hog_features(
    images: np.ndarray,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
    orientations: int = 9,
) -> np.ndarray:
    """Compute HOG feature vectors for a batch of images.

    Returns
    -------
    features : np.ndarray, shape (n_samples, n_hog_features)
    """
    feats: list[np.ndarray] = []
    for img in tqdm(images, desc="Extracting HOG"):
        gray = _to_gray_uint8(img)
        fd = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            feature_vector=True,
        )
        feats.append(fd)
    return np.array(feats, dtype=np.float32)


def extract_lbp_features(
    images: np.ndarray,
    n_points: int = 24,
    radius: int = 3,
    n_bins: int | None = None,
) -> np.ndarray:
    """Compute LBP histogram features for a batch of images.

    Returns
    -------
    features : np.ndarray, shape (n_samples, n_bins)
    """
    if n_bins is None:
        n_bins = n_points + 2
    feats: list[np.ndarray] = []
    for img in tqdm(images, desc="Extracting LBP"):
        gray = _to_gray_uint8(img)
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
        )
        feats.append(hist)
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# PCA (appearance-based / dimensionality reduction)
# ---------------------------------------------------------------------------

def apply_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 100,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Flatten images or feature rows, fit PCA on training data, transform both splits.

    Returns
    -------
    X_train_pca : np.ndarray, shape (n_train, n_components)
    X_test_pca  : np.ndarray, shape (n_test, n_components)
    pca         : fitted PCA object
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(n_test, -1)

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)
    return X_train_pca, X_test_pca, pca


def standardize_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit ``StandardScaler`` on training rows only; transform train and test."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# ---------------------------------------------------------------------------
# Kaggle submission
# ---------------------------------------------------------------------------

def generate_submission_csv(
    ids: Sequence[int],
    predictions: Sequence[float] | np.ndarray,
    output_path: str | Path,
) -> Path:
    """Write a Kaggle-format submission CSV (columns: id, label).

    Parameters
    ----------
    ids : sequence of int
        Image ids matching the test set filenames.
    predictions : sequence of float
        Model predictions (probability of dog, or 0/1 labels).
    output_path : path
        Where to save the CSV file.

    Returns
    -------
    Path to the written CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"id": ids, "label": predictions})
    df = df.sort_values("id").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return output_path


# ---------------------------------------------------------------------------
# PyTorch helpers (Option C)
# ---------------------------------------------------------------------------

def get_pytorch_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    img_size: tuple[int, int] = (224, 224),
):
    """Wrap numpy arrays into PyTorch DataLoaders with ImageNet normalization.

    Images are expected as float32 [0, 1] in (N, H, W, 3) format.
    Returns (train_loader, val_loader).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def _prepare(X: np.ndarray, y: np.ndarray) -> TensorDataset:
        # (N, H, W, 3) -> (N, 3, H, W)
        t = torch.from_numpy(X).permute(0, 3, 1, 2).float()
        if (t.shape[2], t.shape[3]) != img_size:
            t = torch.nn.functional.interpolate(
                t, size=img_size, mode="bilinear", align_corners=False
            )
        t = torch.stack([normalize(img) for img in t])
        labels = torch.from_numpy(y).long()
        return TensorDataset(t, labels)

    train_ds = _prepare(X_train, y_train)
    val_ds = _prepare(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

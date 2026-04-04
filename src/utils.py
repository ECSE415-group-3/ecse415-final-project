"""
Small project utilities shared across notebooks and modules.

Provides data loading, train/test splitting, feature extraction
(HOG, LBP), PCA, PyTorch dataloader helpers, and Kaggle submission
generation for the Dogs vs. Cats classification pipeline (Part 1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm

from src.config import (
    OUTPUTS_DIR,
    PART1_TRAIN_DIR,
    PART1_TEST_DIR,
    LABEL_MAP,
    IMG_SIZE_CLASSICAL,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _resize_image_to_shape(
    img: np.ndarray,
    out_h: int,
    out_w: int,
    grayscale: bool,
    letterbox: bool,
) -> np.ndarray:
    """Resize to (out_h, out_w); stretch, or preserve aspect ratio with centered padding."""
    if not letterbox:
        return cv2.resize(img, (out_w, out_h))

    in_h, in_w = img.shape[:2]
    scale = min(out_w / in_w, out_h / in_h)
    new_w = max(1, int(round(in_w * scale)))
    new_h = max(1, int(round(in_h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2

    if grayscale:
        canvas = np.zeros((out_h, out_w), dtype=img.dtype)
    else:
        canvas = np.zeros((out_h, out_w, 3), dtype=img.dtype)

    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas


def load_labeled_images(
    img_size: tuple[int, int] = IMG_SIZE_CLASSICAL,
    grayscale: bool = False,
    max_samples: int | None = None,
    return_ids: bool = False,
    letterbox: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
):
    """Load labeled cat/dog images from the Part 1 training directory.

    Parameters
    ----------
    img_size : tuple
        Target (height, width) for resizing.
    letterbox : bool
        If True, preserve aspect ratio and pad to ``img_size``; else stretch to ``img_size``.
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
            img = _resize_image_to_shape(
                img,
                img_size[0],
                img_size[1],
                grayscale,
                letterbox,
            )
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


def _collect_labeled_paths(max_samples: int | None) -> tuple[list[Path], list[int], list[str]]:
    paths_flat: list[Path] = []
    labels_flat: list[int] = []
    ids_flat: list[str] = []
    for class_name, label in LABEL_MAP.items():
        class_dir = PART1_TRAIN_DIR / f"{class_name}s"
        path_list = sorted(class_dir.glob("*.jpg"))
        if max_samples is not None:
            path_list = path_list[:max_samples]
        for p in path_list:
            paths_flat.append(p)
            labels_flat.append(label)
            ids_flat.append(p.stem)
    return paths_flat, labels_flat, ids_flat


def _memmap_meta_matches(
    meta: dict,
    img_size: tuple[int, int],
    grayscale: bool,
    max_samples: int | None,
    letterbox: bool,
) -> bool:
    if tuple(meta["img_size"]) != tuple(img_size):
        return False
    if bool(meta["grayscale"]) != grayscale:
        return False
    if bool(meta.get("letterbox", False)) != letterbox:
        return False
    meta_ms = meta.get("max_samples")
    if meta_ms != max_samples:
        return False
    return True


def load_labeled_images_memmap(
    img_size: tuple[int, int] = IMG_SIZE_CLASSICAL,
    grayscale: bool = False,
    max_samples: int | None = None,
    return_ids: bool = False,
    cache_dir: Path | None = None,
    rebuild: bool = False,
    letterbox: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
):
    """Labeled images as a float32 on-disk memmap (same API as ``load_labeled_images``)."""
    if cache_dir is None:
        cache_dir = OUTPUTS_DIR / "cache" / "part1_labeled_memmap"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_path = cache_dir / "X_memmap.dat"
    meta_path = cache_dir / "meta.json"
    y_path = cache_dir / "y.npy"
    ids_path = cache_dir / "ids.npy"

    h, w = img_size[0], img_size[1]

    use_cache = (
        not rebuild
        and data_path.is_file()
        and meta_path.is_file()
        and y_path.is_file()
    )
    meta: dict | None = None
    if use_cache:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if not _memmap_meta_matches(meta, img_size, grayscale, max_samples, letterbox):
            use_cache = False
        elif return_ids and not ids_path.is_file():
            use_cache = False

    if use_cache and meta is not None:
        shape = tuple(meta["shape"])
        X = np.memmap(data_path, dtype=np.float32, mode="r", shape=shape)
        y = np.load(y_path)
        if return_ids:
            train_ids = np.load(ids_path, allow_pickle=True)
            return X, y, train_ids
        return X, y

    paths_flat, labels_flat, ids_flat = _collect_labeled_paths(max_samples)
    n = len(paths_flat)
    if grayscale:
        shape = (n, h, w)
    else:
        shape = (n, h, w, 3)

    mm = np.memmap(data_path, dtype=np.float32, mode="w+", shape=shape)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    desc = "Building labeled memmap cache"
    for i, p in enumerate(tqdm(paths_flat, desc=desc)):
        img = cv2.imread(str(p), flag)
        if img is None:
            raise RuntimeError(f"Could not read image: {p}")
        img = _resize_image_to_shape(img, h, w, grayscale, letterbox)
        if not grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        row = img.astype(np.float32) / 255.0
        mm[i, ...] = row

    mm.flush()
    del mm

    meta_out = {
        "shape": list(shape),
        "dtype": "float32",
        "img_size": list(img_size),
        "grayscale": grayscale,
        "max_samples": max_samples,
        "letterbox": letterbox,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    y_arr = np.array(labels_flat, dtype=np.int64)
    np.save(y_path, y_arr)
    ids_arr = np.array(ids_flat, dtype=object)
    np.save(ids_path, ids_arr, allow_pickle=True)

    X = np.memmap(data_path, dtype=np.float32, mode="r", shape=shape)
    if return_ids:
        return X, y_arr, ids_arr
    return X, y_arr


def load_test_images(
    img_size: tuple[int, int] = IMG_SIZE_CLASSICAL,
    grayscale: bool = False,
    letterbox: bool = False,
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
        img = _resize_image_to_shape(
            img,
            img_size[0],
            img_size[1],
            grayscale,
            letterbox,
        )
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


def _combined_hog_hsv_for_chunk(
    images: np.ndarray,
    hog_orientations: int,
    hog_cells_per_block: tuple[int, int],
    hog_scales_ppc: list,
    hsv_bins: int,
    show_progress: bool,
    progress_desc: str,
    include_lbp: bool = False,
    lbp_n_points: int = 24,
    lbp_radius: int = 3,
) -> np.ndarray:
    """One row per image: multi-scale HOG concatenated with HSV histogram."""
    n = images.shape[0]
    rows: list[np.ndarray] = []
    if show_progress:
        idx_iter = tqdm(range(n), desc=progress_desc)
    else:
        idx_iter = range(n)

    for i in idx_iter:
        img = images[i]
        gray = _to_gray_uint8(img)
        hog_parts: list[np.ndarray] = []
        for ppc in hog_scales_ppc:
            fd = hog(
                gray,
                orientations=hog_orientations,
                pixels_per_cell=ppc,
                cells_per_block=hog_cells_per_block,
                feature_vector=True,
            )
            hog_parts.append(fd)
        hog_row = np.concatenate(hog_parts)

        rgb_u8 = (img * 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        hist_h, _ = np.histogram(
            hsv[..., 0], bins=hsv_bins, range=(0, 180), density=True
        )
        hist_s, _ = np.histogram(
            hsv[..., 1], bins=hsv_bins, range=(0, 256), density=True
        )
        hist_v, _ = np.histogram(
            hsv[..., 2], bins=hsv_bins, range=(0, 256), density=True
        )
        hsv_row = np.concatenate([hist_h, hist_s, hist_v])

        parts: list[np.ndarray] = [hog_row, hsv_row]
        if include_lbp:
            lbp = local_binary_pattern(
                gray, lbp_n_points, lbp_radius, method="uniform"
            )
            n_bins_lbp = lbp_n_points + 2
            hist_lbp, _ = np.histogram(
                lbp.ravel(), bins=n_bins_lbp, range=(0, n_bins_lbp), density=True
            )
            parts.append(hist_lbp)

        rows.append(np.concatenate(parts))

    return np.array(rows, dtype=np.float32)


def extract_multiscale_hog_hsv_features(
    images: np.ndarray,
    hog_orientations: int = 9,
    hog_cells_per_block: tuple[int, int] = (2, 2),
    hog_scales_ppc: list | None = None,
    hsv_bins: int = 12,
    n_jobs: int = 1,
    include_lbp: bool = False,
    lbp_n_points: int = 24,
    lbp_radius: int = 3,
) -> np.ndarray:
    """Multi-scale HOG + HSV histogram per image (skimage HOG is CPU-only).

    Use ``n_jobs=-1`` to run chunks in parallel processes (uses all cores).
    """
    if hog_scales_ppc is None:
        hog_scales_ppc = [(8, 8), (16, 16)]

    n = images.shape[0]
    if n_jobs == 1:
        return _combined_hog_hsv_for_chunk(
            images,
            hog_orientations,
            hog_cells_per_block,
            hog_scales_ppc,
            hsv_bins,
            show_progress=True,
            progress_desc="Extracting HOG+HSV (single process)",
            include_lbp=include_lbp,
            lbp_n_points=lbp_n_points,
            lbp_radius=lbp_radius,
        )

    n_jobs_eff = effective_n_jobs(n_jobs)
    splits = np.array_split(images, n_jobs_eff, axis=0)
    chunks_in = []
    for split in splits:
        if split.shape[0] > 0:
            chunks_in.append(split)

    tasks = []
    for chunk in chunks_in:
        tasks.append(
            delayed(_combined_hog_hsv_for_chunk)(
                chunk,
                hog_orientations,
                hog_cells_per_block,
                hog_scales_ppc,
                hsv_bins,
                False,
                "",
                include_lbp,
                lbp_n_points,
                lbp_radius,
            )
        )

    parts = Parallel(n_jobs=n_jobs_eff, backend="loky")(tasks)

    return np.vstack(parts)


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
    df.to_csv(output_path, index=False, lineterminator="\n")
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
    normalize_train: bool = True,
):
    """Wrap numpy arrays into PyTorch DataLoaders with ImageNet normalization.

    Images are expected as float32 [0, 1] in (N, H, W, 3) format.
    When *normalize_train* is False the training tensors are left in [0, 1]
    so that augmentation transforms (ColorJitter, etc.) can be applied first
    inside the training loop, followed by manual normalization.
    Validation data is always normalized.
    Returns (train_loader, val_loader).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _to_tensor(X: np.ndarray, do_normalize: bool) -> torch.Tensor:
        t = torch.from_numpy(X).permute(0, 3, 1, 2).float().clone()
        if (t.shape[2], t.shape[3]) != img_size:
            t = torch.nn.functional.interpolate(
                t, size=img_size, mode="bilinear", align_corners=False
            )
        if do_normalize:
            t.sub_(_mean).div_(_std)
        return t

    train_t = _to_tensor(X_train, do_normalize=normalize_train)
    val_t = _to_tensor(X_val, do_normalize=True)

    train_ds = TensorDataset(train_t, torch.from_numpy(y_train).long())
    val_ds = TensorDataset(val_t, torch.from_numpy(y_val).long())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_gpu_augmentation(img_size: tuple[int, int] = (224, 224)):
    """Return a transform pipeline that runs on GPU-resident batches (NCHW).

    Expects images in [0, 1] range. The pipeline applies augmentation first,
    then ImageNet normalization as the final step.
    Call this from the training notebook and apply the returned callable
    to each image *after* moving the batch to the device.
    """
    from torchvision import transforms as T

    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomErasing(p=0.1),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

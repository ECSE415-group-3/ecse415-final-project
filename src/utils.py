"""
Small project utilities shared across notebooks and modules.

Provides data loading, train/test splitting, feature extraction
(HOG, LBP), PCA, PyTorch dataloader helpers, and Kaggle submission
generation for the Dogs vs. Cats classification pipeline (Part 1).
"""

from __future__ import annotations

import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import pandas as pd
from joblib import effective_n_jobs
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
    path_order: list[tuple[Path, int]] = []
    for class_name, label in LABEL_MAP.items():
        class_dir = PART1_TRAIN_DIR / f"{class_name}s"
        paths = sorted(class_dir.glob("*.jpg"))
        if max_samples is not None:
            paths = paths[:max_samples]
        for p in paths:
            path_order.append((p, label))

    h, w = img_size[0], img_size[1]
    n_paths = len(path_order)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR

    if grayscale:
        X = np.empty((n_paths, h, w), dtype=np.float32)
    else:
        X = np.empty((n_paths, h, w, 3), dtype=np.float32)

    labels = np.empty(n_paths, dtype=np.int64)
    ids_out: list[str] = []
    write_i = 0

    for p, label in tqdm(path_order, desc="Loading labeled train images"):
        img = cv2.imread(str(p), flag)
        if img is None:
            continue
        img = _resize_image_to_shape(img, h, w, grayscale, letterbox)
        if not grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X[write_i, ...] = img.astype(np.float32) / 255.0
        labels[write_i] = label
        if return_ids:
            ids_out.append(p.stem)
        write_i += 1

    if write_i < n_paths:
        X = X[:write_i].copy()
    y = labels[:write_i].copy()
    if return_ids:
        return X, y, np.array(ids_out, dtype=object)
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
    h, w = img_size[0], img_size[1]
    if cache_dir is None:
        ms_key = max_samples if max_samples is not None else "all"
        key = f"{h}x{w}_lb{int(letterbox)}_g{int(grayscale)}_ms{ms_key}"
        cache_dir = OUTPUTS_DIR / "cache" / "part1_labeled_memmap" / key
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_path = cache_dir / "X_memmap.dat"
    meta_path = cache_dir / "meta.json"
    y_path = cache_dir / "y.npy"
    ids_path = cache_dir / "ids.npy"

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
    h, w = img_size[0], img_size[1]
    n_paths = len(paths)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR

    if grayscale:
        X = np.empty((n_paths, h, w), dtype=np.float32)
    else:
        X = np.empty((n_paths, h, w, 3), dtype=np.float32)

    ids_out: list[int] = []
    write_i = 0

    for p in tqdm(paths, desc="Loading test images"):
        img = cv2.imread(str(p), flag)
        if img is None:
            continue
        img = _resize_image_to_shape(img, h, w, grayscale, letterbox)
        if not grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X[write_i, ...] = img.astype(np.float32) / 255.0
        ids_out.append(int(p.stem))
        write_i += 1

    if write_i < n_paths:
        X = X[:write_i].copy()
    return X, ids_out


def _test_memmap_meta_matches(
    meta: dict,
    img_size: tuple[int, int],
    grayscale: bool,
    letterbox: bool,
) -> bool:
    if tuple(meta["img_size"]) != tuple(img_size):
        return False
    if bool(meta["grayscale"]) != grayscale:
        return False
    if bool(meta.get("letterbox", False)) != letterbox:
        return False
    return True


def load_test_images_memmap(
    img_size: tuple[int, int] = IMG_SIZE_CLASSICAL,
    grayscale: bool = False,
    letterbox: bool = False,
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """Kaggle test images as float32 on-disk memmap (same layout as ``load_test_images``)."""
    h, w = img_size[0], img_size[1]
    if cache_dir is None:
        key = f"{h}x{w}_lb{int(letterbox)}_g{int(grayscale)}"
        cache_dir = OUTPUTS_DIR / "cache" / "part1_test_memmap" / key
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_path = cache_dir / "X_memmap.dat"
    meta_path = cache_dir / "meta.json"
    ids_path = cache_dir / "ids.npy"

    paths = sorted(PART1_TEST_DIR.glob("*.jpg"), key=lambda p: int(p.stem))
    n = len(paths)

    use_cache = not rebuild and data_path.is_file() and meta_path.is_file() and ids_path.is_file()
    meta: dict | None = None
    if use_cache:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if not _test_memmap_meta_matches(meta, img_size, grayscale, letterbox):
            use_cache = False

    if use_cache and meta is not None:
        shape = tuple(meta["shape"])
        X = np.memmap(data_path, dtype=np.float32, mode="r", shape=shape)
        ids_arr = np.load(ids_path)
        return X, [int(x) for x in ids_arr.tolist()]

    if grayscale:
        shape = (n, h, w)
    else:
        shape = (n, h, w, 3)

    mm = np.memmap(data_path, dtype=np.float32, mode="w+", shape=shape)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    ids_row: list[int] = []

    for i, p in enumerate(tqdm(paths, desc="Building test memmap cache")):
        img = cv2.imread(str(p), flag)
        if img is None:
            raise RuntimeError(f"Could not read image: {p}")
        img = _resize_image_to_shape(img, h, w, grayscale, letterbox)
        if not grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mm[i, ...] = img.astype(np.float32) / 255.0
        ids_row.append(int(p.stem))

    mm.flush()
    del mm

    meta_out = {
        "shape": list(shape),
        "dtype": "float32",
        "img_size": list(img_size),
        "grayscale": grayscale,
        "letterbox": letterbox,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    ids_arr = np.array(ids_row, dtype=np.int64)
    np.save(ids_path, ids_arr)

    X = np.memmap(data_path, dtype=np.float32, mode="r", shape=shape)
    return X, [int(x) for x in ids_arr.tolist()]


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


def _feat_extract_max_procs() -> int:
    """Upper bound on feature-extraction processes (spawn + large array IPC)."""
    raw = os.environ.get("FEAT_EXTRACT_MAX_PROCS", "8")
    try:
        v = int(raw)
    except ValueError:
        v = 8

    return max(1, v)


def _pool_worker_limit_blas_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def _spawn_pool_map(
    fn: Callable[[tuple], np.ndarray],
    arg_tuples: list[tuple],
    max_workers: int,
) -> list[np.ndarray]:
    """Run ``fn`` on each tuple via ``spawn`` processes (avoids fork + native libs)."""
    n = len(arg_tuples)
    if n == 0:
        return []

    cap = _feat_extract_max_procs()
    workers = min(max(1, max_workers), cap, n)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_pool_worker_limit_blas_threads,
    ) as ex:
        return list(ex.map(fn, arg_tuples))


def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a float32 [0,1] image to uint8 grayscale."""
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3:
        r = img[..., 0]
        g = img[..., 1]
        b = img[..., 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        gray = img

    return (gray * 255).astype(np.uint8)


def _hog_rows_for_chunk(
    images: np.ndarray,
    pixels_per_cell: tuple[int, int],
    cells_per_block: tuple[int, int],
    orientations: int,
    show_progress: bool,
    progress_desc: str,
) -> np.ndarray:
    """HOG rows for a slice of images (skimage; used by parallel workers)."""
    images = np.ascontiguousarray(np.asarray(images, dtype=np.float32))
    n = images.shape[0]
    rows: list[np.ndarray] = []
    if show_progress:
        idx_iter = tqdm(range(n), desc=progress_desc)
    else:
        idx_iter = range(n)

    for i in idx_iter:
        img = images[i]
        gray = _to_gray_uint8(img)
        fd = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            feature_vector=True,
        )
        rows.append(fd)

    return np.array(rows, dtype=np.float32)


def _hog_rows_star(args: tuple) -> np.ndarray:
    return _hog_rows_for_chunk(*args)


def extract_hog_features(
    images: np.ndarray,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
    orientations: int = 9,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute HOG feature vectors for a batch of images.

    Use ``n_jobs=-1`` for parallel ``spawn`` processes (safer than fork with skimage/OpenCV).
    Parallelism is capped by env ``FEAT_EXTRACT_MAX_PROCS`` (default 8) to limit RAM and worker count.

    Returns
    -------
    features : np.ndarray, shape (n_samples, n_hog_features)
    """
    if n_jobs == 1:
        return _hog_rows_for_chunk(
            images,
            pixels_per_cell,
            cells_per_block,
            orientations,
            True,
            "Extracting HOG",
        )

    n_jobs_eff = effective_n_jobs(n_jobs)
    n_splits = min(n_jobs_eff, _feat_extract_max_procs())
    splits = np.array_split(images, n_splits, axis=0)
    chunks_in = []
    for split in splits:
        if split.shape[0] > 0:
            chunks_in.append(split)

    arg_tuples: list[tuple] = []
    for chunk in chunks_in:
        arg_tuples.append(
            (
                chunk,
                pixels_per_cell,
                cells_per_block,
                orientations,
                False,
                "",
            )
        )

    parts = _spawn_pool_map(_hog_rows_star, arg_tuples, n_splits)

    return np.vstack(parts)


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
    images = np.ascontiguousarray(np.asarray(images, dtype=np.float32))
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


def _combined_hog_hsv_star(args: tuple) -> np.ndarray:
    return _combined_hog_hsv_for_chunk(*args)


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

    Use ``n_jobs=-1`` to run chunks in parallel ``spawn`` processes (safer than fork with native libs).
    Parallelism is capped by env ``FEAT_EXTRACT_MAX_PROCS`` (default 8) to limit RAM and worker count.
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
    n_splits = min(n_jobs_eff, _feat_extract_max_procs())
    splits = np.array_split(images, n_splits, axis=0)
    chunks_in = []
    for split in splits:
        if split.shape[0] > 0:
            chunks_in.append(split)

    arg_tuples: list[tuple] = []
    for chunk in chunks_in:
        arg_tuples.append(
            (
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

    parts = _spawn_pool_map(_combined_hog_hsv_star, arg_tuples, n_splits)

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


def resolve_svc(
    use_cuda: bool,
    label_prefix: str = "SVC",
    prefer_gpu: bool = True,
):
    """Pick sklearn SVC (CPU) or cuML SVC (GPU) when CUDA is active and RAPIDS is installed.

    Set ``prefer_gpu=False`` to force sklearn SVC even if CUDA is available.
    """
    from sklearn.svm import SVC as SklearnSVC

    if not prefer_gpu:
        print("{}: sklearn (CPU) [prefer_gpu=False]".format(label_prefix))
        return SklearnSVC, SklearnSVC, "sklearn (CPU)"

    if not use_cuda:
        print("{}: sklearn (CPU)".format(label_prefix))
        return SklearnSVC, SklearnSVC, "sklearn (CPU)"
    try:
        from cuml.svm import SVC as CuSVC

        print("{}: cuml (GPU)".format(label_prefix))
        return CuSVC, SklearnSVC, "cuml (GPU)"
    except Exception as exc:
        print("cuml SVC unavailable ({}); using sklearn".format(exc))
        print("{}: sklearn (CPU)".format(label_prefix))
        return SklearnSVC, SklearnSVC, "sklearn (CPU)"


def clone_svc(est, sklearn_svc_class):
    """sklearn.clone for CPU SVC; fresh hyperparams for cuML (avoids clone quirks)."""
    from sklearn.base import clone

    if est.__class__ is sklearn_svc_class:
        return clone(est)
    return type(est)(**est.get_params(deep=False))

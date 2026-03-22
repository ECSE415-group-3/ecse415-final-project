"""Centralized project path configuration.

Required dataset layout:
- Part 1: data/part1/ecse-415-winter-2026-dog-vs-cat-classification
- Part 2: data/part2/Stanford Dog Dataset/Images and
		  data/part2/Stanford Dog Dataset/Annotation
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

PART1_DATASET_FOLDER = "ecse-415-winter-2026-dog-vs-cat-classification"
PART2_DATASET_FOLDER = "Stanford Dog Dataset"

# Canonical dataset directories used throughout the notebooks/scripts.
PART1_KAGGLE_DIR = DATA_ROOT / "part1" / PART1_DATASET_FOLDER
PART2_DATASET_DIR = DATA_ROOT / "part2" / PART2_DATASET_FOLDER
PART2_IMAGES_DIR = PART2_DATASET_DIR / "Images"
PART2_ANNOTATIONS_DIR = PART2_DATASET_DIR / "Annotation"


def get_data_paths() -> dict[str, Path]:
	"""Return all canonical dataset paths in a single mapping."""
	return {
		"DATA_ROOT": DATA_ROOT,
		"PART1_DATASET_FOLDER": PART1_DATASET_FOLDER,
		"PART2_DATASET_FOLDER": PART2_DATASET_FOLDER,
		"PART1_KAGGLE_DIR": PART1_KAGGLE_DIR,
		"PART2_DATASET_DIR": PART2_DATASET_DIR,
		"PART2_IMAGES_DIR": PART2_IMAGES_DIR,
		"PART2_ANNOTATIONS_DIR": PART2_ANNOTATIONS_DIR,
	}


def validate_data_layout() -> None:
	"""Validate required local dataset folder layout and raise clear guidance."""
	missing: list[str] = []

	if not PART1_KAGGLE_DIR.exists():
		missing.append(
			"Part 1 dataset folder is missing. Expected: "
			f"{PART1_KAGGLE_DIR} "
			f"(exact folder name: {PART1_DATASET_FOLDER})."
		)

	if not PART2_IMAGES_DIR.exists():
		missing.append(
			"Part 2 images folder is missing. Expected: "
			f"{PART2_IMAGES_DIR} "
			f"(inside exact folder name: {PART2_DATASET_FOLDER})."
		)

	if not PART2_ANNOTATIONS_DIR.exists():
		missing.append(
			"Part 2 annotations folder is missing. "
			f"Expected: {PART2_ANNOTATIONS_DIR} "
			f"(inside exact folder name: {PART2_DATASET_FOLDER})."
		)

	if missing:
		raise FileNotFoundError(
			"Dataset layout validation failed.\n"
			"Put datasets in the repository under data/part1 and data/part2.\n"
			+ "\n".join(missing)
		)

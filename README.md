# ECSE 415 Final Project
## Student names
**Student 1**: Clara Ghenassia, 261096490  
**Student 2**: Victor Odoux, 261099183   
**Student 3**: Thomas Cottereau, 261083741  
**Student 4**: Bora Denizasan,  
**Student 5**: Le-Tao Li 261041500,  

## Quick Explanation
This project implements and compares multiple Computer Vision approaches for classifying images in the Dogs vs. Cats dataset (Part 1). Using the best-performing classifier from Part 1, we then build a detection/localization pipeline that predicts dog bounding boxes on the Stanford Dogs dataset (Part 2), evaluated using Intersection over Union (IoU).

Kaggle competition (Part 1):
https://www.kaggle.com/t/95bd3900e6ea43cbbb48e5699b8b3820

Stanford Dogs dataset download (Part 2):
https://mcgill-my.sharepoint.com/:f:/g/personal/benjamin_beggs_mcgill_ca/IgCI5Z5WAg3lSJC13KKnyQncAWATdWR2qGs7ZW8ste28FlY?e=nzlWUn

Specifically:
- Part 1 (classification benchmark, 50 pts total): implement 3 distinct classifier techniques to distinguish dogs vs. cats, then compare them using confusion matrices and performance metrics on an internal test split.
- Part 2 (detection + localization, 50 pts total): use the best classifier to localize dogs (bounding boxes) in complex scenes, and evaluate predictions vs. ground-truth annotations with IoU plus qualitative “successes & failures” analysis.

## Setup
Clone the repo:
```bash
git clone <YOUR_GIT_URL> ecse415-final-project
cd ecse415-final-project
```

Create the required dataset folders in the repository:
```bash
mkdir -p data/part1/ecse-415-winter-2026-dog-vs-cat-classification
mkdir -p data/part2/Stanford\ Dog\ Dataset/Annotation
mkdir -p data/part2/Stanford\ Dog\ Dataset/Images
```

Download + unzip the Kaggle Dogs vs. Cats dataset (Part 1) into `data/part1/ecse-415-winter-2026-dog-vs-cat-classification/`:
```bash
kaggle competitions download -c ecse-415-winter-2026-dog-vs-cat-classification -p data/part1/ecse-415-winter-2026-dog-vs-cat-classification --force
unzip -o data/part1/ecse-415-winter-2026-dog-vs-cat-classification/ecse-415-winter-2026-dog-vs-cat-classification.zip -d data/part1/ecse-415-winter-2026-dog-vs-cat-classification
```

Download the Stanford Dogs dataset (Part 2) from:
https://mcgill-my.sharepoint.com/:f:/g/personal/benjamin_beggs_mcgill_ca/IgCI5Z5WAg3lSJC13KKnyQncAWATdWR2qGs7ZW8ste28FlY?e=nzlWUn

Save/extract it into `data/part2/Stanford Dog Dataset/` (so it provides the `Images/` and `Annotation/` folders).

Required folder policy used by the code:
- Part 1 must exist exactly at: `data/part1/ecse-415-winter-2026-dog-vs-cat-classification`
- Part 2 must exist exactly at: `data/part2/Stanford Dog Dataset`
- Part 2 must provide both folders: `data/part2/Stanford Dog Dataset/Images` and `data/part2/Stanford Dog Dataset/Annotation`

Use these centralized paths in notebooks/scripts (avoid hardcoded relative data paths):
```python
from src.config import (
	PART1_KAGGLE_DIR,
	PART2_IMAGES_DIR,
	PART2_ANNOTATIONS_DIR,
	validate_data_layout,
)

validate_data_layout()
```

Create a local virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repo Structure
This repository is organized as:
- `src/`: shared utilities imported by the notebooks
  - `config.py`: centralized dataset paths, image-size constants, label mappings, and output directory paths
  - `utils.py`: data loading, train/test splitting, feature extraction (HOG, LBP), PCA, PyTorch dataloader helpers, and Kaggle submission CSV generation
  - `evaluation.py`: metric computation (accuracy, precision, recall, F1), confusion matrix, and multi-model comparison table
  - `visualization.py`: plotting helpers — confusion-matrix heatmaps, sample-prediction grids, model-comparison bar charts, and training-history curves
- `notebooks/`: end-to-end experiment runner / analysis notebooks (each notebook corresponds to one step of the project pipeline)
- `data/part1/ecse-415-winter-2026-dog-vs-cat-classification/`: Dogs vs. Cats classification dataset (Kaggle `train/` + `test/` + `sample_submission.csv`, local only)
- `data/part2/Stanford Dog Dataset/Annotation/` and `data/part2/Stanford Dog Dataset/Images/`: Stanford Dogs images + bounding-box annotation folders (local only)
- `outputs/figures/`, `outputs/models/`, `outputs/localization/`: generated artifacts
- `docs/`: project instruction PDFs

Notebooks currently included:
- `01-feature-based-model-optionA.ipynb`: classifier for Option A (feature-based)
- `02-appearace-based-model-optionB.ipynb`: classifier for Option B (appearance-based / PCA + classifier)
- `03-deep-learning-model-optionC.ipynb`: classifier for Option C (deep learning / CNN)
- `04-analysis-and-evaluation.ipynb`: confusion matrices + quantitative evaluation + method comparison
- `05-localization.ipynb`: dog detection/localization pipeline on Stanford Dogs + qualitative evaluation

## `src/` Module API

### `config.py` — Constants
| Constant | Value / Description |
|---|---|
| `PART1_TRAIN_DIR` | `data/part1/…/train/train` (cats/ and dogs/ subdirs) |
| `PART1_TEST_DIR` | `data/part1/…/test/test` (unlabeled Kaggle test images) |
| `PART1_SAMPLE_SUBMISSION` | Path to `sample_submission.csv` |
| `FIGURES_DIR` / `MODELS_DIR` / `LOCALIZATION_DIR` | Output subdirectories under `outputs/` |
| `IMG_SIZE_CLASSICAL` | `(128, 128)` — for HOG/LBP/PCA methods |
| `IMG_SIZE_CNN` | `(224, 224)` — for pre-trained CNN fine-tuning |
| `CLASS_NAMES` | `["cat", "dog"]` |
| `LABEL_MAP` / `LABEL_MAP_INV` | `{"cat": 0, "dog": 1}` and reverse |

### `utils.py` — Data & Features
| Function | Purpose |
|---|---|
| `load_labeled_images(img_size, grayscale, max_samples)` | Load cat/dog training images, resize, normalize to [0,1] |
| `load_test_images(img_size, grayscale)` | Load unlabeled Kaggle test images with numeric ids |
| `split_data(X, y, test_size, random_state)` | Stratified train/internal-test split |
| `extract_hog_features(images, …)` | HOG descriptors for Option A |
| `extract_lbp_features(images, …)` | LBP histogram features for Option A |
| `apply_pca(X_train, X_test, n_components)` | Fit PCA on train, transform both — for Option B |
| `generate_submission_csv(ids, predictions, output_path)` | Write Kaggle-format CSV |
| `get_pytorch_dataloaders(X_train, y_train, X_val, y_val, …)` | ImageNet-normalized DataLoaders for Option C |

### `evaluation.py` — Metrics
| Function | Purpose |
|---|---|
| `compute_metrics(y_true, y_pred)` | Returns dict with accuracy, precision, recall, F1 |
| `compute_confusion_matrix(y_true, y_pred)` | 2×2 confusion matrix array |
| `compare_models(results)` | DataFrame comparing metrics across all classifiers |

### `visualization.py` — Plots
| Function | Purpose |
|---|---|
| `plot_confusion_matrix(cm, …, save_path)` | Annotated seaborn heatmap |
| `plot_sample_predictions(images, y_true, y_pred, …, save_path)` | Grid with green/red labels |
| `plot_model_comparison(results, save_path)` | Grouped bar chart of metrics |
| `plot_training_history(history, save_path)` | Train/val loss & accuracy curves |

## Workflow Explanation
The workflow is designed so the notebooks orchestrate the pipeline, while the core logic lives in `src/`:
- Do not implement core functions inside the notebooks.
- Implement reusable logic in `src/` modules and import/call it from the notebooks.
- Save all generated artifacts (figures, models, submission files) into `outputs/` under an appropriate subfolder.

Pipeline summary:
1. Download Kaggle Dogs vs. Cats dataset, split labeled data into `train` + `internal_test` (do not use the Kaggle public test set for evaluation because it has no labels).
2. Train and evaluate 3 classifiers (Option A, Option B, Option C), compute confusion matrices on the internal test split, and compare trade-offs.
3. Pick the single best model and run it on Kaggle’s unlabeled public test set to generate the CSV submission.
4. Use the best classifier to localize dogs on the Stanford Dogs dataset: run your localization strategy (e.g., sliding window + thresholding, optional post-processing like Non-Maximum Suppression), evaluate predicted bounding boxes using IoU vs. ground truth, and show successes/failures with a discussion of why failures happen (lighting, occlusion, background clutter, etc.)

## Final Submission (What to submit)
From the project instructions:
- A report in PDF format (up to 10 pages excluding references)
- All code required to reproduce results in a single Jupyter notebook
- The codebase should run without errors, and code should be appropriately commented

Presentation deliverable (separately):
- Submit the slides as a PDF on the course presentation deadline.

## Outputs
Expected outputs written by the pipeline:
- `outputs/figures/`: plots/visualizations (confusion matrices, sample predictions, IoU examples, bbox overlays)
- `outputs/models/`: saved trained models / checkpoints (as applicable)
- `outputs/localization/`: localization artifacts (bbox overlays, IoU evaluation) and any prediction CSVs generated for submission (if applicable)

## Docs
- `docs/final-project-instructions.pdf`: full project specification (classification benchmark + detection/localization requirements + scoring + submission rules)
- `docs/final-presentation-detail.pdf`: presentation guidelines and required slide/PDF format



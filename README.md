# ECSE 415 Final Project
## Student names
**Student 1**: Clara Ghenassia,   
**Student 2**: Victor Odoux, 261099183   
**Student 3**: Thomas Cottereau, 261083741
**Student 4**: Bora Denizasan,  
**Student 5**: Le-Tao Li,  

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

Create the folders used by the pipeline:
```bash
mkdir -p data/part1/data/kaggle
mkdir -p data/part2/data/Annotation
mkdir -p data/part2/data/Images
```

Download + unzip the Kaggle Dogs vs. Cats dataset (Part 1) into `data/part1/data/kaggle/`:
```bash
kaggle competitions download -c ecse-415-winter-2026-dog-vs-cat-classification -p data/part1/data/kaggle --force
unzip -o data/part1/data/kaggle/ecse-415-winter-2026-dog-vs-cat-classification.zip -d data/part1/data/kaggle
```

Download the Stanford Dogs dataset (Part 2) from:
https://mcgill-my.sharepoint.com/:f:/g/personal/benjamin_beggs_mcgill_ca/IgCI5Z5WAg3lSJC13KKnyQncAWATdWR2qGs7ZW8ste28FlY?e=nzlWUn

Save/extract it into `data/part2/data/` (so it provides the `Images/` and `Annotation/` folders).

Create a local virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repo Structure
This repository is organized as:
- `src/`: shared utilities imported by the notebooks
- `notebooks/`: end-to-end experiment runner / analysis notebooks (each notebook corresponds to one step of the project pipeline)
- `data/part1/data/kaggle/`: Dogs vs. Cats classification dataset (Kaggle `train/` + `test/` + `sample_submission.csv`)
- `data/part2/data/Annotation/` and `data/part2/data/Images/`: Stanford Dogs images + bounding-box annotation folders (localization/IoU evaluation)
- `outputs/figures/`, `outputs/models/`, `outputs/localization/`: generated artifacts
- `docs/`: project instruction PDFs

Notebooks currently included:
- `01-feature-based-model-optionA.ipynb`: classifier for Option A (feature-based)
- `02-appearace-based-model-optionB.ipynb`: classifier for Option B (appearance-based / PCA + classifier)
- `03-deep-learning-model-optionC.ipynb`: classifier for Option C (deep learning / CNN)
- `04-analysis-and-evaluation.ipynb`: confusion matrices + quantitative evaluation + method comparison
- `05-localization.ipynb`: dog detection/localization pipeline on Stanford Dogs + qualitative evaluation

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



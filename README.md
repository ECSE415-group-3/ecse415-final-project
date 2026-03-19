# ECSE 415 Final Project
## Student names
**Student 1**: Clara Ghenassia,   
**Student 2**: Victor Odoux,   
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
mkdir -p data/kaggle
mkdir -p data/stanford_dogs
```

Download + unzip the Kaggle Dogs vs. Cats dataset (Part 1) into `data/kaggle/`:
```bash
kaggle competitions download -c ecse-415-winter-2026-dog-vs-cat-classification -p data/kaggle --force
unzip -o data/kaggle/ecse-415-winter-2026-dog-vs-cat-classification.zip -d data/kaggle
```

Download the Stanford Dogs dataset (Part 2) from:
https://mcgill-my.sharepoint.com/:f:/g/personal/benjamin_beggs_mcgill_ca/IgCI5Z5WAg3lSJC13KKnyQncAWATdWR2qGs7ZW8ste28FlY?e=nzlWUn

Save/extract it into `data/stanford_dogs/` (this is where the Part 2 localization pipeline expects the data).

Create a local virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repo Structure
This repository is organized as:
- `src/`: reusable pipeline modules (dataset loading, preprocessing, classifiers, evaluation, localization orchestration, Kaggle submission generation, visualization helpers)
- `notebooks/`: end-to-end experiment runner / analysis notebooks (each notebook corresponds to one step of the project pipeline)
- `data/`: input data downloads (gitignored contents); `data/kaggle/` contains the Kaggle competition files (e.g., downloaded `*.zip`, including `train.zip`/`test.zip`, plus extracted `train/` and `test/` images as needed), and `data/stanford_dogs/` contains the Stanford Dogs images + annotation files used for localization/IoU evaluation
- `outputs/`: generated artifacts (figures, trained models, and Kaggle submission CSV outputs)
- `docs/`: project instruction PDFs

Notebooks currently included (by responsibility):
- `01-data-setup-and-processing.ipynb`: dataset download/extraction + preprocessing setup
- `02-feature-based-model-optionA copy.ipynb`: classifier for Option A (feature-based: handcrafted features + classical classifier)
- `03-appearace-based-model-optionB.ipynb`: classifier for Option B (appearance-based / PCA + classifier)
- `04-deep-learning-model-optionC copy.ipynb`: classifier for Option C (CNN fine-tuning / deep learning)
- `05-analysis-and-evaluation copy.ipynb`: confusion matrices + quantitative evaluation + method comparison
- `06-kaggle-submission-generation.ipynb`: run best classifier on the unlabeled Kaggle public test set and create the leaderboard CSV
- `07-localization-pipeline.ipynb`: dog detection/localization pipeline using the best classifier
- `08-localization-analysis-and-evaluation copy.ipynb`: localization quantitative evaluation (IoU) + qualitative analysis
- `09-failure-success-analysis.ipynb`: successes/failures visualization + discussion

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
- `outputs/kaggle-submission/`: generated Kaggle prediction CSV(s)

## Docs
- `docs/final-project-instructions.pdf`: full project specification (classification benchmark + detection/localization requirements + scoring + submission rules)
- `docs/final-presentation-detail.pdf`: presentation guidelines and required slide/PDF format



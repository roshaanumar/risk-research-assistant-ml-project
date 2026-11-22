# Risk Research Assistant — Machine Learning Course Project

A supervised **multiclass classification** study using the National Institute of Justice (NIJ) Recidivism Forecasting Challenge dataset. The target is the first supervision risk-assessment score grouped into **Low (1–4), Medium (5–7), and High (8–10)**. The project compares a baseline with five machine-learning classifiers and interprets their results.

> **Important limitation:** This is an academic experiment on an existing dataset. It is not a validated risk assessment and must not be used for hiring, parole, policing, or decisions about real people. Dataset IDs refer to existing records whose target values are already present.

## Project questions

1. Can the available input features classify the dataset's supervision risk-score group?
2. Which tested classifier performs best on the held-out test set and during cross-validation?
3. Which features are most influential in the Random Forest model, and what limitations affect interpretation?

## Dataset

- **Source:** NIJ Recidivism Forecasting Challenge training data (official public data portal; see [Dataset information](Dataset_Information.txt)).
- **Dataset file:** `data/NIJ_DataSet.csv`
- **Rows retained for modeling:** 17,698 records with a non-missing target, as reported by the notebook.
- **Target:** `Supervision_Risk_Score_First`, grouped into Low/Medium/High.
- The identifier, target, supervision level, and recidivism outcome columns are excluded from model inputs.
- Dataset-specific variable definitions and source are documented in `Dataset_Information.txt`.

## Repository structure

```
├── data/
│   └── NIJ_DataSet.csv
├── notebooks/
│   └── risk_research_assistant_ml_project.ipynb
├── src/
│   └── risk_research_assistant.py
├── docs/figures/          # Saved project visualizations
├── outputs/               # Generated tables and cleaned data (local)
├── requirements.txt
├── Dataset_Information.txt
├── Links.txt
└── README.md
```

## Run locally

From the repository root:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python src/risk_research_assistant.py
```

Or launch Jupyter and open the notebook:

```bash
jupyter notebook
```

The script locates the dataset and output folder relative to its own file, so it can be launched from another working directory too. Keep the dataset at the path shown above.

## Workflow implemented

- Load, inspect, clean whitespace and duplicate rows, and remove rows missing the target.
- Exploratory analysis: data types, descriptive statistics, missingness through summaries, target distribution, outlier counts, histograms, box plots, correlation heatmap, and group-level target rates.
- Feature preparation and feature engineering; categorical encoding and numeric imputation/scaling are performed in a pipeline.
- Stratified 70/30 train-test split.
- Baseline: majority-class DummyClassifier.
- Five predictive classifiers: Naive Bayes, Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- GridSearchCV with 5-fold stratified cross-validation and macro-F1 scoring for the tuned models.
- Test evaluation: accuracy, macro precision, macro recall, macro F1, multiclass ROC-AUC, confusion matrices, ROC/PR curves, and feature importance.
- Results and cleaned data are saved under `outputs/`; figures are saved there when the code runs. Selected figures are also committed under `docs/figures/`.

## Current recorded test results

| Model | Accuracy | Macro F1 | Macro ROC-AUC |
|---|---:|---:|---:|
| Baseline (majority) | 0.4313 | 0.2009 | 0.5000 |
| Naive Bayes | 0.5019 | 0.5044 | 0.6874 |
| Logistic Regression | 0.6228 | 0.6283 | 0.8029 |
| Decision Tree | 0.5994 | 0.6059 | 0.7726 |
| Random Forest | 0.6143 | 0.6125 | 0.7992 |
| Gradient Boosting | 0.6352 | 0.6370 | 0.8125 |

Gradient Boosting had the highest recorded cross-validated macro F1 (0.6358). These are the project's existing run results; rerun the notebook/script to reproduce them in your environment.

## Reproducibility and course submission

- Fixed random seed: `42`.
- Train/test and cross-validation are stratified.
- Preprocessing is inside the model pipeline, preventing the test set from fitting imputation or scaling parameters.
- Dependencies are listed in `requirements.txt`.
- Dataset source and useful URLs are in `Dataset_Information.txt` and `Links.txt`.

The course PDF also requires a separate formal report (including related work and at least 15 verified scholarly references in IEEE style) and a 7–10 minute PowerPoint presentation. Those documents are **not currently included in this repository** and must be prepared and added for the final course submission.

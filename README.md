# Risk Research Assistant
### Machine Learning Course Project

A supervised machine-learning project that studies whether selected features from the **National Institute of Justice (NIJ) Recidivism Forecasting Challenge** dataset can classify a record's first supervision risk-assessment score into three groups: **Low, Medium, or High**.

This project follows a complete practical ML workflow: problem definition, dataset understanding, exploratory data analysis (EDA), preprocessing, model training, evaluation, comparison, visualization, and interpretation.

> **Research and ethical-use note**
> This is a classroom experiment using an existing public dataset. It is not a validated risk-assessment system and must not be used to make decisions about real people, including hiring, parole, policing, or supervision. The ID lookup in the project retrieves an existing dataset record; it does not assess an unknown person, and the record's target is already present in the dataset.

## 1. Problem definition

**Problem statement:** Determine how well the dataset's available input features classify the initial supervision risk score into Low, Medium, or High categories.

**Machine-learning task:** Supervised, multiclass classification.

**Target variable:** `Supervision_Risk_Score_First`

| Original score | Class label |
|---|---|
| 1–4 | Low |
| 5–7 | Medium |
| 8–10 | High |

**Project objectives**
- Understand and explore the selected real-world dataset.
- Prepare features and build a reproducible classification pipeline.
- Compare a simple baseline with five machine-learning classifiers.
- Evaluate models using multiple metrics, not accuracy alone.
- Interpret model results and feature importance while acknowledging limitations.

**Project questions**
1. Can the available input features classify the supervision risk-score group?
2. Which tested model performs best on the held-out test set and in cross-validation?
3. Which features are most influential in the Random Forest model, and what limitations affect that interpretation?

## 2. Dataset

- **Dataset:** NIJ Recidivism Forecasting Challenge — training dataset
- **Source:** [Official NIJ dataset portal](https://data.ojp.gov/)
- **Local file:** `data/NIJ_DataSet.csv`
- **Records retained for modeling:** 17,698 records with a non-missing target, as reported by the project notebook.
- **Dataset documentation:** See [Dataset_Information.txt](Dataset_Information.txt) for the source, target details, and relevant links.

The identifier, target, supervision-level, and recidivism-outcome columns are excluded from model inputs to reduce leakage and keep the prediction task focused on the defined target. Refer to the notebook and source code for the exact feature-preparation steps.

## 3. Repository contents

```
risk-research-assistant-ml-project/
├── data/
│   └── NIJ_DataSet.csv
├── notebooks/
│   └── risk_research_assistant_ml_project.ipynb
├── src/
│   └── risk_research_assistant.py
├── docs/
│   └── figures/                 # Selected saved project figures
├── requirements.txt              # Python dependencies
├── Dataset_Information.txt       # Dataset source and notes
├── Links.txt                    # Project and resource links
└── README.md
```

The `outputs/` directory is created locally when the code runs. Generated outputs are ignored by Git, so an empty `outputs/` folder does not need to be committed. Selected visualizations are kept in `docs/figures/`.

## 4. Methodology

The implemented workflow includes:

1. **Data loading and inspection** — review dataset dimensions, data types, target, missing values, duplicates, and descriptive statistics.
2. **Data cleaning** — standardize relevant text values, remove duplicate records, and exclude records without the target.
3. **EDA** — examine target distribution, feature distributions, outliers, correlations, and relationships with the target; generate and interpret plots.
4. **Preprocessing and feature preparation** — use pipeline-based imputation, categorical encoding, scaling where needed, and feature engineering.
5. **Data split** — stratified 70% training and 30% testing.
6. **Baseline** — majority-class `DummyClassifier`.
7. **Model comparison** — train and evaluate five classifiers:
   - Naive Bayes
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
8. **Tuning** — use stratified 5-fold cross-validation and GridSearchCV with macro F1 scoring for tuned models.
9. **Evaluation and interpretation** — compare models with accuracy, macro precision, macro recall, macro F1, multiclass ROC-AUC, confusion matrices, ROC/PR curves, and feature-importance analysis.

Preprocessing is fitted within the model pipeline to help prevent information from the test set leaking into training.

## 5. Recorded model results

The following are the project's recorded test-set results:

| Model | Accuracy | Macro F1 | Macro ROC-AUC |
|---|---:|---:|---:|
| Majority-class baseline | 0.4313 | 0.2009 | 0.5000 |
| Naive Bayes | 0.5019 | 0.5044 | 0.6874 |
| Logistic Regression | 0.6228 | 0.6283 | 0.8029 |
| Decision Tree | 0.5994 | 0.6059 | 0.7726 |
| Random Forest | 0.6143 | 0.6125 | 0.7992 |
| Gradient Boosting | **0.6352** | **0.6370** | **0.8125** |

Gradient Boosting achieved the highest recorded test accuracy, macro F1, and macro ROC-AUC among the listed models. It also achieved the highest recorded cross-validated macro F1 of **0.6358**.

These values are existing project results, not a guarantee that every run will produce identical values. Use the notebook and script to inspect the full experiment and reproduce the results in your own environment.

## 6. Run the project

### Requirements
- Python
- Packages listed in `requirements.txt`
- The dataset placed at `data/NIJ_DataSet.csv`

### Setup

From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
```

**Windows**
```bash
.venv\\Scripts\\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the Python script:

```bash
python src/risk_research_assistant.py
```

Or open the notebook:

```bash
jupyter notebook
```

Then open `notebooks/risk_research_assistant_ml_project.ipynb` and run the cells in order. The script uses paths relative to the project location, so it does not depend on the terminal's current directory.

## 7. Reproducibility

- Random seed: `42`
- Stratified train/test split and stratified cross-validation
- Preprocessing integrated into model pipelines
- Dependencies recorded in `requirements.txt`
- Dataset source and project resources documented in `Dataset_Information.txt` and `Links.txt`

## 8. Limitations and future work

The results describe patterns in this dataset only. They do not establish that the model is accurate, fair, or suitable for use outside this academic experiment. Risk-related data can reflect historical and institutional biases, and feature importance does not prove causation. The ID lookup is a record-retrieval demonstration, not a prediction for a new individual.

Possible future improvements include stronger external validation, deeper error and class-wise analysis, additional explainability methods, and further investigation of dataset bias and feature limitations.

## 9. Course submission checklist

The course instructions recommend a README and require the project to provide its dataset source, code, notebook, dependencies, preprocessing and evaluation details, results, and presentation materials.

Still to add for the final submission:
- **Formal project report** with the required sections and at least 15 verified scholarly references in IEEE style.
- **PowerPoint presentation** of approximately 7–10 minutes, including the implementation and key results.
- Presentation or demonstration link in `Links.txt`, when available.

Please verify all results and understand the code, graphs, and references before presenting or submitting the project.

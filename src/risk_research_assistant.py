# Risk Research Assistant - ML Project

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score)

RANDOM_STATE = 42
DATA_PATH = "data/NIJ_DataSet.csv"
os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid")

SCORE_COL = "Supervision_Risk_Score_First"
DROP_COLS = ["ID", SCORE_COL, "Supervision_Level_First",
             "Recidivism_Within_3years", "Recidivism_Arrest_Year1",
             "Recidivism_Arrest_Year2", "Recidivism_Arrest_Year3"]


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}", dpi=120)
    plt.show()


def number_from_text(value):
    """'3 or more' -> 3, '10 or more' -> 10, blank -> NaN."""
    if pd.isna(value):
        return np.nan
    found = re.search(r"\d+", str(value))
    return float(found.group()) if found else np.nan


# STEP 1: Load and clean the data (EDA)
df = pd.read_csv(DATA_PATH, na_values=[""])
text_cols = df.select_dtypes(exclude="number").columns
df[text_cols] = df[text_cols].apply(lambda col: col.str.strip())
df = df.drop_duplicates()
df = df.dropna(subset=[SCORE_COL])
print("Rows with a risk score:", len(df))

y = pd.cut(df[SCORE_COL], bins=[0, 4, 7, 10], labels=["Low", "Medium", "High"])
y = y.astype(str)
X = df.drop(columns=DROP_COLS)

X["Age_at_Release"] = X["Age_at_Release"].map(
    {"18-22": 0, "23-27": 1, "28-32": 2, "33-37": 3, "38-42": 4, "43-47": 5, "48 or older": 6})
X["Prison_Years"] = X["Prison_Years"].map(
    {"Less than 1 year": 0, "1-2 years": 1, "Greater than 2 to 3 years": 2, "More than 3 years": 3})
X["Education_Level"] = X["Education_Level"].map(
    {"Less than HS diploma": 0, "High School Diploma": 1, "At least some college": 2})

count_cols = [
    "Dependents", "Prior_Arrest_Episodes_Felony", "Prior_Arrest_Episodes_Misd",
    "Prior_Arrest_Episodes_Violent", "Prior_Arrest_Episodes_Property",
    "Prior_Arrest_Episodes_Drug", "_v1", "Prior_Conviction_Episodes_Felony",
    "Prior_Conviction_Episodes_Misd", "Prior_Conviction_Episodes_Prop",
    "Prior_Conviction_Episodes_Drug", "Delinquency_Reports",
    "Program_Attendances", "Program_UnexcusedAbsences", "Residence_Changes",
]
X[count_cols] = X[count_cols].apply(lambda col: col.map(number_from_text))
X["Total_Prior_Arrests"] = X[[
    "Prior_Arrest_Episodes_Felony", "Prior_Arrest_Episodes_Misd",
    "Prior_Arrest_Episodes_Violent", "Prior_Arrest_Episodes_Property",
    "Prior_Arrest_Episodes_Drug"]].sum(axis=1)
X["Total_Prior_Convictions"] = X[[
    "Prior_Conviction_Episodes_Felony", "Prior_Conviction_Episodes_Misd",
    "Prior_Conviction_Episodes_Prop", "Prior_Conviction_Episodes_Drug"]].sum(axis=1)
X["Residence_PUMA"] = X["Residence_PUMA"].astype(str)

numeric_cols = (
    ["Age_at_Release", "Education_Level", "Prison_Years",
     "Avg_Days_per_DrugTest", "DrugTests_THC_Positive", "DrugTests_Cocaine_Positive",
     "DrugTests_Meth_Positive", "DrugTests_Other_Positive",
     "Percent_Days_Employed", "Jobs_Per_Year",
     "Total_Prior_Arrests", "Total_Prior_Convictions"] + count_cols
)
X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
categorical_cols = [c for c in X.columns if c not in numeric_cols]

print("\nData types:\n", df.dtypes.value_counts().to_string())
print("\nBasic statistics (input features):")
print(X[numeric_cols].describe().T.round(2).to_string())

outlier_counts = {}
for col in numeric_cols:
    s = X[col].dropna()
    if s.nunique() <= 2:
        continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    outlier_counts[col] = int(((s < low) | (s > high)).sum())
print("\nOutliers per numeric feature (IQR rule), top 10:")
print(pd.Series(outlier_counts).sort_values(ascending=False).head(10).to_string())
print("Outliers are reported but kept: tree-based models are not strongly affected by them.")

shares = y.value_counts(normalize=True) * 100
print("\nRisk level shares (%):\n", shares.round(1).to_string())
print(f"The largest level is {shares.max() / shares.min():.2f} times the smallest.")
print("class_weight='balanced' is used for Logistic Regression, Decision Tree and Random Forest.")
print("Gradient Boosting and Naive Bayes do not use it, so their Medium/Low/High balance may differ.")

# STEP 2: EDA plots
order = ["Low", "Medium", "High"]
sns.countplot(x=y, order=order, hue=y, hue_order=order, legend=False,
              palette=["#55A868", "#DD8452", "#C44E52"])
plt.title("How many people are Low, Medium and High risk?")
plt.xlabel("")
save_plot("risk_01_level_counts.png")

X[["Total_Prior_Arrests", "Jobs_Per_Year", "Percent_Days_Employed",
   "Age_at_Release"]].hist(bins=30, figsize=(12, 7), color="#4C72B0")
plt.suptitle("Some input features")
save_plot("risk_02_histograms.png")

box_cols = ["Total_Prior_Arrests", "Percent_Days_Employed", "Jobs_Per_Year", "Age_at_Release"]
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, col in zip(axes, box_cols):
    sns.boxplot(x=y, y=X[col], order=order, hue=y, hue_order=order,
                palette=["#55A868", "#DD8452", "#C44E52"], legend=False, ax=ax)
    ax.set_title(col)
    ax.set_xlabel("")
save_plot("risk_03_boxplots_by_level.png")

corr = X[numeric_cols].assign(risk_score=df.loc[X.index, SCORE_COL]).corr()
plt.figure(figsize=(13, 11))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlations between numeric inputs and the risk score")
save_plot("risk_04_correlation_heatmap.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col in zip(axes, ["Gender", "Race"]):
    (y == "High").groupby(X[col]).mean().mul(100).plot(kind="bar", ax=ax, color="#C44E52")
    ax.set_title(f"% High risk by {col}")
    ax.set_ylabel("% High risk")
    ax.tick_params(axis="x", rotation=0)
save_plot("risk_05_high_risk_rate_by_group.png")

X.assign(**{SCORE_COL: df.loc[X.index, SCORE_COL], "Risk_Level": y}).to_csv(
    "outputs/cleaned_dataset.csv", index=False)

# STEP 3: Data splitting (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
print("Training rows:", len(X_train), "| Test rows:", len(X_test))


# STEP 4: Preprocessing and models
def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("fill", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))]), categorical_cols),
    ])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
models = {
    "Baseline (majority)": (DummyClassifier(strategy="most_frequent"), None),
    "Naive Bayes": (GaussianNB(), None),
    "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced"),
                            {"model__C": [0.01, 0.1, 1, 10]}),
    "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
                      {"model__max_depth": [3, 5, 10, None],
                       "model__min_samples_leaf": [5, 20]}),
    "Random Forest": (RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                             random_state=RANDOM_STATE, n_jobs=-1),
                      {"model__max_depth": [5, 10, None],
                       "model__min_samples_leaf": [1, 10]}),
    "Gradient Boosting": (GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
                          {"model__learning_rate": [0.05, 0.1],
                           "model__max_depth": [2, 3]}),
}

fitted, cv_scores = {}, {}
for name, (model, settings) in models.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
    if settings is None:
        fitted[name] = pipe.fit(X_train, y_train)
        cv_scores[name] = np.nan
    else:
        search = GridSearchCV(pipe, settings, cv=cv, scoring="f1_macro", n_jobs=-1)
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        cv_scores[name] = search.best_score_
        print(f"{name}: best settings {search.best_params_}, CV macro F1 {search.best_score_:.4f}")

# STEP 5: Test-set results
rows, preds, probas = [], {}, {}
for name, model in fitted.items():
    pred = model.predict(X_test)
    preds[name] = pred
    probas[name] = model.predict_proba(X_test)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision (macro)": precision_score(y_test, pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_test, pred, average="macro", zero_division=0),
        "ROC-AUC (macro, one vs rest)": roc_auc_score(
            y_test, probas[name], multi_class="ovr", average="macro", labels=model.classes_),
        "CV macro F1 (train)": cv_scores[name],
    })

results = pd.DataFrame(rows).set_index("Model").round(4)
print("\nTEST SET RESULTS")
print(results.to_string())
results.to_csv("outputs/risk_results_table.csv")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, name in zip(axes.flat, fitted):
    sns.heatmap(confusion_matrix(y_test, preds[name], labels=order), annot=True, fmt="d",
                cmap="Blues", cbar=False, ax=ax, xticklabels=order, yticklabels=order)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
save_plot("risk_08_confusion_matrices.png")

rf = fitted["Random Forest"]
rf_importance = pd.Series(rf.named_steps["model"].feature_importances_,
                          index=rf.named_steps["prep"].get_feature_names_out())
rf_top = rf_importance.sort_values(ascending=False).head(10)
print("\nTop 10 features (Random Forest):\n", rf_top.round(4).to_string())
rf_top.iloc[::-1].plot(kind="barh", figsize=(9, 6), color="#55A868")
plt.title("Top 10 features for risk level - Random Forest")
plt.xlabel("Importance")
save_plot("risk_09_rf_feature_importance.png")

# STEP 6: Risk checker by ID
best_name = results["CV macro F1 (train)"].idxmax()
best_model = fitted[best_name]
print(f"\nModel used for the checker: {best_name}")

best_proba = probas[best_name]
best_classes = list(best_model.classes_)
plt.figure(figsize=(8, 7))
for i, cls in enumerate(best_classes):
    fpr, tpr, _ = roc_curve((y_test == cls).astype(int), best_proba[:, i])
    plt.plot(fpr, tpr, label=f"{cls} vs the rest")
plt.plot([0, 1], [0, 1], "k--", label="Random guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC curves - {best_name}")
plt.legend()
save_plot("risk_06_roc_curves.png")

plt.figure(figsize=(8, 7))
for i, cls in enumerate(best_classes):
    precision, recall, _ = precision_recall_curve((y_test == cls).astype(int), best_proba[:, i])
    ap = average_precision_score((y_test == cls).astype(int), best_proba[:, i])
    plt.plot(recall, precision, label=f"{cls} (AP = {ap:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall curves - {best_name}")
plt.legend()
save_plot("risk_07_pr_curves.png")

id_to_row = dict(zip(df["ID"], df.index))
print("\n========== RISK LEVEL CHECKER ==========")
print("Enter a person's ID from the dataset. Type 'q' to quit.")
while True:
    text = input("\nEnter ID: ").strip()
    if text.lower() == "q":
        print("Goodbye!")
        break
    if not text.isdigit() or int(text) not in id_to_row:
        print("ID not found. Please enter a valid ID number.")
        continue
    row = id_to_row[int(text)]
    person = X.loc[[row]]
    predicted = best_model.predict(person)[0]
    chances = pd.Series(best_model.predict_proba(person)[0], index=best_model.classes_)
    print(f"\nID {text}")
    print(f"  Predicted risk level: {predicted.upper()}")
    print("  Chances:", ", ".join(f"{k} {v * 100:.0f}%" for k, v in chances.items()))
    print(f"  Actual score in the data: {df.loc[row, SCORE_COL]:.0f} (out of 10)")
    print("  Note: this is a student model for learning, not a real decision tool.")
print("Done.")

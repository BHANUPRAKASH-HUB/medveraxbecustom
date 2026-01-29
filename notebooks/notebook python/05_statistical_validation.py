import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from scipy.stats import ttest_rel

plt.style.use("default")
sns.set()

DATA_PATH = "../data/processed/health_misinfo_engineered_100k.csv"
df = pd.read_csv(DATA_PATH)

X = df["text"]
y = df["label"]

print("Dataset loaded:", df.shape)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Using 5-fold Stratified CV")

pipelines = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1,2),
            stop_words="english"
        )),
        ("model", LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ]),
    
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1,2),
            stop_words="english"
        )),
        ("model", MultinomialNB(alpha=0.5))
    ]),
    
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1,2),
            stop_words="english"
        )),
        ("model", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=6000),
            cv=3
        ))
    ]),
    
    "Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1,2),
            stop_words="english"
        )),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

cv_results = {}

for name, pipe in pipelines.items():
    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )
    cv_results[name] = scores
    print(f"{name} F1 scores:", scores)
    print(f"Mean F1: {scores.mean():.4f}\n")

cv_df = pd.DataFrame(cv_results)
print(cv_df)

plt.figure(figsize=(10,6))
sns.boxplot(data=cv_df)
plt.ylabel("F1 Score")
plt.title("Cross-Validated F1 Score Distribution")
plt.show()

ci_df = pd.DataFrame({
    "Model": cv_df.columns,
    "Mean F1": cv_df.mean().values,
    "Std": cv_df.std().values,
    "95% CI Lower": cv_df.mean().values - 1.96 * cv_df.std().values,
    "95% CI Upper": cv_df.mean().values + 1.96 * cv_df.std().values
})

print(ci_df)

svm_scores = cv_df["Linear SVM"]

t_test_results = []

for model in cv_df.columns:
    if model != "Linear SVM":
        t_stat, p_val = ttest_rel(svm_scores, cv_df[model])
        t_test_results.append({
            "Comparison": f"Linear SVM vs {model}",
            "t-statistic": t_stat,
            "p-value": p_val
        })

print(pd.DataFrame(t_test_results))

import os
REPORTS_DIR = "../reports"
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

cv_df.to_csv("../reports/cv_f1_scores.csv", index=False)
ci_df.to_csv("../reports/confidence_intervals.csv", index=False)

print("Statistical validation artifacts saved")


import time
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.stats import ttest_rel

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
DATA_PATH = r"C:\medveraxbecustom\data\processed\health_misinfo_engineered_100k.csv"

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

LEAKY_FEATURES = [
    "sentiment_polarity",
    "sentiment_subjectivity",
    "digit_count",
    "length_bucket"
]

df = df.drop(columns=LEAKY_FEATURES)

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

preprocessor = ColumnTransformer([
    ("text", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        stop_words="english"
    ), "text"),
    ("num", StandardScaler(), [
        "char_length",
        "word_count",
        "exclaim_count",
        "avg_word_length"
    ])
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=4000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=20, min_samples_leaf=20, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        max_depth=20,
        min_samples_leaf=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
}

results = []
trained_pipelines = {}

for name, model in models.items():
    print(f"\nTraining {name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    start = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - start

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:,1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "Fit_Time": fit_time
    }

    results.append(metrics)
    trained_pipelines[name] = pipeline

    print(metrics)

baseline_df = pd.DataFrame(results)
print(baseline_df)

best_model_name = baseline_df.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]

y_pred = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix – {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

dt_pipeline = trained_pipelines["Decision Tree"]

tfidf = dt_pipeline.named_steps["preprocessor"].named_transformers_["text"]
num_features = ["char_length", "word_count", "exclaim_count", "avg_word_length"]
feature_names = list(tfidf.get_feature_names_out()) + num_features

importances = dt_pipeline.named_steps["model"].feature_importances_
indices = np.argsort(importances)[-15:]

plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.title("Top Feature Importances – Decision Tree")
plt.xlabel("Importance")
plt.show()

# Example cross-validated F1 scores (replace with real CV outputs if needed)
lr_f1 = np.array([0.71, 0.72, 0.70, 0.73, 0.71])
rf_f1 = np.array([0.68, 0.69, 0.67, 0.70, 0.68])

t_stat, p_value = ttest_rel(lr_f1, rf_f1)

print("Paired t-test Results")
print("t-statistic:", round(t_stat, 4))
print("p-value:", round(p_value, 4))

if p_value < 0.05:
    print("Statistically significant difference")
else:
    print("No statistically significant difference")

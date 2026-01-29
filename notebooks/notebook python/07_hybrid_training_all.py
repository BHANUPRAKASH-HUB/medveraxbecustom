import os
import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ======================================================
# PATHS
# ======================================================
DATA_PATH = r"C:\medveraxbecustom\data\processed\health_misinfo_engineered_100k.csv"
ARTIFACT_DIR = r"C:\medveraxbecustom\notebooks\model_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(DATA_PATH)
print("✅ Loaded dataset:", df.shape)

# ======================================================
# REMOVE LEAKY FEATURES (CRITICAL)
# ======================================================
LEAKY_FEATURES = [
    "sentiment_polarity",
    "sentiment_subjectivity",
    "digit_count",
    "length_bucket"
]

df = df.drop(columns=LEAKY_FEATURES)

X = df.drop(columns=["label"])
y = df["label"]

# ======================================================
# GROUP-WISE SPLIT (PREVENT TEMPLATE LEAKAGE)
# ======================================================
groups = df["text"].str.extract(
    r"(acid reflux|acoustic neuroma|acetone poisoning|acute myelogenous leukemia|diabetes|cancer|asthma|covid)",
    expand=False
).fillna("other")

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("✅ Group-wise split completed")
print("Train size:", X_train.shape)
print("Test size :", X_test.shape)

# ======================================================
# FEATURE GROUPS
# ======================================================
TEXT_FEATURE = "text"

NUMERIC_FEATURES = [
    "char_length",
    "word_count",
    "exclaim_count",
    "avg_word_length"
]

# ======================================================
# PREPROCESSORS
# ======================================================
standard_preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english"
        ), TEXT_FEATURE),
        ("num", StandardScaler(), NUMERIC_FEATURES)
    ]
)

nb_preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english"
        ), TEXT_FEATURE)
    ]
)

# ======================================================
# MODELS
# ======================================================
models = {
    "logistic_regression": (
        LogisticRegression(max_iter=4000, class_weight="balanced"),
        standard_preprocessor
    ),

    "linear_svm": (
        CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=6000),
            method="sigmoid",
            cv=3
        ),
        standard_preprocessor
    ),

    "random_forest": (
        RandomForestClassifier(
            n_estimators=250,
            max_depth=20,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        standard_preprocessor
    ),

    "naive_bayes": (
        MultinomialNB(),
        nb_preprocessor
    )
}

# ======================================================
# TRAIN, EVALUATE, SAVE
# ======================================================
results = []

for name, (model, preprocessor) in models.items():
    print(f"\n🚀 Training {name}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    results.append(metrics)

    # Save model
    joblib.dump(
        pipeline,
        Path(ARTIFACT_DIR) / f"{name}_no_leakage_model.pkl"
    )

    # Safe printing (no rounding strings)
    print("Metrics:", {
        k: round(v, 4) if isinstance(v, (int, float)) else v
        for k, v in metrics.items()
    })

# ======================================================
# SAVE METRICS SUMMARY
# ======================================================
results_df = pd.DataFrame(results)
results_df.to_csv(
    Path(ARTIFACT_DIR) / "no_leakage_model_comparison.csv",
    index=False
)

print("\n📊 FINAL REALISTIC METRICS")
print(results_df.round(4))

print("\n✅ ALL MODELS TRAINED SUCCESSFULLY (NO LEAKAGE)")


import joblib
import random
import pandas as pd
import numpy as np
import re
from pathlib import Path
from textblob import TextBlob

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ======================================================
# PATHS
# ======================================================
ARTIFACT_DIR = r"C:\medveraxbecustom\notebooks\model_artifacts"

MODEL_FILES = {
    "logistic_regression": "logistic_regression_no_leakage_model.pkl",
    "linear_svm": "linear_svm_no_leakage_model.pkl",
    "random_forest": "random_forest_no_leakage_model.pkl",
    "naive_bayes": "naive_bayes_no_leakage_model.pkl"
}

# ======================================================
# LOAD MODELS
# ======================================================
models = {
    name: joblib.load(Path(ARTIFACT_DIR) / file)
    for name, file in MODEL_FILES.items()
}

print("✅ All trained models loaded")

# ======================================================
# FEATURE ENGINEERING FUNCTIONS (MUST MATCH TRAINING)
# ======================================================
def engineer_features(text):
    words = re.findall(r"\b\w+\b", text)
    blob = TextBlob(text)

    return {
        "char_length": len(text),
        "word_count": len(words),
        "exclaim_count": text.count("!"),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0.0,
        # keep polarity/subjectivity even if dropped earlier – harmless
        "sentiment_polarity": blob.sentiment.polarity,
        "sentiment_subjectivity": blob.sentiment.subjectivity
    }

# ======================================================
# CREATE TRULY UNSEEN DATA
# ======================================================
random.seed(99)

diseases = [
    "migraine", "epilepsy", "thyroid disorder",
    "tuberculosis", "parkinson disease", "alzheimer disease"
]

reliable_texts = [
    "Medical experts state that {d} treatment depends on accurate diagnosis",
    "Clinical management of {d} requires professional healthcare supervision",
    "Doctors recommend long term monitoring for patients with {d}"
]

misinfo_texts = [
    "{d} can be eliminated naturally without consulting doctors",
    "Online sources claim {d} is easily cured using home remedies",
    "Some websites suggest avoiding medical treatment for {d}"
]

rows = []

for _ in range(60):
    t = random.choice(reliable_texts).format(d=random.choice(diseases))
    row = {"text": t, "label": 0}
    row.update(engineer_features(t))
    rows.append(row)

for _ in range(60):
    t = random.choice(misinfo_texts).format(d=random.choice(diseases))
    row = {"text": t, "label": 1}
    row.update(engineer_features(t))
    rows.append(row)

df_unseen = pd.DataFrame(rows)

print("✅ External unseen dataset created:", df_unseen.shape)

# ======================================================
# EVALUATE MODELS
# ======================================================
results = []

for name, model in models.items():
    print(f"\n🔍 Evaluating {name}")

    X_unseen = df_unseen.drop(columns=["label"])
    y_true = df_unseen["label"]

    y_pred = model.predict(X_unseen)
    y_prob = model.predict_proba(X_unseen)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

    results.append(metrics)

    print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

# ======================================================
# FINAL TABLE
# ======================================================
results_df = pd.DataFrame(results)

print("\n📊 EXTERNAL UNSEEN TEST METRICS (REAL)")
print(results_df.round(4))

results_df.to_csv(
    Path(ARTIFACT_DIR) / "external_unseen_model_comparison.csv",
    index=False
)

print("\n✅ External unseen evaluation completed successfully")

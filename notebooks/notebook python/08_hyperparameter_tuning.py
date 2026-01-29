import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

import optuna


df = pd.read_csv("../data/processed/health_misinfo_engineered_100k.csv")

# Remove leaky features
df = df.drop(columns=[
    "digit_count",
    "sentiment_polarity",
    "sentiment_subjectivity",
    "length_bucket"
])

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


preprocessor = ColumnTransformer([
    ("text", TfidfVectorizer(stop_words="english"), "text"),
    ("num", StandardScaler(), [
        "char_length", "word_count",
        "exclaim_count", "avg_word_length"
    ])
])


def evaluate(model, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }


baseline_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ))
])

baseline_lr.fit(X_train, y_train)
results = [evaluate(baseline_lr, "LR Baseline")]


param_dist_lr = {
    "preprocessor__text__max_features": [10000, 20000, 30000],
    "preprocessor__text__ngram_range": [(1,1), (1,2)],
    "model__C": np.logspace(-2, 1, 10)
}

rand_lr = RandomizedSearchCV(
    baseline_lr,
    param_distributions=param_dist_lr,
    n_iter=10,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    random_state=42
)

rand_lr.fit(X_train, y_train)
results.append(evaluate(rand_lr.best_estimator_, "LR Tuned (Randomized)"))


svm_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearSVC(class_weight="balanced", max_iter=6000))
])

svm_grid = GridSearchCV(
    svm_pipe,
    {"model__C": [0.01, 0.1, 1, 10]},
    scoring="f1",
    cv=3,
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)

svm_calibrated = CalibratedClassifierCV(
    svm_grid.best_estimator_,
    method="sigmoid",
    cv=3
)

svm_calibrated.fit(X_train, y_train)
results.append(evaluate(svm_calibrated, "SVM Tuned"))


rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

rf_grid = GridSearchCV(
    rf_pipe,
    {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20, None]
    },
    scoring="roc_auc",
    cv=3,
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
results.append(evaluate(rf_grid.best_estimator_, "Random Forest Tuned"))


nb_pipe = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("text", TfidfVectorizer(stop_words="english"), "text")
    ])),
    ("model", MultinomialNB())
])

nb_grid = GridSearchCV(
    nb_pipe,
    {"model__alpha": [0.1, 0.5, 1.0, 2.0]},
    scoring="roc_auc",
    cv=3,
    n_jobs=-1
)

nb_grid.fit(X_train, y_train)
results.append(evaluate(nb_grid.best_estimator_, "Naive Bayes Tuned"))


def objective(trial):
    max_features = trial.suggest_categorical("max_features", [10000, 20000, 30000])
    C = trial.suggest_float("C", 0.01, 10, log=True)

    vec = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1,2)
    )

    X_tr = vec.fit_transform(X_train["text"])
    X_te = vec.transform(X_test["text"])

    model = LogisticRegression(
        C=C,
        max_iter=3000,
        class_weight="balanced"
    )

    model.fit(X_tr, y_train)
    y_prob = model.predict_proba(X_te)[:,1]

    return roc_auc_score(y_test, y_prob)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best Bayesian ROC-AUC:", study.best_value)


def plot_learning_curve(estimator, title):
    sizes, train, test = learning_curve(
        estimator,
        X_train, y_train,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1
    )

    plt.plot(sizes, train.mean(axis=1), label="Train")
    plt.plot(sizes, test.mean(axis=1), label="Validation")
    plt.title(title)
    plt.xlabel("Training Size")
    plt.ylabel("ROC-AUC")
    plt.legend()
    plt.grid()
    plt.show()

plot_learning_curve(rand_lr.best_estimator_, "Learning Curve – Logistic Regression")


results_df = pd.DataFrame(results)
print(results_df)


plt.figure(figsize=(10,5))
sns.barplot(data=results_df, x="model", y="roc_auc")
plt.title("Model Comparison – ROC AUC")
plt.xticks(rotation=30)
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
sns.barplot(data=results_df, x="model", y="f1_score")
plt.title("Model Comparison – F1 Score")
plt.xticks(rotation=30)
plt.grid()
plt.show()


from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import ttest_rel


best_lr = rand_lr.best_estimator_
best_nb = nb_grid.best_estimator_


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_scores = cross_val_score(
    best_lr,
    X,
    y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

nb_scores = cross_val_score(
    best_nb,
    X,
    y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print("LR ROC-AUC scores:", lr_scores)
print("NB ROC-AUC scores:", nb_scores)


t_stat, p_value = ttest_rel(lr_scores, nb_scores)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")


alpha = 0.05

if p_value < alpha:
    print("✅ Statistically significant difference between models")
else:
    print("❌ No statistically significant difference detected")


import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.plot(lr_scores, marker="o", label="Logistic Regression")
plt.plot(nb_scores, marker="o", label="Naive Bayes")
plt.xlabel("CV Fold")
plt.ylabel("ROC-AUC")
plt.title("Paired ROC-AUC Scores Across Folds")
plt.legend()
plt.grid()
plt.show()


stat_summary = pd.DataFrame({
    "Model A": ["Logistic Regression (Tuned)"],
    "Model B": ["Naive Bayes (Tuned)"],
    "Mean ROC-AUC A": [lr_scores.mean()],
    "Mean ROC-AUC B": [nb_scores.mean()],
    "P-value": [p_value]
})

print(stat_summary)

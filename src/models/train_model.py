# import os, json, joblib, pandas as pd
# from datetime import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# DATA_PATH = "data/synthetic_health_misinfo_15k.csv"
# ARTIFACT_DIR = "model_artifacts"
# os.makedirs(ARTIFACT_DIR, exist_ok=True)

# df = pd.read_csv(DATA_PATH)
# X = df["text"].astype(str)
# y = df["label"]

# vectorizer = TfidfVectorizer(
#     max_features=20000,
#     stop_words="english",
#     ngram_range=(1,2),
#     min_df=2,
#     max_df=0.95
# )

# X_vec = vectorizer.fit_transform(X)

# X_tr, X_te, y_tr, y_te = train_test_split(
#     X_vec, y, test_size=0.2, stratify=y, random_state=42
# )

# # Base model
# base_lr = LogisticRegression(
#     max_iter=3000,
#     class_weight="balanced"
# )

# # ✅ Probability calibration (IMPORTANT)
# model = CalibratedClassifierCV(base_lr, method="sigmoid", cv=5)
# model.fit(X_tr, y_tr)

# pred = model.predict(X_te)
# prob = model.predict_proba(X_te)[:,1]

# cm = confusion_matrix(y_te, pred)

# metrics = {
#     "accuracy": round(accuracy_score(y_te, pred),4),
#     "precision": round(precision_score(y_te, pred),4),
#     "recall": round(recall_score(y_te, pred),4),
#     "f1_score": round(f1_score(y_te, pred),4),
#     "confusion_matrix": {
#         "TN": int(cm[0][0]), "FP": int(cm[0][1]),
#         "FN": int(cm[1][0]), "TP": int(cm[1][1])
#     }
# }

# joblib.dump(model, f"{ARTIFACT_DIR}/model_logistic_regression.pkl")
# joblib.dump(vectorizer, f"{ARTIFACT_DIR}/vectorizer.pkl")

# with open(f"{ARTIFACT_DIR}/metrics.json","w") as f:
#     json.dump(metrics,f,indent=4)

# with open(f"{ARTIFACT_DIR}/config.json","w") as f:
#     json.dump({
#         "rows": len(df),
#         "model": "Calibrated Logistic Regression",
#         "trained_on": datetime.utcnow().isoformat()
#     }, f, indent=4)

# print("✅ Robust model training completed")

import os, json, joblib
import pandas as pd
from pathlib import Path
from datetime import datetime, UTC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "health_misinfo_clean_50k.csv"
ARTIFACT_DIR = BASE_DIR / "model_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded rows: {len(df)}")

X = df["text"].astype(str)
y = df["label"]

vectorizer = TfidfVectorizer(
    max_features=30000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95
)

X_vec = vectorizer.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_vec, y, test_size=0.2, stratify=y, random_state=42
)

base_lr = LogisticRegression(
    max_iter=4000,
    class_weight="balanced",
    n_jobs=-1
)

model = CalibratedClassifierCV(
    estimator=base_lr,
    method="sigmoid",
    cv=3
)

model.fit(X_tr, y_tr)

pred = model.predict(X_te)
prob = model.predict_proba(X_te)[:, 1]

cm = confusion_matrix(y_te, pred)

metrics = {
    "accuracy": round(accuracy_score(y_te, pred), 4),
    "precision": round(precision_score(y_te, pred), 4),
    "recall": round(recall_score(y_te, pred), 4),
    "f1_score": round(f1_score(y_te, pred), 4),
    "confusion_matrix": {
        "TN": int(cm[0][0]),
        "FP": int(cm[0][1]),
        "FN": int(cm[1][0]),
        "TP": int(cm[1][1])
    }
}

joblib.dump(model, f"{ARTIFACT_DIR}/model_logistic_regression.pkl")
joblib.dump(vectorizer, f"{ARTIFACT_DIR}/vectorizer.pkl")

with open(f"{ARTIFACT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

with open(f"{ARTIFACT_DIR}/config.json", "w") as f:
    json.dump({
        "rows": len(df),
        "model": "Calibrated Logistic Regression",
        "trained_on": datetime.now(UTC).isoformat()
    }, f, indent=4)

print("✅ Robust model training completed for 100k dataset")



# from src.loggers.logger import get_logger
# from src.exceptions.custom_exceptions import MedVeraxException

# logger = get_logger(__name__)

# try:
#     logger.info("Training started")

#     df = pd.read_csv(DATA_PATH)
#     logger.info(f"Loaded dataset with {len(df)} rows")

#     # training code...

#     logger.info("Model training completed successfully")

# except Exception as e:
#     logger.error("Training pipeline failed", exc_info=True)
#     raise MedVeraxException("Model training failed", e)

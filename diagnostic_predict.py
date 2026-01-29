import joblib
import pandas as pd
import numpy as np
import re

def engineer_features(text):
    words = re.findall(r"\b\w+\b", str(text))
    return {
        "text": text,
        "char_length": len(str(text)),
        "word_count": len(words),
        "exclaim_count": str(text).count("!"),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0.0
    }

model_path = "model_artifacts/logistic_regression_no_leakage_model.pkl"
model = joblib.load(model_path)

# Create 5 test samples
texts = [
    "This is sample one",
    "Another medical claim here",
    "Third sample for testing",
    "Fourth one is risky",
    "Fifth one is safe"
]
rows = []
for t in texts:
    row = engineer_features(t)
    rows.append(row)

df = pd.DataFrame(rows)
print("Input shape:", df.shape)

try:
    y_pred = model.predict(df)
    print("y_pred shape:", y_pred.shape)
    print("y_pred:", y_pred)
except Exception as e:
    print("Predict error:", e)

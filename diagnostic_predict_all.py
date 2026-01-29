import joblib
import pandas as pd
import numpy as np
import re
import os

def engineer_features(text):
    words = re.findall(r"\b\w+\b", str(text))
    return {
        "text": text,
        "char_length": len(str(text)),
        "word_count": len(words),
        "exclaim_count": str(text).count("!"),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0.0
    }

ARTIFACT_DIR = "model_artifacts"
MODELS = {
    "Logistic Regression": "logistic_regression_no_leakage_model.pkl",
    "Naive Bayes": "naive_bayes_no_leakage_model.pkl",
    "Linear SVM": "linear_svm_no_leakage_model.pkl",
    "Random Forest": "random_forest_no_leakage_model.pkl"
}

# Create 10 test samples
texts = [f"Sample text {i}" for i in range(10)]
rows = [engineer_features(t) for t in texts]
df = pd.DataFrame(rows)
print(f"Input shape: {df.shape}")

for name, filename in MODELS.items():
    path = os.path.join(ARTIFACT_DIR, filename)
    if not os.path.exists(path):
        print(f"[-] {name} missing")
        continue
        
    model = joblib.load(path)
    print(f"\n--- Model: {name} ---")
    print(f"Type: {type(model)}")
    
    try:
        y_pred = model.predict(df)
        print(f"y_pred shape: {y_pred.shape}")
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(df)
            print(f"y_prob shape: {y_prob.shape}")
    except Exception as e:
        print(f"Error: {e}")

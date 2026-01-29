import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "health_misinfo_clean_100k.csv"
OUT_DIR = BASE_DIR / "model_artifacts"

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)

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

    joblib.dump(vectorizer, f"{OUT_DIR}/vectorizer.pkl")

    print("✅ Features built")
    print("Shape:", X_vec.shape)

if __name__ == "__main__":
    main()

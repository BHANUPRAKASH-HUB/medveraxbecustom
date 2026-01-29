import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from textblob import TextBlob
import nltk

plt.style.use("default")
sns.set()

import os
DATA_PATH = "../data/processed/health_misinfo_clean_100k1.csv"
if not os.path.exists(DATA_PATH):
    # Fallback to standard name if specific version not found
    DATA_PATH = "../data/processed/health_misinfo_clean_100k.csv"

df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset from {DATA_PATH}")
print("Shape:", df.shape)

overview = {
    "Rows": df.shape[0],
    "Columns": df.shape[1],
    "Memory Usage (MB)": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
}

print(overview)

print(df.dtypes.value_counts())

df['char_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().apply(len)
df['exclaim_count'] = df['text'].str.count('!')
df['digit_count'] = df['text'].str.count(r'\d')

print(df[['char_length', 'word_count', 'exclaim_count', 'digit_count']].head())

def get_sentiment(text):
    blob = TextBlob(str(text))
    return pd.Series([blob.sentiment.polarity, blob.sentiment.subjectivity])

print("Generating sentiment features...")
df[['sentiment_polarity', 'sentiment_subjectivity']] = df['text'].apply(get_sentiment)
print(df[['sentiment_polarity', 'sentiment_subjectivity']].head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Analyze top keywords using TF-IDF
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'].astype(str))
feature_names = tfidf.get_feature_names_out()

print(f"Top 100 TF-IDF features extracted. Matrix shape: {tfidf_matrix.shape}")
print("Sample features:", feature_names[:80])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(df['char_length'], bins=50, kde=True, ax=axes[0,0])
axes[0,0].set_title("Character Length Distribution")

sns.histplot(df['word_count'], bins=40, kde=True, ax=axes[0,1])
axes[0,1].set_title("Word Count Distribution")

sns.histplot(df['exclaim_count'], bins=20, kde=False, ax=axes[1,0])
axes[1,0].set_title("Exclamation Count Distribution")

sns.histplot(df['digit_count'], bins=20, kde=False, ax=axes[1,1])
axes[1,1].set_title("Digit Count Distribution")

plt.tight_layout()
plt.show()

stats_df = pd.DataFrame({
    "Feature": ["char_length", "word_count", "exclaim_count", "digit_count"],
    "Mean": [
        df['char_length'].mean(),
        df['word_count'].mean(),
        df['exclaim_count'].mean(),
        df['digit_count'].mean()
    ],
    "Std": [
        df['char_length'].std(),
        df['word_count'].std(),
        df['exclaim_count'].std(),
        df['digit_count'].std()
    ],
    "Skewness": [
        skew(df['char_length']),
        skew(df['word_count']),
        skew(df['exclaim_count']),
        skew(df['digit_count'])
    ],
    "Kurtosis": [
        kurtosis(df['char_length']),
        kurtosis(df['word_count']),
        kurtosis(df['exclaim_count']),
        kurtosis(df['digit_count'])
    ]
})

print(stats_df)

print(df.groupby('label')[['char_length','word_count','exclaim_count','digit_count']].mean())

plt.figure(figsize=(12,6))
sns.boxplot(data=df, x='label', y='word_count')
plt.title("Word Count by Class")
plt.show()

corr_features = ['char_length','word_count','exclaim_count','digit_count','label']
corr_matrix = df[corr_features].corr()

print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

df['avg_word_length'] = df['char_length'] / (df['word_count'] + 1)

print(df[['avg_word_length']].describe())

df['length_bucket'] = pd.cut(
    df['word_count'],
    bins=[0,5,10,20,50,100],
    labels=['very_short','short','medium','long','very_long']
)

print(df['length_bucket'].value_counts())

before_features = 2
after_features = df.shape[1]

print(pd.DataFrame({
    "Stage": ["Before Feature Engineering", "After Feature Engineering"],
    "Feature Count": [before_features, after_features]
}))

import os
REPORTS_DIR = "../reports"
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

# Save correlation matrix
corr_matrix.to_csv("../reports/correlation_matrix.csv")

# Save engineered dataset
df.to_csv("../data/processed/health_misinfo_engineered_100k.csv", index=False)

print("EDA artifacts and engineered dataset saved")

print(df.head())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# LOAD DATA
# ======================================================
DATA_PATH = r"C:\medveraxbecustom\data\processed\health_misinfo_engineered_100k.csv"

df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded")
print("Shape:", df.shape)

# ======================================================
# 3️⃣ DATA SUMMARY STATISTICS
# ======================================================
print("\n📌 BASIC INFO")
print(df.info())

print("\n📌 NUMERICAL SUMMARY")
print(df.describe())

print("\n📌 CLASS DISTRIBUTION")
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True))

print("\n📌 MISSING VALUES")
print(df.isnull().sum())

print("\n📌 CATEGORICAL SUMMARY (length_bucket)")
if "length_bucket" in df.columns:
    print(df["length_bucket"].value_counts())

# ======================================================
# 4️⃣ BASIC DATA VISUALIZATION
# ======================================================
sns.set(style="whitegrid")

# -------------------------------
# Class Distribution
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="label", data=df)
plt.title("Class Distribution (0 = Reliable, 1 = Misinformation)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# -------------------------------
# Text Length Distribution
# -------------------------------
plt.figure(figsize=(6,4))
sns.histplot(df["char_length"], bins=30, kde=True)
plt.title("Character Length Distribution")
plt.xlabel("Character Length")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# Word Count Distribution
# -------------------------------
plt.figure(figsize=(6,4))
sns.histplot(df["word_count"], bins=30, kde=True)
plt.title("Word Count Distribution")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# Avg Word Length vs Label
# -------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="label", y="avg_word_length", data=df)
plt.title("Average Word Length by Class")
plt.xlabel("Label")
plt.ylabel("Average Word Length")
plt.show()

# -------------------------------
# Sentiment Polarity vs Label
# -------------------------------
if "sentiment_polarity" in df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="label", y="sentiment_polarity", data=df)
    plt.title("Sentiment Polarity by Class")
    plt.xlabel("Label")
    plt.ylabel("Polarity")
    plt.show()

# -------------------------------
# Correlation Heatmap (Numeric)
# -------------------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

plt.figure(figsize=(10,6))
sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

print("\n✅ Data summary statistics and visualizations completed")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

DATA_PATH = r"C:\medveraxbecustom\data\processed\health_misinfo_engineered_100k.csv"
df = pd.read_csv(DATA_PATH)

sns.set(style="whitegrid")

# ===============================
# Class-wise Character Length
# ===============================
plt.figure(figsize=(6,4))
sns.boxplot(x="label", y="char_length", data=df)
plt.title("Character Length by Class")
plt.xlabel("Label (0 = Reliable, 1 = Misinformation)")
plt.ylabel("Character Length")
plt.show()

# ===============================
# Class-wise Word Count
# ===============================
plt.figure(figsize=(6,4))
sns.boxplot(x="label", y="word_count", data=df)
plt.title("Word Count by Class")
plt.xlabel("Label")
plt.ylabel("Word Count")
plt.show()

# ===============================
# Class-wise Digit Count
# ===============================
plt.figure(figsize=(6,4))
sns.boxplot(x="label", y="digit_count", data=df)
plt.title("Digit Count by Class")
plt.xlabel("Label")
plt.ylabel("Digit Count")
plt.show()

# ===============================
# Class-wise Sentiment Polarity
# ===============================
plt.figure(figsize=(6,4))
sns.boxplot(x="label", y="sentiment_polarity", data=df)
plt.title("Sentiment Polarity by Class")
plt.xlabel("Label")
plt.ylabel("Polarity")
plt.show()


# ===============================
# Correlation with Target Label
# ===============================
numeric_cols = [
    "char_length",
    "word_count",
    "digit_count",
    "sentiment_polarity",
    "sentiment_subjectivity",
    "avg_word_length"
]

corr = df[numeric_cols + ["label"]].corr()["label"].sort_values(ascending=False)

plt.figure(figsize=(6,4))
sns.barplot(x=corr.values, y=corr.index)
plt.title("Correlation of Engineered Features with Target Label")
plt.xlabel("Correlation with Label")
plt.ylabel("Feature")
plt.show()

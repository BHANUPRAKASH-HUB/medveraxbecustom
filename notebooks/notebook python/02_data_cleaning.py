import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")
sns.set()

DATA_PATH = "../data/raw/health_misinfo_raw_100k.csv"

df_raw = pd.read_csv(DATA_PATH)

print("Raw dataset loaded")
print("Shape:", df_raw.shape)

before_stats = {
    "Total Rows": df_raw.shape[0],
    "Missing Values": df_raw.isnull().sum().sum(),
    "Duplicate Rows": df_raw.duplicated().sum(),
    "Duplicate Texts": df_raw['text'].duplicated().sum(),
    "Incorrect Data Types": sum(df_raw.dtypes != ['object', 'int64'])
}

print(before_stats)

import re

class DataCleaningPipeline:
    
    def __init__(self, df):
        self.df = df.copy()
    
    def remove_duplicates(self):
        before = self.df.shape[0]
        # Remove exact duplicates
        self.df.drop_duplicates(inplace=True)
        # Handle redundant text samples individually (subset text)
        self.df = self.df.drop_duplicates(subset=['text'])
        after = self.df.shape[0]
        print(f"Removed {before - after} duplicate rows (total/subset)")
        return self
    
    def handle_missing_values(self):
        missing_before = self.df.isnull().sum().sum()
        # Specifically drop where text is null
        self.df.dropna(subset=['text'], inplace=True)
        missing_after = self.df.isnull().sum().sum()
        print(f"Missing values (text) before: {missing_before}, after: {missing_after}")
        return self
    
    def clean_text_noise(self):
        def _clean(text):
            if not isinstance(text, str): return text
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove special char noise patterns (!!!$$$, etc)
            text = re.sub(r'!!!\$\$\$|\$\$\$!!!', '', text)
            # Normalize whitespace and strip
            text = re.sub(r'\s+', ' ', text).strip()
            return text
            
        self.df['text'] = self.df['text'].apply(_clean)
        print("HTML tags, URLs, and noisy symbols removed")
        return self

    def handle_outliers(self):
        before = self.df.shape[0]
        # Remove extremely long texts using statistical quantile (99.9th percentile)
        self.df['length'] = self.df['text'].str.len()
        upper_limit = self.df['length'].quantile(0.999)
        self.df = self.df[self.df['length'] <= upper_limit]
        self.df.drop(columns=['length'], inplace=True)
        after = self.df.shape[0]
        print(f"Removed {before - after} length outliers")
        return self

    def fix_data_types(self):
        self.df['text'] = self.df['text'].astype(str)
        self.df['label'] = self.df['label'].astype(int)
        print("Data types validated")
        return self
    
    def get_report(self):
        report = {
            "Total Rows": self.df.shape[0],
            "Missing Values": self.df.isnull().sum().sum(),
            "Duplicate Rows": self.df.duplicated().sum(),
            "Duplicate Texts": self.df['text'].duplicated().sum(),
            "Incorrect Data Types": sum(self.df.dtypes != ['object', 'int64'])
        }
        return report

pipeline = DataCleaningPipeline(df_raw)

df_clean = (
    pipeline
    .handle_missing_values()
    .remove_duplicates()
    .clean_text_noise()
    .handle_outliers()
    .fix_data_types()
    .df
)

print("Advanced cleaning pipeline completed")
print("Final clean shape:", df_clean.shape)

after_stats = pipeline.get_report()
print(after_stats)

comparison_df = pd.DataFrame({
    "Metric": before_stats.keys(),
    "Before Cleaning": before_stats.values(),
    "After Cleaning": after_stats.values()
})

print(comparison_df)

quality_df = comparison_df.set_index("Metric")

quality_df.plot(kind="bar", figsize=(10,5))
plt.title("Data Quality Before vs After Cleaning")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

import os
OUTPUT_DIR = "../data/processed"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_PATH = "../data/processed/health_misinfo_clean_100k1.csv"

df_clean.to_csv(OUTPUT_PATH, index=False)

print("Cleaned dataset saved to:", OUTPUT_PATH)

df_clean.info()

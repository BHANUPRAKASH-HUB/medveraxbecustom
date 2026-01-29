import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")
sns.set()

# Path to raw data
DATA_PATH = "../data/raw/health_misinfo_raw_100k.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print(df.head())

df.info()

label_counts = df['label'].value_counts()
label_ratio = df['label'].value_counts(normalize=True)

print("Label Counts:")
print(label_counts)

print("\nLabel Ratios:")
print(label_ratio)

sns.countplot(x='label', data=df)
plt.title("Target Class Distribution")
plt.show()

missing = df.isnull().sum()

print("Missing values per column:")
print(missing)

duplicates = df.duplicated().sum()
print("Number of duplicate rows:", duplicates)

# Drop missing values
df = df.dropna(subset=['text'])

# Drop duplicates
df = df.drop_duplicates()

# Feature Engineering: Length and Word Count
df['char_length'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

print("Data cleaned and features created!")
print("New Shape:", df.shape)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df['char_length'], bins=50, kde=True, ax=ax[0])
ax[0].set_title("Character Length Distribution")

sns.histplot(df['word_count'], bins=40, kde=True, ax=ax[1])
ax[1].set_title("Word Count Distribution")

plt.show()

print(df.groupby('label')[['char_length', 'word_count']].mean())

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
X_counts = cv.fit_transform(df['text'])

vocab_size = len(cv.vocabulary_)
print("Vocabulary size:", vocab_size)

avg_words = df['word_count'].mean()
avg_chars = df['char_length'].mean()

print(f"Average words per text: {avg_words:.2f}")
print(f"Average characters per text: {avg_chars:.2f}")

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def stopword_ratio(text):
    words = text.lower().split()
    if len(words) == 0:
        return 0
    return sum(1 for w in words if w in stop_words) / len(words)

df['stopword_ratio'] = df['text'].apply(stopword_ratio)

print(df['stopword_ratio'].describe())

print(df.groupby('label')['stopword_ratio'].mean())

duplicate_texts = df['text'].duplicated().sum()
print("Duplicate texts:", duplicate_texts)

quality_report = pd.DataFrame({
    "Metric": [
        "Total Rows",
        "Total Columns",
        "Missing Values",
        "Duplicate Rows",
        "Duplicate Texts",
        "Vocabulary Size",
        "Avg Words per Text",
        "Avg Characters per Text"
    ],
    "Value": [
        df.shape[0],
        df.shape[1],
        df.isnull().sum().sum(),
        df.duplicated().sum(),
        duplicate_texts,
        vocab_size,
        round(avg_words, 2),
        round(avg_chars, 2)
    ]
})

quality_report

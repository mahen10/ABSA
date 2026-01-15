# ============================
# FILE: 4_sentiment_classification.py
# Deskripsi:
# Klasifikasi Sentimen
# Logistic Regression + TF-IDF
# DENGAN class_weight (IMBALANCE SAFE)
# ============================

import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ============================
# 1. Load data
# ============================
input_path = os.path.join("output", "absa_labeled_numeric.xlsx")
df = pd.read_excel(input_path)

# ============================
# 2. Filter data valid
# ============================
df = df[["processed_opinion", "label_text"]].dropna()
df["label_text"] = df["label_text"].str.lower().str.strip()

df = df[df["label_text"].isin(["positive", "negative"])]
df = df[df["processed_opinion"].str.strip() != ""]

if df.empty:
    raise ValueError("❌ Tidak ada data valid untuk klasifikasi")

# ============================
# 3. Split X dan y
# ============================
X = df["processed_opinion"]
y = df["label_text"]

# ============================
# 4. TF-IDF
# ============================
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    max_df=0.95,
    min_df=2
)

X_tfidf = tfidf.fit_transform(X)

# ============================
# 5. Simpan TF-IDF (opsional, untuk skripsi)
# ============================
tfidf_df = pd.DataFrame(
    X_tfidf.toarray(),
    columns=tfidf.get_feature_names_out()
)
tfidf_df["label_text"] = y.values

tfidf_df.to_excel("output/tfidf_output.xlsx", index=False)
print("✅ TF-IDF disimpan")

# ============================
# 6. Train-test split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================
# 7. Logistic Regression (CLASS WEIGHT!)
# ============================
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",   # Class Wight
    solver="liblinear"
)

model.fit(X_train, y_train)

# ============================
# 8. Evaluasi
# ============================
y_pred = model.predict(X_test)

print("\n=== Classification Report (Class Weight) ===")
print(classification_report(y_test, y_pred, zero_division=0))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

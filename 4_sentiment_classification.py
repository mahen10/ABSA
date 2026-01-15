# =====================================================
# FILE: 4_sentiment_classification.py
# Klasifikasi Sentimen – Logistic Regression + TF-IDF
# Fokus evaluasi (TANPA output Excel berat)
# =====================================================

import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

def run(input_path, output_dir):
    # ============================
    # Load data
    # ============================
    if not os.path.exists(input_path):
        raise FileNotFoundError("File input tidak ditemukan")

    df = pd.read_excel(input_path)

    required_cols = {"processed_opinion", "label_text"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Kolom processed_opinion / label_text tidak ditemukan")

    df = df[["processed_opinion", "label_text"]].dropna()
    df["label_text"] = df["label_text"].str.lower().str.strip()

    df = df[df["label_text"].isin(["positive", "negative"])]
    df = df[df["processed_opinion"].astype(str).str.strip() != ""]

    if df.empty:
        raise ValueError("Data kosong setelah preprocessing")

    # ============================
    # Split X dan y
    # ============================
    X = df["processed_opinion"]
    y = df["label_text"]

    # ============================
    # TF-IDF (RINGAN)
    # ============================
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=3,
        stop_words="english"
    )

    X_tfidf = tfidf.fit_transform(X)

    # ============================
    # Train-Test Split
    # ============================
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ============================
    # Logistic Regression
    # ============================
    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    )

    model.fit(X_train, y_train)

    # ============================
    # Evaluasi
    # ============================
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

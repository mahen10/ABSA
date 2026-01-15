# ============================
# FILE: 4_sentiment_classification.py
# Deskripsi:
# Klasifikasi Sentimen Aspect-Based Opinion
# TF-IDF + Logistic Regression (Binary)
# TANPA DATA LEAKAGE
# CLASS WEIGHT BALANCED
# ============================

import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


# ============================
# FUNGSI UTAMA (AMAN DI-IMPORT)
# ============================
def run(input_path: str, output_dir: str):
    """
    Parameters
    ----------
    input_path : str
        Path ke file Excel hasil post-ABSA preprocessing
    output_dir : str
        Folder output untuk menyimpan hasil TF-IDF (opsional)

    Returns
    -------
    dict
        accuracy, classification_report, confusion_matrix, model
    """

    # ============================
    # 1. VALIDASI FILE
    # ============================
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"❌ File input tidak ditemukan: {input_path}"
        )

    # ============================
    # 2. LOAD DATA
    # ============================
    df = pd.read_excel(input_path)

    required_cols = ["processed_opinion", "label_text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Kolom '{col}' tidak ditemukan di data")

    # ============================
    # 3. FILTER DATA VALID
    # ============================
    df = df[required_cols].dropna()

    df["label_text"] = (
        df["label_text"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df = df[df["label_text"].isin(["positive", "negative"])]
    df = df[df["processed_opinion"].astype(str).str.strip() != ""]

    if df.empty:
        raise ValueError("❌ Data kosong setelah preprocessing & filtering")

    # ============================
    # 4. SPLIT X & y
    # ============================
    X = df["processed_opinion"]
    y = df["label_text"]

    # ============================
    # 5. TRAIN - TEST SPLIT (SEBELUM TF-IDF!)
    # ============================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ============================
    # 6. TF-IDF (NO DATA LEAKAGE)
    # ============================
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2
        # stop_words TIDAK DIPAKAI
        # karena sudah dilakukan di preprocessing sebelumnya
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    # ============================
    # 7. SIMPAN INFO TF-IDF (OPSIONAL & RINGAN)
    # ============================
    os.makedirs(output_dir, exist_ok=True)

    tfidf_info = pd.DataFrame({
        "term": tfidf.get_feature_names_out(),
        "idf": tfidf.idf_
    })

    tfidf_info_path = os.path.join(output_dir, "tfidf_terms.xlsx")
    tfidf_info.to_excel(tfidf_info_path, index=False)

    # ============================
    # 8. LOGISTIC REGRESSION
    # ============================
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear"
    )

    model.fit(X_train_tfidf, y_train)

    # ============================
    # 9. EVALUASI
    # ============================
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    # ============================
    # 10. RETURN UNTUK STREAMLIT
    # ============================
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "model": model,
        "tfidf_terms_path": tfidf_info_path
    }

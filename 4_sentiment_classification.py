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
# LOGIC DALAM FUNGSI (Agar aman saat di-import)
# ============================
def run(input_path, output_dir):
    # Cek apakah file input ada
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File input tidak ditemukan di: {input_path}. Pastikan langkah sebelumnya sudah dijalankan.")

    # ============================
    # 1. Load data
    # ============================
    df = pd.read_excel(input_path)

    # ============================
    # 2. Filter data valid
    # ============================
    # Pastikan kolom ada
    if "processed_opinion" not in df.columns or "label_text" not in df.columns:
         raise ValueError("Kolom 'processed_opinion' atau 'label_text' tidak ditemukan di Excel.")

    df = df[["processed_opinion", "label_text"]].dropna()
    df["label_text"] = df["label_text"].str.lower().str.strip()

    df = df[df["label_text"].isin(["positive", "negative"])]
    df = df[df["processed_opinion"].astype(str).str.strip() != ""]

    if df.empty:
        raise ValueError("❌ Tidak ada data valid untuk klasifikasi (Data kosong setelah filter).")

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
    # 5. Simpan TF-IDF (Opsional)
    # ============================
    # Simpan di output_dir yang dinamis
    tfidf_output_path = os.path.join(output_dir, "tfidf_output.xlsx")
    os.makedirs(output_dir, exist_ok=True)
    
    tfidf_df = pd.DataFrame(
        X_tfidf.toarray(),
        columns=tfidf.get_feature_names_out()
    )
    tfidf_df["label_text"] = y.values
    tfidf_df.to_excel(tfidf_output_path, index=False)

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
        class_weight="balanced",   # Class Weight
        solver="liblinear"
    )

    model.fit(X_train, y_train)

    # ============================
    # 8. Evaluasi
    # ============================
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Kembalikan hasil agar bisa ditampilkan di Streamlit/App
    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "model": model,
        "tfidf_path": tfidf_output_path
    }

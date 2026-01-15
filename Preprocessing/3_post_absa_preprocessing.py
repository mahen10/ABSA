# ============================
# FILE: 3_post_absa_preprocessing.py
# Deskripsi:
# Pisahkan Tokenisasi, Stopword Removal, dan Stemming
# ke kolom sendiri (POST-ABSA)
# ============================

import pandas as pd
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# ============================
# NLTK RESOURCE CHECK (AMAN CLOUD)
# ============================
def ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


# ============================
# PREPROCESSING FUNCTION (TIDAK DIUBAH)
# ============================
def preprocess_steps(text, stop_words, stemmer):
    # Bersihkan spasi berlebih
    text = re.sub(r'\s+', ' ', text.strip())

    # Tokenisasi
    tokens = word_tokenize(text.lower())

    # Stopword Removal
    tokens_no_stop = [
        t for t in tokens
        if t.lower() not in stop_words
    ]

    # Stemming
    stemmed = [
        stemmer.stem(t)
        for t in tokens_no_stop
    ]

    return tokens, tokens_no_stop, stemmed


# ============================
# MAIN PIPELINE
# ============================
def run_post_absa_preprocessing():
    ensure_nltk_resources()

    INPUT_PATH = os.path.join("output", "absa_output.xlsx")
    OUTPUT_PATH = os.path.join("output", "absa_processed.xlsx")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_PATH}"
        )

    # Load data hasil ABSA
    df = pd.read_excel(INPUT_PATH, engine="openpyxl")

    # Stopwords & stemmer
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    # Terapkan preprocessing
    df[["token", "no_stopword", "stem"]] = df["opinion_context"].apply(
        lambda x: pd.Series(
            preprocess_steps(str(x), stop_words, stemmer)
        )
    )

    # Kolom final
    df["processed_opinion"] = df["stem"].apply(
        lambda x: " ".join(x)
    )

    # Placeholder label (dipakai step berikutnya)
    df["label_sentimen"] = ""

    # Simpan hasil
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")

    print("✅ Post-ABSA preprocessing selesai")
    print(f"File disimpan di: {OUTPUT_PATH}")
    print(
        df[
            ["original_review", "aspect", "opinion_word",
             "token", "no_stopword", "stem"]
        ].head(5)
    )

    return df

# ============================
# FILE: 3_post_absa_preprocessing.py
# Deskripsi:
# Preprocessing Tahap 2 (PASCA-ABSA)
# - Tokenisasi
# - Stopword Removal
# - Stemming
# Output:
# - processed_opinion (siap TF-IDF)
# ============================

import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ============================
# NLTK RESOURCE CHECK (AMAN)
# ============================
def ensure_nltk_resources():
    resources = [
        ("corpora/stopwords", "stopwords"),
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


# ============================
# GLOBAL OBJECT
# ============================
STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


# ============================
# TEXT PREPROCESS FUNCTION
# ============================
def preprocess_text(text: str):
    if not isinstance(text, str):
        return "", "", ""

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text.strip().lower())

    # Tokenization (regex-safe)
    tokens = re.findall(r"\b\w+\b", text)

    # Stopword removal
    tokens_no_stop = [
        t for t in tokens
        if t not in STOPWORDS and len(t) > 2
    ]

    # Stemming
    stemmed = [STEMMER.stem(t) for t in tokens_no_stop]

    return (
        " ".join(tokens),
        " ".join(tokens_no_stop),
        " ".join(stemmed)
    )


# ============================
# PIPELINE ENTRY POINT
# ============================
def run(input_path: str, output_path: str):
    """
    Dipanggil dari app.py
    """
    ensure_nltk_resources()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")

    df = pd.read_excel(input_path)

    if "opinion_context" not in df.columns:
        raise ValueError("Kolom 'opinion_context' tidak ditemukan")

    tokens, no_stop, stem = [], [], []

    for text in df["opinion_context"]:
        t, ns, st = preprocess_text(str(text))
        tokens.append(t)
        no_stop.append(ns)
        stem.append(st)

    df["token"] = tokens
    df["no_stopword"] = no_stop
    df["stem"] = stem
    df["processed_opinion"] = stem

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)

    return output_path

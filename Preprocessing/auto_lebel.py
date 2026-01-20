# =====================================================
# FILE: auto_lebel.py
# Deskripsi:
# Auto Label Sentimen (Text)
# - UMIGON (priority)
# - VADER (fallback)
# - Word-first → context fallback
# Output:
# - label_text (positive / negative)
# - label_sentimen (1 / -1)
# =====================================================

import pandas as pd
import re
import os

# ===============================
# PATH DEFAULT
# ===============================
UMIGON_PATH = os.path.join("dict", "umigon-lexicon.tsv.txt")
VADER_PATH = os.path.join("dict", "vader_lexicon.txt")



# ===============================
# LOAD VADER LEXICON
# ===============================
def load_vader(path):
    vader = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            word = parts[0].lower().strip()
            try:
                score = float(parts[1])
                vader[word] = score
            except:
                continue

    return vader

# ===============================
# LOAD UMIGON LEXICON
# ===============================
def load_umigon(path):
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1],
        names=["term", "valence"],
        engine="python",
        on_bad_lines="skip"
    )

    df["term"] = df["term"].astype(str).str.lower().str.strip()
    df["valence"] = df["valence"].astype(str).str.lower().str.strip()
    df = df[df["valence"].isin(["positive", "negative"])]

    return dict(zip(df["term"], df["valence"]))




# ===============================
# CONTEXT FALLBACK
# ===============================
def context_fallback(text, umigon, vader):
    tokens = re.findall(r"\b\w+\b", str(text).lower())

   
    # VADER fallback
    for t in tokens:
        if t in vader:
            return vader[t]

            # UMIGON first
    for t in tokens:
        if t in umigon:
            return "positive" if vader[t] > 0 else "negative"

    return None


# ===============================
# FINAL LABEL FUNCTION
# ===============================
def label_sentiment(opinion_word, opinion_context, vader, umigon):

    # split opinion_word (comma / space safe)
    words = [
        w.strip()
        for w in re.split(r"[,\s]+", str(opinion_word).lower())
        if w.strip()
    ]

   

    # 2️⃣ VADER — opinion word
    for w in words:
        if w in vader:
            return vader[w]
    # 1️⃣ UMIGON — opinion word
    for w in words:
        if w in umigon:
            return "positive" if vader[t] > 0 else "negative"

    # 3️⃣ CONTEXT FALLBACK
    return context_fallback(opinion_context, vader, umigon)


# ===============================
# PIPELINE ENTRY POINT
# ===============================
def run(input_path: str, output_path: str):
    """
    Dipanggil dari app.py
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")

    

    if not os.path.exists(VADER_PATH):
        raise FileNotFoundError("VADER lexicon tidak ditemukan")
        
        if not os.path.exists(UMIGON_PATH):
        raise FileNotFoundError("UMIGON lexicon tidak ditemukan")

    vader = load_vader(VADER_PATH)
        umigon = load_umigon(UMIGON_PATH)


    df = pd.read_excel(input_path)

    required_cols = {"opinion_word", "opinion_context"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Kolom opinion_word / opinion_context tidak lengkap")

    df["label_text"] = df.apply(
        lambda row: label_sentiment(
            row["opinion_word"],
            row["opinion_context"],
            umigon,
            vader
        ),
        axis=1
    )

    df["label_sentimen"] = df["label_text"].map({
        "positive": 1,
        "negative": -1
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)

    return output_path

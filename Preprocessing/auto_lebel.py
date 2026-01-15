# =====================================================
# AUTO LABEL SENTIMENT (NUMERIC VERSION)
# - Umigon (priority)
# - VADER lexicon (fallback)
# - Word-first → context-fallback
# - positive = 1, negative = -1, unlabeled = NaN
# =====================================================

import pandas as pd
import re
import numpy as np
import os


# ===============================
# PATH
# ===============================
UMIGON_PATH = os.path.join("dict", "umigon-lexicon.tsv.txt")
VADER_PATH = os.path.join("dict", "vader_lexicon.txt")
DATA_PATH = os.path.join("output", "absa_processed.xlsx")
OUTPUT_PATH = os.path.join("output", "absa_labeled_numeric.xlsx")


# ===============================
# 1. LOAD UMIGON LEXICON
# ===============================
def load_umigon():
    if not os.path.exists(UMIGON_PATH):
        raise FileNotFoundError(f"UMIGON lexicon tidak ditemukan: {UMIGON_PATH}")

    df = pd.read_csv(
        UMIGON_PATH,
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

    umigon_dict = dict(zip(df["term"], df["valence"]))

    print("✅ UMIGON loaded")
    print("Positive words:", sum(df["valence"] == "positive"))
    print("Negative words:", sum(df["valence"] == "negative"))

    return umigon_dict


# ===============================
# 2. LOAD VADER LEXICON
# ===============================
def load_vader():
    if not os.path.exists(VADER_PATH):
        raise FileNotFoundError(f"VADER lexicon tidak ditemukan: {VADER_PATH}")

    vader_dict = {}

    with open(VADER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            word = parts[0].lower().strip()
            try:
                score = float(parts[1])
                vader_dict[word] = score
            except ValueError:
                continue

    print("✅ VADER loaded")
    print("VADER words:", len(vader_dict))

    return vader_dict


# ===============================
# 3. CONTEXT FALLBACK
# ===============================
def context_fallback(context, umigon_dict, vader_dict):
    tokens = re.findall(r"\b\w+\b", str(context).lower())

    # UMIGON first
    for t in tokens:
        if t in umigon_dict:
            return umigon_dict[t]

    # VADER fallback
    for t in tokens:
        if t in vader_dict:
            return "positive" if vader_dict[t] > 0 else "negative"

    return None


# ===============================
# 4. FINAL LABEL FUNCTION
# ===============================
def label_sentiment(opinion_word, opinion_context, umigon_dict, vader_dict):

    # STEP 0 — split opinion_word
    words = [
        w.strip()
        for w in re.split(r"[,\s]+", str(opinion_word).lower())
        if w.strip()
    ]

    # STEP 1 — OPINION WORD (UMIGON)
    for w in words:
        if w in umigon_dict:
            return umigon_dict[w]

    # STEP 2 — OPINION WORD (VADER)
    for w in words:
        if w in vader_dict:
            return "positive" if vader_dict[w] > 0 else "negative"

    # STEP 3 — CONTEXT FALLBACK
    return context_fallback(opinion_context, umigon_dict, vader_dict)


# ===============================
# 5. MAIN PIPELINE
# ===============================
def run_auto_label():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    umigon_dict = load_umigon()
    vader_dict = load_vader()

    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    print("✅ Dataset loaded:", df.shape)

    # Apply labeling
    df["label_text"] = df.apply(
        lambda row: label_sentiment(
            row["opinion_word"],
            row["opinion_context"],
            umigon_dict,
            vader_dict
        ),
        axis=1
    )

    # Convert to numeric
    df["label_sentimen"] = df["label_text"].map({
        "positive": 1,
        "negative": -1
    })

    print("\n📊 Label distribution (text):")
    print(df["label_text"].value_counts(dropna=False))

    print("\n📊 Label distribution (numeric):")
    print(df["label_sentimen"].value_counts(dropna=False))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")

    print("\n📁 File disimpan di:", OUTPUT_PATH)

    return df

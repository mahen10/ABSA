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

# ===============================
# PATH
# ===============================

UMIGON_PATH = "dict/umigon-lexicon.tsv.txt"
VADER_PATH = "dict/vader_lexicon.txt"
DATA_PATH = "output/absa_processed.xlsx"
OUTPUT_PATH = "output/absa_labeled_numeric.xlsx"

# ===============================
# 1. LOAD UMIGON LEXICON
# ===============================

umigon_df = pd.read_csv(
    UMIGON_PATH,
    sep="\t",
    header=None,
    usecols=[0, 1],
    names=["term", "valence"],
    engine="python",
    on_bad_lines="skip"
)

umigon_df["term"] = umigon_df["term"].astype(str).str.lower().str.strip()
umigon_df["valence"] = umigon_df["valence"].astype(str).str.lower().str.strip()
umigon_df = umigon_df[umigon_df["valence"].isin(["positive", "negative"])]

umigon_dict = dict(zip(umigon_df["term"], umigon_df["valence"]))

positive_words = set(umigon_df[umigon_df["valence"] == "positive"]["term"])
negative_words = set(umigon_df[umigon_df["valence"] == "negative"]["term"])

print("✅ UMIGON loaded")
print("Positive words:", len(positive_words))
print("Negative words:", len(negative_words))

# ===============================
# 2. LOAD VADER LEXICON
# ===============================

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
        except:
            continue

print("✅ VADER loaded")
print("VADER words:", len(vader_dict))

# ===============================
# 3. CONTEXT FALLBACK (UMIGON → VADER)
# ===============================

def context_fallback(context):
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

def label_sentiment(opinion_word, opinion_context, original_review=None):

    # =========================
    # STEP 0 — split opinion_word
    # =========================
    words = [
        w.strip()
        for w in re.split(r"[,\s]+", str(opinion_word).lower())
        if w.strip()
    ]

    # =========================
    # STEP 1 — OPINION WORD (UMIGON)
    # =========================
    for w in words:
        if w in umigon_dict:
            return umigon_dict[w]

    # =========================
    # STEP 2 — OPINION WORD (VADER)
    # =========================
    for w in words:
        if w in vader_dict:
            return "positive" if vader_dict[w] > 0 else "negative"

    # =========================
    # STEP 3 — CONTEXT FALLBACK
    # =========================
    return context_fallback(opinion_context)

# ===============================
# 5. LOAD DATA
# ===============================

def main():
    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    # proses labeling (TIDAK DIUBAH)
    df.to_excel(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()


# ===============================
# 6. APPLY LABELING
# ===============================

df["label_text"] = df.apply(
    lambda row: label_sentiment(
        row["opinion_word"],
        row["opinion_context"],
        row.get("original_review", None)
    ),
    axis=1
)

# ===============================
# 7. CONVERT TO NUMERIC
# ===============================

df["label_sentimen"] = df["label_text"].map({
    "positive": 1,
    "negative": -1
})

print("\n📊 Label distribution (text):")
print(df["label_text"].value_counts(dropna=False))

print("\n📊 Label distribution (numeric):")
print(df["label_sentimen"].value_counts(dropna=False))

# ===============================
# 8. SAVE OUTPUT
# ===============================

df.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")
print("\n📁 File disimpan di:", OUTPUT_PATH)

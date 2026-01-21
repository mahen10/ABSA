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

UMIGON_PATH = os.path.join("dict", "umigon-lexicon.tsv.txt")
VADER_PATH = os.path.join("dict", "vader_lexicon.txt")

def load_umigon(path):
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1],
                     names=["term", "valence"], engine="python", on_bad_lines="skip")
    df["term"] = df["term"].astype(str).str.lower().str.strip()
    df["valence"] = df["valence"].astype(str).str.lower().str.strip()
    df = df[df["valence"].isin(["positive", "negative"])]
    return dict(zip(df["term"], df["valence"]))

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
                vader[word] = float(parts[1])
            except:
                continue
    return vader

# ✅ PERBAIKAN 1: Negation Handling
NEGATIONS = {"not", "no", "never", "neither", "nobody", "nothing", "nowhere",
             "none", "hardly", "scarcely", "barely", "doesn't", "isn't", "wasn't",
             "shouldn't", "wouldn't", "couldn't", "won't", "can't", "don't"}

def has_negation_before(tokens, index, window=3):
    """Check if there's negation within 'window' words before target"""
    start = max(0, index - window)
    return any(tokens[i] in NEGATIONS for i in range(start, index))

# ✅ PERBAIKAN 2: Score-Based Context Analysis
def context_fallback_advanced(text, umigon, vader):
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    
    pos_score = 0
    neg_score = 0
    
    for i, token in enumerate(tokens):
        is_negated = has_negation_before(tokens, i)
        
        # UMIGON scoring
        if token in umigon:
            if umigon[token] == "positive":
                if is_negated:
                    neg_score += 1  # Flip polarity
                else:
                    pos_score += 1
            else:  # negative
                if is_negated:
                    pos_score += 1  # Flip polarity
                else:
                    neg_score += 1
        
        # VADER scoring (fallback)
        elif token in vader:
            score = vader[token]
            if is_negated:
                score = -score  # Flip polarity
            
            if score > 0:
                pos_score += abs(score)
            else:
                neg_score += abs(score)
    
    # Return dominant sentiment
    if pos_score == neg_score == 0:
        return None
    
    return "positive" if pos_score > neg_score else "negative"

# ✅ PERBAIKAN 3: Hybrid Label Function
def label_sentiment(opinion_word, opinion_context, umigon, vader):
    words = [w.strip() for w in re.split(r"[,\s]+", str(opinion_word).lower()) if w.strip()]
    
    # 1️⃣ Cek opinion word dengan negation context
    context_tokens = re.findall(r"\b\w+\b", str(opinion_context).lower())
    
    for i, token in enumerate(context_tokens):
        if token in words:
            # Check if negated in context
            is_negated = has_negation_before(context_tokens, i)
            
            if token in umigon:
                label = umigon[token]
                return "negative" if (is_negated and label == "positive") or \
                                   (not is_negated and label == "negative") else "positive"
            
            if token in vader:
                score = vader[token]
                if is_negated:
                    score = -score
                return "positive" if score > 0 else "negative"
    
    # 2️⃣ Full context analysis (fallback)
    return context_fallback_advanced(opinion_context, umigon, vader)

def run(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")
    if not os.path.exists(UMIGON_PATH):
        raise FileNotFoundError("UMIGON lexicon tidak ditemukan")
    if not os.path.exists(VADER_PATH):
        raise FileNotFoundError("VADER lexicon tidak ditemukan")
    
    umigon = load_umigon(UMIGON_PATH)
    vader = load_vader(VADER_PATH)
    df = pd.read_excel(input_path)
    
    required_cols = {"opinion_word", "opinion_context"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Kolom opinion_word / opinion_context tidak lengkap")
    
    df["label_text"] = df.apply(
        lambda row: label_sentiment(row["opinion_word"], row["opinion_context"], umigon, vader),
        axis=1
    )
    
    df["label_sentimen"] = df["label_text"].map({"positive": 1, "negative": -1})
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    
    return output_path

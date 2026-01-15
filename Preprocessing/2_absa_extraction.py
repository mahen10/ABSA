# =====================================================
# 2_absa_extraction.py (STREAMLIT SAFE)
# =====================================================

import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

# =====================================================
# NLTK INIT (WAJIB)
# =====================================================
def ensure_nltk():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

# =====================================================
# KONFIGURASI
# =====================================================
STOPWORDS = set(stopwords.words("english"))

ASPECTS = {
    "graphics": ["graphics", "visual", "ui", "art"],
    "gameplay": ["gameplay", "control", "mechanic"],
    "story": ["story", "plot", "narrative"],
    "performance": ["performance", "lag", "bug", "fps", "crash"],
    "music": ["music", "sound", "audio", "ost"],
}

# =====================================================
# CORE ABSA
# =====================================================
def extract_aspect_opinion(text, window=4):
    results = []

    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)

    for i, (word, _) in enumerate(tagged):
        for aspect, keys in ASPECTS.items():
            if word in keys:
                ctx = tagged[max(0, i-window): i+window+1]
                opinions = [
                    w for w, t in ctx
                    if t.startswith("JJ") and w not in STOPWORDS
                ]
                if opinions:
                    results.append({
                        "aspect": aspect,
                        "opinion": ", ".join(opinions),
                        "context": " ".join(w for w, _ in ctx)
                    })
    return results

# =====================================================
# ENTRY POINT (SATU-SATUNYA YANG BOLEH JALAN)
# =====================================================
def run(input_path, output_path):
    ensure_nltk()

    df = pd.read_excel(input_path)
    rows = []

    for _, row in df.iterrows():
        text = str(row["cleaned_review"])
        pairs = extract_aspect_opinion(text)

        for p in pairs:
            rows.append({
                "review": row["review"],
                "cleaned_review": text,
                "aspect": p["aspect"],
                "opinion": p["opinion"],
                "context": p["context"],
            })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_excel(output_path, index=False)

    return out_df

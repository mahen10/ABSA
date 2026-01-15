# ============================
# FILE: 2_absa_extraction.py
# Deskripsi:
# ABSA Rule-Based (FINAL)
# - Regex Tokenization (Streamlit-safe)
# - POS Tagging (Adjective priority)
# - Clause Boundary handling
# - No pronoun leakage
# ============================

import pandas as pd
import os
import re
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords

# ============================
# NLTK RESOURCE CHECK (AMAN STREAMLIT)
# ============================
def ensure_nltk_resources():
    resources = [
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/stopwords", "stopwords")
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


# ============================
# GLOBAL CONSTANTS
# ============================
STOPWORDS = set(stopwords.words("english"))
PRONOUN_BLOCKLIST = {"i", "we", "you", "they", "he", "she", "it"}

CLAUSE_BREAKERS = {
    "but", "however", "although", "though", "yet", "ofc"
}

EVAL_VERBS = {
    "suck", "sucks", "hate", "love",
    "recommend", "avoid", "enjoy",
    "worth", "refund"
}

# ============================
# ASPECT DICTIONARY (FINAL)
# ============================
aspect_keywords = {
    "graphics": [
        "graphics", "graphic", "visual", "visuals", "ui", "grafis",
        "art", "artstyle", "look", "resolution", "texture", "animation"
    ],

    "gameplay": [
        "gameplay", "control", "controls", "mechanic", "mechanics",
        "combat", "movement", "interact", "jump", "shoot", "run",
        "action", "fun", "challenging", "responsive",
        "attack", "defend", "transaction", "transactions",
        "quest", "quests"
    ],

    "story": [
        "story", "plot", "narrative", "lore", "writing", "dialogue",
        "ending", "cutscene", "quest", "mission", "twist",
        "character", "development", "script", "storyline"
    ],

    "performance": [
        "performance", "lag", "bug", "fps", "crash", "glitch",
        "smooth", "loading", "freeze", "stutter", "frame",
        "drop", "optimization", "hang", "delay", "disconnect",
        "rate", "memory", "rendering",
        "execution", "garbage", "collection"
    ],

    "music": [
        "music", "sound", "audio", "sfx", "voice", "soundtrack",
        "ost", "noise", "volume", "melody",
        "instrumental", "harmony", "song"
    ]
}


# ============================
# VALIDASI KATA OPINI
# ============================
def valid_word(word: str) -> bool:
    return (
        word.isalpha()
        and word not in STOPWORDS
        and word not in PRONOUN_BLOCKLIST
        and len(word) > 2
    )


# ============================
# CORE ABSA FUNCTION (FINAL)
# ============================
def extract_aspect_opinion(text: str):
    results = []

    # 🔥 REGEX TOKENIZATION (BUKAN word_tokenize)
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    tagged = pos_tag(tokens)

    # Global adjective (fallback TERKONTROL)
    global_adjs = [
        w for w, t in tagged
        if t.startswith("JJ") and valid_word(w)
    ]

    used_aspects = set()

    for i, (word, tag) in enumerate(tagged):
        for aspect, keywords in aspect_keywords.items():
            if word in keywords and aspect not in used_aspects:
                used_aspects.add(aspect)

                # ============================
                # Context Window + Clause Break
                # ============================
                raw_window = tagged[max(0, i - 4): min(len(tagged), i + 6)]
                window = []

                for w, t in raw_window:
                    if w in CLAUSE_BREAKERS:
                        break
                    window.append((w, t))

                # ============================
                # Extract Opinion
                # ============================
                local_adjs = [
                    w for w, t in window
                    if t.startswith("JJ") and valid_word(w)
                ]

                local_eval_verbs = [
                    w for w, _ in window
                    if w in EVAL_VERBS
                ]

                # ============================
                # FINAL DECISION LOGIC
                # ============================
                if local_adjs:
                    opinion = ", ".join(local_adjs)
                    context = " ".join(w for w, _ in window)

                elif local_eval_verbs:
                    opinion = ", ".join(local_eval_verbs)
                    context = " ".join(w for w, _ in window)

                elif len(global_adjs) == 1:
                    opinion = global_adjs[0]
                    context = global_adjs[0]

                else:
                    continue

                results.append({
                    "aspect": aspect,
                    "opinion_word": opinion,
                    "opinion_context": context
                })

    return results


# ============================
# PIPELINE ENTRY POINT (WAJIB)
# ============================
def run(input_path: str, output_path: str):
    """
    Dipanggil oleh app.py
    """
    ensure_nltk_resources()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")

    df = pd.read_excel(input_path)
    final_rows = []

    for _, row in df.iterrows():
        text = str(row.get("cleaned_review", ""))
        matches = extract_aspect_opinion(text)

        for m in matches:
            final_rows.append({
                "original_review": row.get("review", ""),
                "cleaned_review": text,
                "aspect": m["aspect"],
                "opinion_word": m["opinion_word"],
                "opinion_context": m["opinion_context"]
            })

    output_df = pd.DataFrame(final_rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_excel(output_path, index=False)

    return output_path

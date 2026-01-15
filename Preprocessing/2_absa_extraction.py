# ============================
# FILE: 2_absa_extraction.py
# ============================
# ABSA Rule-Based (FINAL - STREAMLIT SAFE)
# ============================

import pandas as pd
import os
import re
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords

# ============================
# NLTK RESOURCE CHECK (WAJIB DI FUNGSI)
# ============================
def ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

# ============================
# ASPECT DICTIONARY (TIDAK DIUBAH)
# ============================
aspect_keywords = {
    "graphics": [
        "graphics", "graphic", "visual", "visuals", "ui",
        "art", "artstyle", "look", "resolution", "texture", "animation"
    ],
    "gameplay": [
        "gameplay", "control", "controls", "mechanic", "mechanics",
        "combat", "movement", "action", "fun", "responsive",
        "attack", "defend", "quest", "quests"
    ],
    "story": [
        "story", "plot", "narrative", "lore", "writing", "dialogue",
        "ending", "cutscene", "mission", "character"
    ],
    "performance": [
        "performance", "lag", "bug", "fps", "crash", "glitch",
        "loading", "freeze", "stutter", "optimization"
    ],
    "music": [
        "music", "sound", "audio", "sfx", "voice", "soundtrack",
        "ost", "melody"
    ],
}

CLAUSE_BREAKERS = {"but", "however", "although", "though", "yet", "ofc"}
EVAL_VERBS = {"suck", "sucks", "hate", "love", "recommend", "avoid", "enjoy", "worth", "refund"}
PRONOUN_BLOCKLIST = {"i", "we", "you", "they", "he", "she", "it"}

# ============================
# VALIDASI OPINI
# ============================
def valid_word(word, stopwords_set):
    return (
        word.isalpha()
        and word not in stopwords_set
        and word not in PRONOUN_BLOCKLIST
        and len(word) > 2
    )

# ============================
# CORE ABSA (LOGIKA TETAP)
# ============================
def extract_aspect_opinion(text, stopwords_set):
    results = []

    tokens = re.findall(r"\b\w+\b", text.lower())
    tagged = pos_tag(tokens)

    global_adjs = [
        w for w, t in tagged
        if t.startswith("JJ") and valid_word(w, stopwords_set)
    ]

    used_aspects = set()

    for i, (word, _) in enumerate(tagged):
        for aspect, keywords in aspect_keywords.items():
            if word in keywords and aspect not in used_aspects:
                used_aspects.add(aspect)

                raw_window = tagged[max(0, i-4): i+6]
                window = []
                for w, t in raw_window:
                    if w in CLAUSE_BREAKERS:
                        break
                    window.append((w, t))

                local_adjs = [
                    w for w, t in window
                    if t.startswith("JJ") and valid_word(w, stopwords_set)
                ]

                local_eval = [w for w, _ in window if w in EVAL_VERBS]

                if local_adjs:
                    opinion = ", ".join(local_adjs)
                    context = " ".join(w for w, _ in window)
                elif local_eval:
                    opinion = ", ".join(local_eval)
                    context = " ".join(w for w, _ in window)
                elif len(global_adjs) == 1:
                    opinion = global_adjs[0]
                    context = global_adjs[0]
                else:
                    continue

                results.append({
                    "aspect": aspect,
                    "opinion_word": opinion,
                    "opinion_context": context,
                })

    return results

# ============================
# ENTRY POINT (WAJIB UNTUK APP)
# ============================
def run(input_path, output_dir):
    ensure_nltk_resources()
    stopwords_set = set(stopwords.words("english"))

    df = pd.read_excel(input_path)
    rows = []

    for _, row in df.iterrows():
        text = str(row["cleaned_review"])
        matches = extract_aspect_opinion(text, stopwords_set)
        for m in matches:
            rows.append({
                "original_review": row["review"],
                "cleaned_review": text,
                **m
            })

    out_df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "absa_output.xlsx")
    out_df.to_excel(output_path, index=False)

    return output_path

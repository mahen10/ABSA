# ============================
# FILE: 2_absa_extraction.py
# ============================
# ABSA Rule-Based
# - POS Tagging (Adjective priority)
# - Clause Boundary (but, however, etc)
# - No pronoun leakage
# - Aman untuk review naratif panjang
# ============================

import pandas as pd
import os
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords


# ============================
# NLTK RESOURCE CHECK (AMAN CLOUD)
# ============================
def ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/stopwords", "stopwords")
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


# ============================
# CONSTANTS
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
# ASPECT DICTIONARY
# ============================
aspect_keywords = {
    'graphics': [
        'graphics', 'graphic', 'visual', 'visuals', 'ui', 'grafis',
        'art', 'artstyle', 'look', 'resolution', 'texture', 'animation'
    ],

    'gameplay': [
        'gameplay', 'control', 'controls', 'mechanic', 'mechanics',
        'combat', 'movement', 'interact', 'jump', 'shoot', 'run',
        'action', 'fun', 'challenging', 'responsive',
        'attack', 'defend', 'transaction', 'transactions',
        'quest', 'quests'
    ],

    'story': [
        'story', 'plot', 'narrative', 'lore', 'writing', 'dialogue',
        'ending', 'cutscene', 'quest', 'mission', 'twist',
        'character', 'development', 'script', 'storyline'
    ],

    'performance': [
        'performance', 'lag', 'bug', 'fps', 'crash', 'glitch',
        'smooth', 'loading', 'freeze', 'stutter', 'frame',
        'drop', 'optimization', 'hang', 'delay', 'disconnect',
        'rate', 'memory', 'rendering',
        'execution', 'garbage', 'collection'
    ],

    'music': [
        'music', 'sound', 'audio', 'sfx', 'voice', 'soundtrack',
        'ost', 'noise', 'volume', 'melody',
        'instrumental', 'harmony', 'song'
    ]
}


# ============================
# VALIDASI OPINI
# ============================
def valid_word(word: str) -> bool:
    return (
        word.isalpha()
        and word not in STOPWORDS
        and word not in PRONOUN_BLOCKLIST
        and len(word) > 2
    )


# ============================
# ABSA CORE FUNCTION (TIDAK DIUBAH)
# ============================
def extract_aspect_opinion(text: str):
    results = []

    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)

    global_adjs = [
        w for w, t in tagged
        if t.startswith("JJ") and valid_word(w)
    ]

    used_aspects = set()

    for i, (word, tag) in enumerate(tagged):
        for aspect, keywords in aspect_keywords.items():
            if word in keywords and aspect not in used_aspects:
                used_aspects.add(aspect)

                raw_window = tagged[max(0, i-4):min(len(tagged), i+6)]
                window = []

                for w, t in raw_window:
                    if w in CLAUSE_BREAKERS:
                        break
                    window.append((w, t))

                local_adjs = [
                    w for w, t in window
                    if t.startswith("JJ") and valid_word(w)
                ]

                local_eval_verbs = [
                    w for w, t in window
                    if w in EVAL_VERBS
                ]

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
# MAIN PIPELINE
# ============================
def run_absa_extraction():
    ensure_nltk_resources()

    INPUT_PATH = os.path.join("Output", "cleaned_reviews.xlsx")
    OUTPUT_PATH = os.path.join("output", "absa_output.xlsx")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"File tidak ditemukan: {INPUT_PATH}")

    df = pd.read_excel(INPUT_PATH)

    final_rows = []

    for _, row in df.iterrows():
        text = str(row["cleaned_review"])
        matches = extract_aspect_opinion(text)

        for m in matches:
            final_rows.append({
                "original_review": row["review"],
                "cleaned_review": text,
                "aspect": m["aspect"],
                "opinion_word": m["opinion_word"],
                "opinion_context": m["opinion_context"]
            })

    output_df = pd.DataFrame(final_rows)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_df.to_excel(OUTPUT_PATH, index=False)

    print("✅ ABSA FINAL SELESAI")
    print(output_df.head(10))

    return output_df

# =====================================================
# FILE: 2_absa_extraction.py
# =====================================================

import pandas as pd
import os
import nltk
import re
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# =====================================================
# NLTK SAFE INIT
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
# CONSTANTS
# =====================================================
STOPWORDS = set(stopwords.words("english"))
PRONOUN_BLOCKLIST = {"i", "we", "you", "they", "he", "she", "it"}

CLAUSE_BREAKERS = {"but", "however", "although", "though", "yet"}

EVAL_VERBS = {
    "love", "hate", "recommend", "avoid", "enjoy",
    "worth", "refund", "suck", "sucks"
}

ASPECT_KEYWORDS = {
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
        'rate', 'memory',  'rendering',
        'execution', 'garbage', 'collection'
    ],

    'music': [
        'music', 'sound', 'audio', 'sfx', 'voice', 'soundtrack',
        'ost', 'noise', 'volume', 'melody',
        'instrumental', 'harmony', 'song'
    ]
}


# =====================================================
# UTIL
# =====================================================
def valid_word(w):
    return (
        w.isalpha()
        and w not in STOPWORDS
        and w not in PRONOUN_BLOCKLIST
        and len(w) > 2
    )


# =====================================================
# ABSA CORE
# =====================================================
def extract_aspect_opinion(text):
    results = []

    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)

    global_adjs = [
        w for w, t in tagged if t.startswith("JJ") and valid_word(w)
    ]

    used_aspects = set()

    for i, (word, _) in enumerate(tagged):
        for aspect, keys in ASPECT_KEYWORDS.items():
            if word in keys and aspect not in used_aspects:
                used_aspects.add(aspect)

                window = tagged[max(0, i-4): i+6]

                filtered = []
                for w, t in window:
                    if w in CLAUSE_BREAKERS:
                        break
                    filtered.append((w, t))

                local_adj = [
                    w for w, t in filtered
                    if t.startswith("JJ") and valid_word(w)
                ]

                local_eval = [w for w, _ in filtered if w in EVAL_VERBS]

                if local_adj:
                    opinion = ", ".join(local_adj)
                elif local_eval:
                    opinion = ", ".join(local_eval)
                elif len(global_adjs) == 1:
                    opinion = global_adjs[0]
                else:
                    continue

                results.append({
                    "aspect": aspect,
                    "opinion_word": opinion,
                    "opinion_context": " ".join(w for w, _ in filtered)
                })

    return results


# =====================================================
# PIPELINE ENTRY POINT (WAJIB)
# =====================================================
def run(input_path, output_path):
    ensure_nltk()

    df = pd.read_excel(input_path)
    rows = []

    for _, r in df.iterrows():
        text = str(r["cleaned_review"])
        matches = extract_aspect_opinion(text)

        for m in matches:
            rows.append({
                "original_review": r["review"],
                "cleaned_review": text,
                "aspect": m["aspect"],
                "opinion_word": m["opinion_word"],
                "opinion_context": m["opinion_context"]
            })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_excel(output_path, index=False)

    return out_df

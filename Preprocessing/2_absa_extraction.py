# ============================
# FILE: 03_absa_extraction_excel_FINAL.py
# ============================
# ABSA Rule-Based
# - POS Tagging (Adjective priority)
# - Clause Boundary (but, however, etc)
# - No pronoun leakage
# - Aman untuk review naratif panjang
# ============================

import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
import pandas as pd
import os
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords

# ============================
# Download resource (sekali saja)
# ============================
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
PRONOUN_BLOCKLIST = {"i", "we", "you", "they", "he", "she", "it"}

# 🔴 CLAUSE BREAKERS (WAJIB)
CLAUSE_BREAKERS = {
    "but", "however", "although", "though", "yet","ofc"
}

# 🔴 VERB EVALUATIF TERKONTROL
EVAL_VERBS = {
    "suck", "sucks", "hate", "love",
    "recommend", "avoid", "enjoy",
    "worth", "refund"
}

# ============================
# 1. Load Data
# ============================
INPUT_PATH = os.path.join("output", "cleaned_reviews.xlsx")
df = pd.read_excel(INPUT_PATH)

# ============================
# 2. Kamus Aspek
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
        'rate', 'memory',  'rendering',
        'execution', 'garbage', 'collection'
    ],

    'music': [
        'music', 'sound', 'audio', 'sfx', 'voice', 'soundtrack',
        'ost', 'noise', 'volume', 'melody',
        'instrumental', 'harmony', 'song'
    ]
}


ALL_ASPECT_WORDS = set(
    kw for kws in aspect_keywords.values() for kw in kws
)

# ============================
# 3. Validasi Kata Opini
# ============================
def valid_word(word: str) -> bool:
    return (
        word.isalpha()
        and word not in STOPWORDS
        and word not in PRONOUN_BLOCKLIST
        and len(word) > 2
    )

# ============================
# 4. Fungsi ABSA FINAL
# ============================
def extract_aspect_opinion(text: str):
    results = []

    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)

    # 🔹 Global adjective (fallback TERKONTROL)
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
                # Window + Clause Boundary
                # ============================
                raw_window = tagged[max(0, i-4):min(len(tagged), i+6)]
                window = []

                for w, t in raw_window:
                    if w in CLAUSE_BREAKERS:
                        break
                    window.append((w, t))

                # ============================
                # Ekstraksi Opini
                # ============================
                local_adjs = [
                    w for w, t in window
                    if t.startswith("JJ") and valid_word(w)
                ]

                local_eval_verbs = [
                    w for w, t in window
                    if w in EVAL_VERBS
                ]

                # ============================
                # LOGIKA FINAL (URUTAN KRITIS)
                # ============================

                # 1️⃣ PRIORITAS: adjective lokal
                if local_adjs:
                    opinion = ", ".join(local_adjs)
                    context = " ".join(w for w, _ in window)

                # 2️⃣ PRIORITAS: evaluative verb lokal
                elif local_eval_verbs:
                    opinion = ", ".join(local_eval_verbs)
                    context = " ".join(w for w, _ in window)

                # 3️⃣ FALLBACK GLOBAL (HANYA JIKA SATU)
                elif len(global_adjs) == 1:
                    opinion = global_adjs[0]
                    context = global_adjs[0]

                # 4️⃣ TIDAK ADA OPINI → SKIP
                else:
                    continue

                results.append({
                    "aspect": aspect,
                    "opinion_word": opinion,
                    "opinion_context": context
                })

    return results

# ============================
# 5. Proses Seluruh Dataset
# ============================
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

# ============================
# 6. Simpan ke Excel
# ============================
output_df = pd.DataFrame(final_rows)
output_PATH = os.path.join("output", "absa_output.xlsx")
output_df.to_excel(output_PATH, index=False)

print("✅ ABSA FINAL SELESAI")
print(output_df.head(10))

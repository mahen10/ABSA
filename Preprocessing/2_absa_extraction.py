# =====================================================
# FILE: Preprocessing/2_absa_extraction.py
# =====================================================

import pandas as pd
import os
import nltk
import re
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# =====================================================
# NLTK SAFE INIT (Ditaruh di atas dan dipanggil segera)
# =====================================================
def ensure_nltk():
    """
    Memastikan semua resource NLTK yang dibutuhkan tersedia.
    Menambahkan "averaged_perceptron_tagger_eng" untuk support NLTK terbaru.
    """
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("help/taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"), # <-- PERBAIKAN DI SINI
        ("corpora/stopwords", "stopwords"),
    ]
    
    print("--- Checking NLTK Resources ---")
    for path, name in resources:
        try:
            # Coba cari resource (path mungkin perlu penyesuaian untuk _eng, jadi kita pakai try-except luas)
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading missing resource: {name}...")
            try:
                nltk.download(name, quiet=True)
            except Exception as e:
                print(f"Gagal download {name}: {e}")

# --- Panggil fungsi ini SEKARANG, sebelum STOPWORDS didefinisikan ---
ensure_nltk() 

# =====================================================
# CONSTANTS
# =====================================================
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    # Fallback darurat jika download gagal
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

PRONOUN_BLOCKLIST = {"i", "we", "you", "they", "he", "she", "it"}

CLAUSE_BREAKERS = {"but", "however", "although", "though", "yet"}

EVAL_VERBS = {
    "love", "hate", "recommend", "avoid", "enjoy",
    "worth", "refund", "suck", "sucks"
}

ASPECT_KEYWORDS = {
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
        "rate", "memory",  "rendering",
        "execution", "garbage", "collection"
    ],

    "music": [
        "music", "sound", "audio", "sfx", "voice", "soundtrack",
        "ost", "noise", "volume", "melody",
        "instrumental", "harmony", "song"
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

    # Pastikan text string
    if not isinstance(text, str):
        text = str(text) if pd.notna(text) else ""

    if not text.strip():
        return []

    try:
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
    except LookupError as e:
        # Emergency handling jika resource masih missing saat runtime
        print(f"NLTK Error during extraction: {e}")
        return []

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
    print(f"--- Memulai Step 2: ABSA Extraction ---")
    
    # Cek apakah file ada sebelum dibaca
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")

    df = pd.read_excel(input_path)
    rows = []
    
    # Cek kolom input
    if "cleaned_review" not in df.columns:
        print("Warning: Kolom "cleaned_review" tidak ditemukan. Mencoba menggunakan kolom lain.")
        # Fallback logic
        possible_cols = ["cleaned_review", "review", "content", "text"]
        target_col = next((c for c in possible_cols if c in df.columns), None)
        if not target_col:
             raise ValueError("Tidak ada kolom teks yang valid untuk diproses.")
    else:
        target_col = "cleaned_review"

    print(f"Menggunakan kolom: {target_col}")

    for _, r in df.iterrows():
        raw_text = r.get(target_col, "") 
        text = str(raw_text) if pd.notna(raw_text) else ""
        
        matches = extract_aspect_opinion(text)

        # Jika tidak ada aspek ditemukan, baris ini dilewati (atau bisa disimpan sebagai None)
        # Di sini kita hanya menyimpan yang ada match-nya agar tabel hasil bersih
        for m in matches:
            rows.append({
                "original_review": r.get("review", text), # Simpan review asli
                "cleaned_review": text,
                "aspect": m["aspect"],
                "opinion_word": m["opinion_word"],
                "opinion_context": m["opinion_context"]
            })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_excel(output_path, index=False)
    
    print(f"Step 2 Selesai. Hasil: {len(out_df)} baris aspek terdeteksi.")
    print(f"Disimpan di: {output_path}")

    return out_df

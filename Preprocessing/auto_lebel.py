# =====================================================
# FILE: Preprocessing/auto_lebel.py
# Deskripsi: Auto Label dengan Sistem Scoring & Gaming Dict
# =====================================================

import pandas as pd
import re
import os

# ===============================
# PATH DEFAULT
# ===============================
UMIGON_PATH = os.path.join("dict", "umigon-lexicon.tsv.txt")
VADER_PATH = os.path.join("dict", "vader_lexicon.txt")

# ===============================
# KAMUS SPESIFIK GAMING (Overrides)
# ===============================
# Kata-kata ini akan mengalahkan kamus umum (Vader/Umigon)
GAMING_DICT = {
    "addictive": "positive",
    "addicting": "positive",
    "masterpiece": "positive",
    "cinema": "positive",
    "kino": "positive",
    "refund": "negative",
    "crash": "negative",
    "unplayable": "negative",
    "bug": "negative",
    "buggy": "negative",
    "trash": "negative",
    "garbage": "negative",
    "solid": "positive",
    "smooth": "positive",
    "clunky": "negative",
    "lag": "negative",
    "stutter": "negative"
}

# ===============================
# LOAD DICTIONARIES
# ===============================
def load_vader(path):
    vader = {}
    if not os.path.exists(path):
        return vader
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            parts = line.split("\t")
            if len(parts) < 2: continue
            try: vader[parts[0].lower().strip()] = float(parts[1])
            except: continue
    return vader

def load_umigon(path):
    if not os.path.exists(path): return {}
    try:
        df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1],
                         names=["term", "valence"], engine="python", on_bad_lines="skip")
        df["term"] = df["term"].astype(str).str.lower().str.strip()
        df["valence"] = df["valence"].astype(str).str.lower().str.strip()
        return dict(zip(df["term"], df["valence"]))
    except: return {}

# ===============================
# CORE LOGIC: SENTIMENT SCORING
# ===============================
def calculate_sentiment_score(text, umigon, vader):
    """
    Menghitung total skor sentimen dari sebuah teks.
    Positive menambah skor, Negative mengurangi skor.
    """
    if not isinstance(text, str): return 0.0
    
    tokens = re.findall(r"\b\w+\b", text.lower())
    score = 0.0
    
    # Modifier sederhana (bisa dikembangkan)
    negation = False
    
    for word in tokens:
        # Reset negation jika ada tanda baca (simple check)
        
        # 1. Cek Gaming Dict (Prioritas Tertinggi)
        if word in GAMING_DICT:
            val = 1.5 if GAMING_DICT[word] == 'positive' else -1.5
            score += val
            continue
            
        # 2. Cek Vader (Skor detail)
        if word in vader:
            score += vader[word]
            continue
            
        # 3. Cek Umigon (Skor biner)
        if word in umigon:
            val = 1.0 if umigon[word] == 'positive' else -1.0
            score += val
            
    return score

# ===============================
# FINAL LABEL FUNCTION
# ===============================
def label_sentiment(opinion_word, opinion_context, umigon, vader):
    # Bersihkan input
    op_word = str(opinion_word).lower() if pd.notna(opinion_word) else ""
    op_context = str(opinion_context).lower() if pd.notna(opinion_context) else ""

    # --- LANGKAH 1: Cek Kata Opini (Sangat Spesifik) ---
    # Jika kata opini ada di Gaming Dict, langsung ambil keputusannya
    if op_word in GAMING_DICT:
        return GAMING_DICT[op_word]

    # --- LANGKAH 2: Hitung Skor Konteks (Voting) ---
    # Kita gabungkan kata opini dan konteks untuk mendapatkan gambaran utuh
    # Memberi bobot lebih pada opinion_word (dikalikan 2)
    full_text = f"{op_word} {op_word} {op_context}" 
    
    total_score = calculate_sentiment_score(full_text, umigon, vader)
    
    # --- LANGKAH 3: Tentukan Label berdasarkan Skor ---
    # Threshold 0.05 untuk menghindari noise
    if total_score >= 0.05:
        return "positive"
    elif total_score <= -0.05:
        return "negative"
    
    # --- FALLBACK: Jika skor 0 (Netral/Tidak Tau) ---
    # Cek Umigon direct match pada kata opini sebagai upaya terakhir
    if op_word in umigon:
        return umigon[op_word]
        
    return "positive" # Default bias ke positif jika bingung

# ===============================
# PIPELINE ENTRY POINT
# ===============================
def run(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")

    # Load dictionaries
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    root_dir = os.path.dirname(base_dir) 
    
    umigon = load_umigon(os.path.join(root_dir, UMIGON_PATH))
    vader = load_vader(os.path.join(root_dir, VADER_PATH))

    # Baca Data
    df = pd.read_excel(input_path)

    # Handle kolom
    target_col = "opinion_word"
    context_col = "opinion_context"
    
    if target_col not in df.columns and "processed_opinion" in df.columns:
        target_col = "processed_opinion" # Fallback

    if context_col not in df.columns:
        df[context_col] = "" # Kosongkan jika tidak ada

    # Proses
    df["label_text"] = df.apply(
        lambda row: label_sentiment(
            row.get(target_col, ""),
            row.get(context_col, ""),
            umigon,
            vader
        ),
        axis=1
    )

    # Map ke angka
    df["label_sentimen"] = df["label_text"].map({"positive": 1, "negative": -1}).fillna(0)

    # Simpan
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)

    return df

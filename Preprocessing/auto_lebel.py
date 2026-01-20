# =====================================================
# FILE: Preprocessing/auto_lebel.py
# Deskripsi: Auto Label (Smart Logic: Gaming Dict + Negation Handling)
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
# 1. KAMUS SPESIFIK GAMING (Wajib untuk Akurasi Tinggi)
# ===============================
GAMING_DICT = {
    # Positif
    "addictive": "positive", "masterpiece": "positive", "cinema": "positive",
    "kino": "positive", "solid": "positive", "smooth": "positive",
    "crisp": "positive", "fun": "positive", "optimized": "positive",
    "immersive": "positive",
    
    # Negatif
    "refund": "negative", "crash": "negative", "unplayable": "negative",
    "bug": "negative", "buggy": "negative", "trash": "negative",
    "garbage": "negative", "clunky": "negative", "lag": "negative",
    "stutter": "negative", "fps drops": "negative", "boring": "negative",
    "repetitive": "negative", "broken": "negative", "woke": "negative"
}

# ===============================
# 2. LOAD DICTIONARIES
# ===============================
def load_vader(path):
    vader = {}
    if not os.path.exists(path): return vader
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
# 3. SMART LOGIC: NEGATION CHECK
# ===============================
def check_negation(word, context_str):
    """
    Mengecek apakah ada kata 'not', 'no', 'never' dsb 
    dalam jarak 3 kata SEBELUM kata target di dalam konteks.
    """
    if pd.isna(context_str) or not context_str:
        return False
    
    context_str = str(context_str).lower()
    word = str(word).lower()
    
    # Cari posisi kata di konteks
    # Kita pakai regex boundary \b agar akurat
    matches = list(re.finditer(r'\b' + re.escape(word) + r'\b', context_str))
    
    if not matches:
        return False
        
    # Ambil match pertama saja untuk simplifikasi
    start_index = matches[0].start()
    
    # Ambil teks sebelum kata tersebut (max 20 karakter mundur)
    preceding_text = context_str[max(0, start_index - 25) : start_index]
    
    # Kata-kata negasi
    negations = ["not", "no", "never", "n't", "hardly", "barely", "lack"]
    
    # Cek apakah ada negasi di teks sebelumnya
    tokens = re.split(r'\s+', preceding_text.strip())
    # Ambil 3 kata terakhir sebelum target
    last_3_tokens = tokens[-3:] 
    
    for t in last_3_tokens:
        if any(neg in t for neg in negations):
            return True
            
    return False

def flip_sentiment(label):
    if label == "positive": return "negative"
    if label == "negative": return "positive"
    return label

# ===============================
# FINAL LABEL FUNCTION
# ===============================
def label_sentiment(opinion_word, opinion_context, umigon, vader):
    op_word = str(opinion_word).lower().strip() if pd.notna(opinion_word) else ""
    # Pecah jika ada koma (misal "smooth, fun")
    words = [w.strip() for w in re.split(r'[,\s]+', op_word) if w.strip()]
    
    final_label = None
    
    # --- LOGIC LOOP ---
    for w in words:
        current_label = None
        
        # 1. Cek Gaming Dict (Priority 1)
        if w in GAMING_DICT:
            current_label = GAMING_DICT[w]
            
        # 2. Cek Umigon (Priority 2)
        elif w in umigon:
            current_label = umigon[w]
            
        # 3. Cek Vader (Priority 3)
        elif w in vader:
            score = vader[w]
            if score >= 0.05: current_label = "positive"
            elif score <= -0.05: current_label = "negative"
            
        # --- JIKA KETEMU LABEL, CEK NEGASI ---
        if current_label:
            # Cek apakah ada kata "not" sebelumnya di konteks
            is_negated = check_negation(w, opinion_context)
            if is_negated:
                current_label = flip_sentiment(current_label)
            
            return current_label # Langsung return (First Match Logic)

    # --- FALLBACK: CONTEXT SCAN ---
    # Jika opinion_word tidak ketemu di kamus manapun, scan konteksnya
    if pd.notna(opinion_context):
        ctx_tokens = re.findall(r"\b\w+\b", str(opinion_context).lower())
        for t in ctx_tokens:
            if t in GAMING_DICT: return GAMING_DICT[t] # Cek gaming dict di konteks
            
    return "positive" # Bias default

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

# =====================================================
# FILE: Preprocessing/auto_lebel.py
# Deskripsi: Auto Label Sentimen (UMIGON + VADER)
# =====================================================

import pandas as pd
import re
import os

# ===============================
# PATH DEFAULT
# ===============================
# Pastikan path ini sesuai dengan struktur folder Anda
UMIGON_PATH = os.path.join("dict", "umigon-lexicon.tsv.txt")
VADER_PATH = os.path.join("dict", "vader_lexicon.txt")

# ===============================
# LOAD VADER LEXICON
# ===============================
def load_vader(path):
    vader = {}
    if not os.path.exists(path):
        print(f"Warning: VADER dictionary not found at {path}")
        return vader

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            word = parts[0].lower().strip()
            try:
                score = float(parts[1])
                vader[word] = score
            except:
                continue
    return vader

# ===============================
# LOAD UMIGON LEXICON
# ===============================
def load_umigon(path):
    if not os.path.exists(path):
        print(f"Warning: UMIGON dictionary not found at {path}")
        return {}

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            usecols=[0, 1],
            names=["term", "valence"],
            engine="python",
            on_bad_lines="skip"
        )
        df["term"] = df["term"].astype(str).str.lower().str.strip()
        df["valence"] = df["valence"].astype(str).str.lower().str.strip()
        df = df[df["valence"].isin(["positive", "negative"])]
        return dict(zip(df["term"], df["valence"]))
    except Exception as e:
        print(f"Error loading Umigon: {e}")
        return {}

# ===============================
# HELPER: NORMALIZE VADER SCORE
# ===============================
def normalize_vader(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return None

# ===============================
# CONTEXT FALLBACK
# ===============================
def context_fallback(text, umigon, vader):
    if not isinstance(text, str):
        return None
        
    tokens = re.findall(r"\b\w+\b", str(text).lower())

    # 1. Cek UMIGON di context (Prioritas Utama)
    for t in tokens:
        if t in umigon:
            return umigon[t]

    # 2. Cek VADER di context (Fallback)
    for t in tokens:
        if t in vader:
            return normalize_vader(vader[t])

    return None

# ===============================
# FINAL LABEL FUNCTION
# ===============================
def label_sentiment(opinion_word, opinion_context, umigon, vader):
    # Bersihkan input
    if pd.isna(opinion_word): opinion_word = ""
    if pd.isna(opinion_context): opinion_context = ""

    # Split kata opini (jika ada koma, misal: "smooth, addictive")
    words = [
        w.strip()
        for w in re.split(r"[,\s]+", str(opinion_word).lower())
        if w.strip()
    ]

    # PRIORITAS 1: Cek Kata Opini di UMIGON
    for w in words:
        if w in umigon:
            return umigon[w]

    # PRIORITAS 2: Cek Kata Opini di VADER
    for w in words:
        if w in vader:
            res = normalize_vader(vader[w])
            if res: return res

    # PRIORITAS 3: Cek Konteks Kalimat (Fallback)
    res_context = context_fallback(opinion_context, umigon, vader)
    if res_context:
        return res_context

    # DEFAULT (Jika tidak ketemu di manapun)
    return "positive" # Bias positif atau bisa diganti 'neutral'

# ===============================
# PIPELINE ENTRY POINT
# ===============================
def run(input_path: str, output_path: str):
    """
    Dipanggil dari app.py
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")

    # Load dictionaries
    # Kita gunakan relative path agar aman
    base_dir = os.path.dirname(os.path.abspath(__file__)) # Folder Preprocessing
    root_dir = os.path.dirname(base_dir) # Folder Root (absa)
    
    # Path absolut ke kamus
    abs_umigon = os.path.join(root_dir, UMIGON_PATH)
    abs_vader = os.path.join(root_dir, VADER_PATH)

    umigon = load_umigon(abs_umigon)
    vader = load_vader(abs_vader)

    # Baca Data
    df = pd.read_excel(input_path)

    # Cek kolom wajib
    required_cols = {"opinion_word", "opinion_context"}
    if not required_cols.issubset(df.columns):
        # Jika kolom opinion_word tidak ada tapi processed_opinion ada, pakai itu
        if "processed_opinion" in df.columns:
            df["opinion_word"] = df["processed_opinion"]
        else:
            raise ValueError("Kolom 'opinion_word' atau 'opinion_context' hilang dari data.")

    # Terapkan Pelabelan
    # Perhatikan urutan argumen: opinion_word, opinion_context, umigon, vader
    df["label_text"] = df.apply(
        lambda row: label_sentiment(
            row.get("opinion_word", ""),
            row.get("opinion_context", ""),
            umigon,
            vader
        ),
        axis=1
    )

    # Konversi ke Angka (Opsional)
    df["label_sentimen"] = df["label_text"].map({
        "positive": 1,
        "negative": -1
    }).fillna(0) # 0 untuk netral/tidak diketahui

    # Simpan
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)

    return df

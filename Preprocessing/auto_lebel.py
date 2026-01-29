# =====================================================
# FILE: auto_lebel.py
# Deskripsi:
# Auto Label Sentimen (Text)
# - UMIGON (priority)
# - VADER (fallback)
# - Word-first → context fallback
# - SARCASM DETECTION (New Feature)
# Output:
# - label_text (positive / negative)
# - label_sentimen (1 / -1)
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
# GAMING SARCASM DICTIONARY
# ===============================
# Frasa dan pola ini digunakan untuk membalik sentimen Positive -> Negative
SARCASM_PHRASES = [
    "refund simulator", "crash simulator", "loading simulator", "walking simulator",
    "slideshow", "unplayable", "garbage optimization", "waste of money",
    "don't buy", "do not buy", "save your money", "mixed feelings",
    "potato pc", "nice ppt", 
    "yeah right", "as if", "oh really", "thanks for nothing",
    "love the lag", "love the crash", "love the bugs", "amazing bug", 
    "great crash", "best crash", "enjoy the lag", "favorite bug",
    "totally playable", "barely playable", "looks like trash", 
    "looks like a joke", "looks broken", "go fix it", "fix your game"
]

# ===============================
# LOAD UMIGON LEXICON
# ===============================
def load_umigon(path):
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


# ===============================
# LOAD VADER LEXICON
# ===============================
def load_vader(path):
    vader = {}

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
# SARCASM CHECKER
# ===============================
def check_sarcasm(text):
    """
    Mengecek apakah text mengandung indikasi sarkasme.
    Input: text (sebaiknya raw sentence dengan tanda baca).
    """
    if pd.isna(text) or str(text).strip() == "":
        return False
        
    text_lower = str(text).lower()
    
    # 1. Cek Frasa Pasti
    for phrase in SARCASM_PHRASES:
        if phrase in text_lower:
            return True
            
    # 2. Cek Pola Regex (Pujian diikuti Masalah / Ironi)
    technical_issues = r"(crash|bug|lag|freeze|glitch|delay|error|broken|trash|unplayable)"
    mock_praises = r"(great|good|awesome|amazing|perfect|brilliant|wonderful|love|best|nice)"
    
    sarcasm_patterns = [
        # Pattern: Pujian + Masalah (Ex: "Great crash")
        rf"{mock_praises}\s+.*{technical_issues}",
        
        # Pattern: 10/10 tapi ironis (Menangani variasi 10/10, 10 10, dll)
        r"10\s*[/:\s]\s*10.*(uninstall|crash|waste|bug|trash|refund)",
        
        # Pattern: Negasi di akhir (Ex: "Good game... NOT")
        r".*\bnot[.!?]*$", 
        
        # Pattern: Tanda universal sarkasme
        r"/s$"
    ]
    
    for pattern in sarcasm_patterns:
        if re.search(pattern, text_lower):
            return True
            
    return False


# ===============================
# HELPER: GET RAW CONTEXT
# ===============================
def get_raw_context(original_review, opinion_word):
    """
    Mencari kalimat ASLI (dengan tanda baca) dari original_review 
    yang mengandung opinion_word.
    """
    if pd.isna(original_review) or pd.isna(opinion_word):
        return ""
    
    # Pecah review panjang menjadi kalimat-kalimat (berdasarkan . ! ?)
    # Regex ini memisahkan kalimat saat ketemu titik/tanda seru/tanya
    sentences = re.split(r'(?<=[.!?])\s+', str(original_review))
    
    target_word = str(opinion_word).lower()
    
    for sent in sentences:
        if target_word in sent.lower():
            return sent  # Kembalikan kalimat mentah
            
    return "" # Jika tidak ketemu, kembalikan kosong


# ===============================
# CONTEXT FALLBACK
# ===============================
def context_fallback(text, umigon, vader):
    tokens = re.findall(r"\b\w+\b", str(text).lower())

    # UMIGON first
    for t in tokens:
        if t in umigon:
            return umigon[t]

    # VADER fallback
    for t in tokens:
        if t in vader:
            return "positive" if vader[t] > 0 else "negative"

    return None


# ===============================
# FINAL LABEL FUNCTION
# ===============================
def label_sentiment(opinion_word, opinion_context, umigon, vader, sarcasm_check_text=None):
    """
    sarcasm_check_text: Text raw untuk pengecekan sarkasme (opsional)
    """

    # --- 1. DETERMINE INITIAL LABEL (EXISTING LOGIC) ---
    label = None

    # split opinion_word (comma / space safe)
    words = [
        w.strip()
        for w in re.split(r"[,\s]+", str(opinion_word).lower())
        if w.strip()
    ]

    # 1️⃣ UMIGON — opinion word
    if not label:
        for w in words:
            if w in umigon:
                label = umigon[w]
                break

    # 2️⃣ VADER — opinion word
    if not label:
        for w in words:
            if w in vader:
                label = "positive" if vader[w] > 0 else "negative"
                break

    # 3️⃣ CONTEXT FALLBACK
    if not label:
        label = context_fallback(opinion_context, umigon, vader)
    
    # Default Safety Net
    if not label:
        return "negative" # Asumsi default jika tidak ketemu apa-apa

    # --- 2. SARCASM CHECK (NEW FEATURE) ---
    # Kita hanya perlu cek sarkasme jika label awalnya POSITIVE.
    # Jika label awal sudah NEGATIVE, sarkasme tidak mengubah apa-apa.
    if label == "positive":
        # Gunakan text raw jika ada, jika tidak pakai opinion_context
        text_to_check = sarcasm_check_text if sarcasm_check_text else opinion_context
        
        if check_sarcasm(text_to_check):
            return "negative" # FLIP KE NEGATIF

    return label


# ===============================
# PIPELINE ENTRY POINT
# ===============================
def run(input_path: str, output_path: str):
    """
    Dipanggil dari app.py
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")

    if not os.path.exists(UMIGON_PATH):
        raise FileNotFoundError("UMIGON lexicon tidak ditemukan")

    if not os.path.exists(VADER_PATH):
        raise FileNotFoundError("VADER lexicon tidak ditemukan")

    umigon = load_umigon(UMIGON_PATH)
    vader = load_vader(VADER_PATH)

    df = pd.read_excel(input_path)

    required_cols = {"opinion_word", "opinion_context"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Kolom opinion_word / opinion_context tidak lengkap")

    # Cek apakah kita punya akses ke review asli (untuk akurasi sarkasme lebih baik)
    has_original_review = "original_review" in df.columns

    df["label_text"] = df.apply(
        lambda row: label_sentiment(
            row["opinion_word"],
            row["opinion_context"],
            umigon,
            vader,
            # Ambil raw context dari original_review jika ada, untuk sarkasme
            sarcasm_check_text=get_raw_context(row["original_review"], row["opinion_word"]) if has_original_review else row["opinion_context"]
        ),
        axis=1
    )

    df["label_sentimen"] = df["label_text"].map({
        "positive": 1,
        "negative": -1
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)

    return output_path

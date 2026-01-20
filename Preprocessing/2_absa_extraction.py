# =====================================================
# FILE: 2_absa_extraction.py
# FIXED: Use Original Text (No Stemming) + Column Name Fix
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
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                print(f"Downloading {name}...")
                nltk.download(name, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {name}: {e}")

ensure_nltk()

# =====================================================
# CONSTANTS
# =====================================================
STOPWORDS = set(stopwords.words("english"))

ASPECT_KEYWORDS = {
     "graphics": [
        "graphics", "graphic", "visual", "visuals", "ui", "gui",
        "art", "artstyle", "resolution", "texture", "animation", 
        "lighting", "shadow", "design", "scenery", "environment"
    ],

    "gameplay": [
        "gameplay", "control", "controls", "mechanic", "mechanics",
        "combat", "movement", "system", "feature", "features",
        "action", "battle", "attack", "defend", "quest", "quests",
        "level", "enemy", "boss", "difficulty"
    ],

    "story": [
        "story", "plot", "narrative", "lore", "writing", "dialogue",
        "ending", "cutscene", "mission", "twist", "script",
        "character", "development", "storyline", "arc", "pacing"
    ],

    "performance": [
        "performance", "fps", "frame", "rate", "optimization", 
        "memory", "rendering", "loading", "server", "connection",
        "ping", "latency", "bug", "glitch", "crash", "freeze" 
    ],

    "music": [
        "music", "sound", "audio", "sfx", "voice", "acting",
        "soundtrack", "ost", "bgm", "volume", "melody", "song",
        "noise", "dubbing"
    ]
}

# Regex untuk memecah kalimat menjadi klausa
CLAUSE_SPLIT_PATTERN = r'[.,!?;]| but | however | although | though | yet | while | whereas | except '

# =====================================================
# UTIL
# =====================================================
def valid_word(w):
    # Kita izinkan kata panjang > 2 huruf
    return (
        w.isalpha() 
        and w not in STOPWORDS 
        and len(w) > 2
    )

def light_clean(text):
    """Membersihkan teks asli tanpa stemming agar enak dibaca"""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Hapus karakter aneh tapi biarkan tanda baca penting untuk pemisah klausa
    text = re.sub(r'[^a-z0-9\s.,!?;]', '', text) 
    return text

# =====================================================
# ABSA CORE
# =====================================================
def extract_aspect_opinion(text):
    if not text:
        return []

    results = []
    
    # Bersihkan ringan (TANPA STEMMING)
    text = light_clean(text)
    
    # Pecah kalimat menjadi segmen (Klausa)
    clauses = re.split(CLAUSE_SPLIT_PATTERN, text)
    
    seen_aspects_in_text = set()

    for clause in clauses:
        clause = clause.strip()
        if len(clause) < 3:
            continue
            
        tokens = word_tokenize(clause)
        tagged = pos_tag(tokens)
        
        # Cari kata sifat (JJ), Verb (VB), Adverb (RB)
        # Karena teks asli, NLTK akan lebih akurat mendeteksi Adjective (misal: "addictive" vs "addict")
        potential_opinions = []
        for w, t in tagged:
            if (t.startswith("JJ") or t.startswith("VB") or t.startswith("RB")) and valid_word(w):
                potential_opinions.append(w)
        
        # Cari aspek
        found_aspect = None
        for w_clause in tokens:
            for aspect, keys in ASPECT_KEYWORDS.items():
                if w_clause in keys:
                    found_aspect = aspect
                    break 
            if found_aspect:
                break
        
        # Simpan jika ada pasangan Aspek + Opini
        if found_aspect and potential_opinions:
            # Opsional: Mencegah duplikat aspek dalam satu review
            # if found_aspect in seen_aspects_in_text: continue
            
            seen_aspects_in_text.add(found_aspect)
            
            # Gabungkan opini jadi string cantik (misal: "smooth, addictive")
            opinion_str = ", ".join(potential_opinions)
            
            results.append({
                "aspect": found_aspect,
                "opinion_word": opinion_str,
                "opinion_context": clause
            })

    return results

# =====================================================
# PIPELINE ENTRY POINT
# =====================================================
def run(input_path, output_path):
    print(f"Processing: {input_path}")
    
    try:
        df = pd.read_excel(input_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    rows = []
    
    # === PERBAIKAN LOGIKA SUMBER DATA ===
    # Kita prioritaskan kolom 'review' (teks asli) daripada 'cleaned_review'.
    # Kenapa? Karena 'cleaned_review' dari Step 1 mungkin sudah kena Stemming (addictive -> addict).
    # Kita ingin hasil ekstraksi tetap utuh ("addictive").
    
    source_col = "review" # Default ke review asli
    if "review" not in df.columns:
        # Fallback jika tidak ada kolom review (misal nama kolomnya 'content' atau 'text')
        # Coba cari kolom lain yang mungkin berisi teks asli
        alternatives = ["content", "text", "body", "ulasan", "cleaned_review"]
        for alt in alternatives:
            if alt in df.columns:
                source_col = alt
                break
    
    print(f"Using column '{source_col}' for aspect extraction (to keep words natural).")

    for _, r in df.iterrows():
        # Ambil teks
        text = str(r.get(source_col, ""))
        
        # Skip kosong
        if not text.strip() or text.lower() == "nan":
            continue

        # Ekstraksi
        matches = extract_aspect_opinion(text)
        
        for m in matches:
            rows.append({
                "original_review": r.get("review", text),
                "opinion_word": m["opinion_word"],   # Hasil: "smooth, addictive" (utuh)
                "processed_opinion": m["opinion_word"], # Backup untuk Step 5
                "aspect": m["aspect"],
                "opinion_context": m["opinion_context"]
            })
            
    if not rows:
        print("Warning: No aspects extracted!")
        out_df = pd.DataFrame(columns=["original_review", "opinion_word", "processed_opinion", "aspect", "opinion_context"])
    else:
        out_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_excel(output_path, index=False)
    print(f"Extracted {len(out_df)} aspect-opinion pairs.")
    
    return out_df

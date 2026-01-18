# =====================================================
# FILE: 2_absa_extraction.py
# FIXED: Clause-Based Aspect Extraction
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
# CONSTANTS (PERBAIKAN KAMUS)
# =====================================================
# HAPUS KATA SIFAT (Adjective) DARI SINI! 
# Aspect Keywords harus KATA BENDA (Noun) atau fitur game.
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
        # Note: "lag" dan "stutter" dipindah ke opini/indikator negatif, 
        # tapi bisa jadi keyword jika user bilang "the lag is bad"
    ],

    "music": [
        "music", "sound", "audio", "sfx", "voice", "acting",
        "soundtrack", "ost", "bgm", "volume", "melody", "song",
        "noise", "dubbing"
    ]
}

# Regex untuk memecah kalimat menjadi klausa terpisah
# Memecah saat ketemu: titik, koma, tanda tanya, seru, atau kata hubung kontras
CLAUSE_SPLIT_PATTERN = r'[.,!?;]| but | however | although | though | yet | while | whereas | except '

# =====================================================
# UTIL
# =====================================================
def valid_word(w):
    return (
        w.isalpha() 
        and w not in STOPWORDS 
        and len(w) > 2
    )

# =====================================================
# ABSA CORE (LOGIKA BARU)
# =====================================================
def extract_aspect_opinion(text):
    if not isinstance(text, str):
        return []

    results = []
    text = text.lower()
    
    # 1. Pecah kalimat menjadi segmen-segmen kecil (Klausa)
    # Contoh: "Gameplay is good but graphics are bad" 
    # Menjadi: ["Gameplay is good", "graphics are bad"]
    clauses = re.split(CLAUSE_SPLIT_PATTERN, text)
    
    seen_aspects_in_text = set()

    for clause in clauses:
        clause = clause.strip()
        if len(clause) < 3:
            continue
            
        tokens = word_tokenize(clause)
        tagged = pos_tag(tokens)
        
        # Cari semua kata sifat (JJ) dan kata kerja (VB) yang relevan di klausa ini
        # Kita ambil JJ (adjective), RB (adverb - "really"), VB (verb - "sucks")
        potential_opinions = []
        for w, t in tagged:
            if (t.startswith("JJ") or t.startswith("VB") or t.startswith("RB")) and valid_word(w):
                potential_opinions.append(w)
        
        # Cari aspek apa yang dibahas di klausa ini
        found_aspect = None
        
        # Cek setiap kata di klausa apakah match dengan keyword aspek
        for w_clause in tokens:
            for aspect, keys in ASPECT_KEYWORDS.items():
                if w_clause in keys:
                    found_aspect = aspect
                    break # Prioritas aspek pertama yang ketemu di klausa
            if found_aspect:
                break
        
        # JIKA di klausa ini ada ASPEK dan ada OPINI
        if found_aspect and potential_opinions:
            # Agar tidak duplikat aspek dalam satu review (opsional)
            # if found_aspect in seen_aspects_in_text: continue
            
            seen_aspects_in_text.add(found_aspect)
            
            # Gabungkan opini menjadi string
            opinion_str = ", ".join(potential_opinions)
            
            results.append({
                "aspect": found_aspect,
                "opinion_word": opinion_str,
                "opinion_context": clause # Konteksnya sekarang per klausa, lebih rapi
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
    
    # Pastikan kolom cleaned_review ada, kalau tidak pakai review asli
    target_col = "cleaned_review" if "cleaned_review" in df.columns else "review"
    
    for _, r in df.iterrows():
        text = str(r.get(target_col, ""))
        
        # Skip jika teks kosong/nan
        if text.strip() == "" or text.lower() == "nan":
            continue

        matches = extract_aspect_opinion(text)
        
        # Jika tidak ada aspek terdeteksi, biarkan kosong atau label 'general'
        # Di sini kita hanya simpan yang ada match
        for m in matches:
            rows.append({
                "original_review": r.get("review", text),
                "processed_opinion": m["opinion_word"], # Digunakan untuk prediksi sentimen nanti
                "aspect": m["aspect"],
                "opinion_context": m["opinion_context"]
            })
            
    if not rows:
        print("Warning: No aspects extracted!")
        # Buat dataframe kosong agar tidak error
        out_df = pd.DataFrame(columns=["original_review", "processed_opinion", "aspect", "opinion_context"])
    else:
        out_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_excel(output_path, index=False)
    print(f"Extracted {len(out_df)} aspect-opinion pairs.")
    
    return out_df

# =====================================================
# TEST BLOCK (Hanya jalan jika file dijalankan langsung)
# =====================================================
if __name__ == "__main__":
    test_text = "The gameplay feels smooth and addictive, but the graphics look outdated and blurry on my PC. I really enjoy the music because it creates a great atmosphere, although the story is quite boring."
    print("Testing extraction...")
    res = extract_aspect_opinion(test_text)
    for r in res:
        print(r)

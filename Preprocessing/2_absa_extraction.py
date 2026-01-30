# =====================================================
# FILE: 2_absa_extraction_fixed.py
# FIXED: Extract MULTIPLE aspects per clause
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
# CONSTANTS & DICTIONARIES
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
    return (
        w.isalpha() 
        and w not in STOPWORDS 
        and len(w) > 2
    )

def light_clean(text):
    """Membersihkan teks asli tanpa stemming agar enak dibaca"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,!?;]', '', text) 
    return text

# =====================================================
# ABSA CORE - FIXED VERSION
# =====================================================
def extract_aspect_opinion(text):
    if not text:
        return []
    
    results = []
    
    # Bersihkan ringan (TANPA STEMMING)
    text = light_clean(text)
    
    # Pecah kalimat menjadi segmen (Klausa)
    clauses = re.split(CLAUSE_SPLIT_PATTERN, text)
    
    for clause in clauses:
        clause = clause.strip()
        if len(clause) < 3:
            continue
        
        tokens = word_tokenize(clause)
        tagged = pos_tag(tokens)
        
        # Cari kata sifat (JJ), Verb (VB), Adverb (RB)
        potential_opinions = []
        for w, t in tagged:
            if (t.startswith("JJ") or t.startswith("VB") or t.startswith("RB")) and valid_word(w):
                potential_opinions.append(w)
        
        # ========== PERBAIKAN UTAMA: CARI SEMUA ASPEK ==========
        found_aspects = []  # Ubah jadi LIST untuk menampung multiple aspects
        
        for w_clause in tokens:
            for aspect, keys in ASPECT_KEYWORDS.items():
                if w_clause in keys:
                    # Hindari duplikat aspek dalam satu klausa
                    if aspect not in found_aspects:
                        found_aspects.append(aspect)
        # =======================================================
        
        # Simpan setiap aspek yang ditemukan
        if found_aspects and potential_opinions:
            opinion_str = ", ".join(potential_opinions)
            
            for aspect in found_aspects:  # Loop semua aspek yang ditemukan
                results.append({
                    "aspect": aspect,
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
    
    # 1. Tentukan kolom teks sumber
    source_col = "review"
    if "review" not in df.columns:
        alternatives = ["content", "text", "body", "ulasan", "cleaned_review"]
        for alt in alternatives:
            if alt in df.columns:
                source_col = alt
                break
    
    # 2. Preserve metadata columns
    preserve_cols = [col for col in df.columns if any(x in col.lower() for x in ['appid', 'app_id', 'game_id', 'steam_id', 'author', 'timestamp', 'voted'])]
    
    print(f"Using column '{source_col}' for extraction.")
    print(f"Preserving Metadata Columns: {preserve_cols}")

    for _, r in df.iterrows():
        text = str(r.get(source_col, ""))
        
        if not text.strip() or text.lower() == "nan":
            continue

        matches = extract_aspect_opinion(text)
        
        for m in matches:
            row_data = {
                "original_review": r.get("review", text),
                "opinion_word": m["opinion_word"],   
                "processed_opinion": m["opinion_word"], 
                "aspect": m["aspect"],
                "opinion_context": m["opinion_context"]
            }
            
            for col in preserve_cols:
                row_data[col] = r[col]
            
            rows.append(row_data)
    
    if not rows:
        print("Warning: No aspects extracted!")
        base_cols = ["original_review", "opinion_word", "processed_opinion", "aspect", "opinion_context"]
        out_df = pd.DataFrame(columns=base_cols + preserve_cols)
    else:
        out_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_excel(output_path, index=False)
    print(f"Extracted {len(out_df)} aspect-opinion pairs. Saved to {output_path}")
    
    return out_df

# =====================================================
# TEST DENGAN CONTOH ANDA
# =====================================================
if __name__ == "__main__":
    test_text = "its a good music, but i dont like about gameplay and story"
    
    print("Testing with:", test_text)
    print("\nResults:")
    results = extract_aspect_opinion(test_text)
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. Aspect: {r['aspect']}")
        print(f"   Opinion: {r['opinion_word']}")
        print(f"   Context: {r['opinion_context']}")

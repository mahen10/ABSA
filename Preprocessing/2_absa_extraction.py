# =====================================================
# FILE: 2_absa_extraction.py
# FIXED: Column Naming Consistency (opinion_word)
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

# Regex untuk memecah kalimat menjadi klausa terpisah
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
# ABSA CORE
# =====================================================
def extract_aspect_opinion(text):
    if not isinstance(text, str):
        return []

    results = []
    text = text.lower()
    
    # Pecah kalimat menjadi segmen-segmen kecil (Klausa)
    clauses = re.split(CLAUSE_SPLIT_PATTERN, text)
    
    seen_aspects_in_text = set()

    for clause in clauses:
        clause = clause.strip()
        if len(clause) < 3:
            continue
            
        tokens = word_tokenize(clause)
        tagged = pos_tag(tokens)
        
        # Cari semua kata sifat (JJ), Verb (VB), Adverb (RB)
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
        
        # JIKA di klausa ini ada ASPEK dan ada OPINI
        if found_aspect and potential_opinions:
            seen_aspects_in_text.add(found_aspect)
            
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
    
    # Pastikan kolom cleaned_review ada, kalau tidak pakai review asli
    target_col = "cleaned_review" if "cleaned_review" in df.columns else "review"
    
    for _, r in df.iterrows():
        text = str(r.get(target_col, ""))
        
        if text.strip() == "" or text.lower() == "nan":
            continue

        matches = extract_aspect_opinion(text)
        
        for m in matches:
            rows.append({
                "original_review": r.get("review", text),
                # PERBAIKAN UTAMA: Mengganti nama kolom kembali ke 'opinion_word'
                "opinion_word": m["opinion_word"], 
                "aspect": m["aspect"],
                "opinion_context": m["opinion_context"]
            })
            
    if not rows:
        print("Warning: No aspects extracted!")
        # Fallback empty dataframe dengan kolom yang BENAR
        out_df = pd.DataFrame(columns=["original_review", "opinion_word", "aspect", "opinion_context"])
    else:
        out_df = pd.DataFrame(rows)
    
    # Copy kolom opinion_word ke processed_opinion juga untuk jaga-jaga (Step 5 butuh processed_opinion)
    if "opinion_word" in out_df.columns:
        out_df["processed_opinion"] = out_df["opinion_word"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_excel(output_path, index=False)
    print(f"Extracted {len(out_df)} aspect-opinion pairs.")
    
    return out_df

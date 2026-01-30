import pandas as pd
import os
import re
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Init NLTK
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)

# ==========================================
# CONFIG
# ==========================================
BASE_DIR = os.getcwd()
INPUT_PATH = os.path.join(BASE_DIR, 'Output', 'DataSet', '05_cleaning.xlsx')
OUTPUT_PATH = os.path.join(BASE_DIR, 'Output', 'DataSet', '06_absa_extraction.xlsx')

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

# ==========================================
# FUNCTIONS
# ==========================================

def light_clean(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Sisakan tanda baca penting untuk pemisah klausa
    text = re.sub(r'[^a-z0-9\s.,!?;]', '', text) 
    return text

def valid_word(w):
    return w.isalpha() and w not in STOPWORDS and len(w) > 2

def extract(text):
    if not text: return []
    
    text = light_clean(text)
    
    # Pecah kalimat berdasarkan tanda baca
    clauses = re.split(r'[.,!?;]| but | however | although | though | yet | while ', text)
    
    results = []
    
    for clause in clauses:
        clause = clause.strip()
        if len(clause) < 3: continue
            
        tokens = word_tokenize(clause)
        tagged = pos_tag(tokens)
        
        # 1. Cari Opini
        opinions = []
        for w, t in tagged:
            if (t.startswith("JJ") or t.startswith("VB") or t.startswith("RB")) and valid_word(w):
                opinions.append(w)
        
        # 2. Cari SEMUA Aspek dalam kalimat (PERBAIKAN DI SINI)
        found_aspects = set() # Gunakan Set biar tidak duplikat
        for w in tokens:
            for aspect, keys in ASPECT_KEYWORDS.items():
                if w in keys:
                    found_aspects.add(aspect)
                    # Jangan break outer loop, lanjut cari token berikutnya
        
        # 3. Simpan Hasil (Looping setiap aspek yang ketemu)
        if found_aspects and opinions:
            opinion_str = ", ".join(opinions)
            
            for aspect in found_aspects:
                results.append({
                    "aspect": aspect,
                    "opinion_word": opinion_str,
                    "opinion_context": clause
                })
            
    return results

# ==========================================
# MAIN RUN
# ==========================================
def run():
    print("--- [6] Mulai ABSA Extraction ---")
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} tidak ditemukan.")
        return
    
    df = pd.read_excel(INPUT_PATH, engine='openpyxl')
    rows = []
    
    # Cari kolom original
    orig_col = 'review' 
    if orig_col not in df.columns:
        for c in df.columns:
            if 'original' in c or 'content' in c: orig_col = c; break
    
    print(f"Sumber teks: '{orig_col}'")

    # Identifikasi kolom ID
    id_cols = [c for c in df.columns if any(x in c.lower() for x in ['appid', 'app_id', 'game_id', 'steam_id', 'author', 'timestamp', 'voted'])]

    for _, r in df.iterrows():
        original_text = str(r.get(orig_col, ''))
        matches = extract(original_text)
        
        for m in matches:
            row = {
                "original_review": original_text,
                "cleaned_source": r.get('cleaned_review', ''),
                "aspect": m['aspect'],
                "opinion_word": m['opinion_word'],
                "opinion_context": m['opinion_context']
            }
            # Salin ID
            for col in id_cols: row[col] = r[col]
            rows.append(row)
            
    out_df = pd.DataFrame(rows)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df.to_excel(OUTPUT_PATH, index=False)
    
    print(f"✅ Selesai. Extracted {len(out_df)} baris data.")
    print(f"Disimpan di: {OUTPUT_PATH}")
    
    # Cek output contoh manual Anda
    print("\n--- Test Case (Manual Example) ---")
    manual_test = extract("its a good music, but i dont like about gameplay and story")
    for m in manual_test:
        print(f"Aspek: {m['aspect']} | Opini: {m['opinion_word']}")

if __name__ == "__main__":
    run()

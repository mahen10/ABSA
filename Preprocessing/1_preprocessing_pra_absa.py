# ============================
# FILE: 1_preprocessing_pra_absa.py
# Deskripsi: Preprocessing Tahap 1 (Pra-ABSA)
# Tahapan FINAL:
# 1. Case Folding
# 2. Normalisasi Kontraksi
# 3. Normalisasi Slang
# 4. Normalisasi Elongation
# 5. Cleaning (FINAL)
# ============================

import pandas as pd
import re
import os
import contractions


# ============================
# LOAD SLANG LEXICON
# format: slang`normalisasi
# ============================
def load_slang_dict(file_path):
    slang_dict = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '`' not in line:
                continue
            
            slang, normal = line.strip().split('`', 1)
            normal = normal.split('|')[0]  # ambil makna pertama
            slang_dict[slang.lower()] = normal.lower()
    
    return slang_dict


# ============================
# 1. CASE FOLDING
# ============================
def case_folding(text):
    if pd.isna(text):
        return ''
    return str(text).lower()


# ============================
# 2. NORMALISASI KONTRAKSI
# contoh: don't -> do not
# ============================
def normalize_contraction(text):
    return contractions.fix(text)


# ============================
# 3. NORMALISASI SLANG
# ============================
def normalize_slang(text, slang_dict):
    tokens = text.split()
    tokens = [slang_dict[t] if t in slang_dict else t for t in tokens]
    return ' '.join(tokens)


# ============================
# 4. NORMALISASI ELONGATION
# contoh: goooooddddd -> good
# ============================
def normalize_elongation(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)


# ============================
# 5. CLEANING (FINAL)
# ============================
def cleaning(text):
    text = re.sub(r'[^a-z0-9\s.,!?;]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================
# MAIN PIPELINE FUNCTION
# ============================
def run(input_path, output_path):
    """
    Menjalankan seluruh preprocessing tahap Pra-ABSA
    
    Args:
        input_path: Path ke file input Excel
        output_path: Path untuk menyimpan output Excel
    
    Returns:
        DataFrame hasil preprocessing
    """
    
    # ===== LOAD DATA =====
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")
    
    df = pd.read_excel(input_path, engine='openpyxl')
    
    if 'review' not in df.columns:
        raise ValueError("Kolom 'review' tidak ditemukan.")
    
    # ===== LOAD SLANG =====
    # Path slang dict relatif terhadap file ini
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    slang_path = os.path.join(base_dir, 'dict', 'slang.txt')
    
    if not os.path.exists(slang_path):
        raise FileNotFoundError(f"File slang tidak ditemukan: {slang_path}")
    
    slang_dict = load_slang_dict(slang_path)
    
    # ===== 1. CASE FOLDING =====
    df['case_folding'] = df['review'].apply(case_folding)
    
    # ===== 2. NORMALISASI KONTRAKSI =====
    df['normalized'] = df['case_folding'].apply(normalize_contraction)
    
    # ===== 3. NORMALISASI SLANG =====
    df['normalisasi'] = df['normalized'].apply(
        lambda x: normalize_slang(x, slang_dict)
    )
    
    # ===== 4. NORMALISASI ELONGATION =====
    df['normalisasi'] = df['normalisasi'].apply(normalize_elongation)
    
    # ===== 5. CLEANING =====
    df['cleaned_review'] = df['normalisasi'].apply(cleaning)
    
    # ===== SAVE OUTPUT =====
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False, engine='openpyxl')
    
    print("Preprocessing Pra-ABSA selesai.")
    print(f"Jumlah data: {len(df)}")
    print(f"File disimpan di: {output_path}")
    
    return df

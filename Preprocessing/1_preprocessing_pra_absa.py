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
# PATH SETTING (TIDAK DIUBAH)
# ============================
INPUT_PATH = os.path.join('Output', 'DataSet.xlsx')
SLANG_PATH = os.path.join('dict', 'slang.txt')
OUTPUT_PATH = os.path.join('Output', 'cleaned_reviews.xlsx')


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
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================
# MAIN PIPELINE FUNCTION
# ============================
def run_preprocessing_pra_absa():
    """
    Menjalankan seluruh preprocessing tahap Pra-ABSA
    Output: Excel cleaned_reviews.xlsx
    """

    # ===== LOAD DATA =====
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"File tidak ditemukan: {INPUT_PATH}")

    df = pd.read_excel(INPUT_PATH, engine='openpyxl')

    if 'review' not in df.columns:
        raise ValueError("Kolom 'review' tidak ditemukan.")

    # ===== LOAD SLANG =====
    slang_dict = load_slang_dict(SLANG_PATH)

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
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False, engine='openpyxl')

    # ===== CONTROL OUTPUT =====
    print("Preprocessing Pra-ABSA selesai.")
    print(f"Jumlah data: {len(df)}")
    print(df[['review', 'case_folding', 'normalized', 'normalisasi', 'cleaned_review']].head(10))
    print(f"File disimpan di: {OUTPUT_PATH}")

    return df

# ============================
# FILE: 1_preprocessing_pra_absa.py
# Deskripsi: Preprocessing Tahap 1 (Pra-ABSA)
# ============================

import pandas as pd
import re
import os
import contractions

# ============================
# LOAD SLANG LEXICON
# ============================
def load_slang_dict(file_path):
    slang_dict = {}
    
    # Cek apakah file ada
    if not os.path.exists(file_path):
        print(f"WARNING: File slang tidak ditemukan di {file_path}. Skip normalisasi slang.")
        return {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '`' not in line:
                continue

            try:
                slang, normal = line.strip().split('`', 1)
                normal = normal.split('|')[0]  # ambil makna pertama
                slang_dict[slang.lower()] = normal.lower()
            except ValueError:
                continue # Skip baris yang formatnya salah

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
# ============================
def normalize_contraction(text):
    try:
        return contractions.fix(text)
    except:
        return text


# ============================
# 3. NORMALISASI SLANG
# ============================
def normalize_slang(text, slang_dict):
    if not slang_dict:
        return text
    
    tokens = text.split()
    tokens = [slang_dict[t] if t in slang_dict else t for t in tokens]
    return ' '.join(tokens)


# ============================
# 4. NORMALISASI ELONGATION
# ============================
def normalize_elongation(text):
    # Mengubah "goooood" menjadi "good"
    return re.sub(r'(.)\1{2,}', r'\1', text)


# ============================
# 5. CLEANING (FINAL)
# ============================
def cleaning(text):
    # Hapus karakter selain huruf dan angka
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================
# MAIN PIPELINE FUNCTION (WAJIB: run)
# ============================
def run(input_path, output_path):
    """
    Fungsi utama yang dipanggil oleh app.py
    Menerima input_path dan output_path dari app.py
    """
    print("--- Memulai Step 1: Preprocessing Pra-ABSA ---")

    # 1. Setup Path untuk Slang Dictionary
    # Asumsi struktur folder:
    # Root/
    #   Preprocessing/1_preprocessing_pra_absa.py
    #   dict/slang.txt
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Folder Preprocessing
    root_dir = os.path.dirname(current_dir) # Folder Root
    slang_path = os.path.join(root_dir, 'dict', 'slang.txt')

    # 2. Cek Input File
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")

    # 3. Load Data
    df = pd.read_excel(input_path)

    # Cek nama kolom (fleksibel)
    target_col = None
    possible_cols = ['review', 'content', 'text', 'body', 'comment']
    for col in df.columns:
        if col.lower() in possible_cols:
            target_col = col
            break
    
    if not target_col:
        # Jika tidak ada kolom yang cocok, ambil kolom string pertama
        # atau raise error jika ingin strict
        raise ValueError(f"Kolom review tidak ditemukan. Kolom yang tersedia: {list(df.columns)}")

    print(f"Processing column: {target_col}")

    # 4. Load Slang
    slang_dict = load_slang_dict(slang_path)

    # ===== EKSEKUSI TAHAPAN =====
    
    # A. Case Folding
    df['cleaned_review'] = df[target_col].apply(case_folding)

    # B. Normalisasi Kontraksi (Pastikan library 'contractions' terinstall)
    df['cleaned_review'] = df['cleaned_review'].apply(normalize_contraction)

    # C. Normalisasi Slang
    if slang_dict:
        df['cleaned_review'] = df['cleaned_review'].apply(lambda x: normalize_slang(x, slang_dict))

    # D. Normalisasi Elongation
    df['cleaned_review'] = df['cleaned_review'].apply(normalize_elongation)

    # E. Cleaning Final
    df['cleaned_review'] = df['cleaned_review'].apply(cleaning)

    # 5. Simpan Output
    # Pastikan folder output ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Simpan, sertakan kolom original untuk referensi
    cols_to_save = [target_col, 'cleaned_review']
    # Jika ada kolom lain yang ingin disimpan, tambahkan di sini
    
    df.to_excel(output_path, index=False)
    
    print(f"Step 1 Selesai. Disimpan di: {output_path}")
    print(f"Contoh hasil:\n{df[['cleaned_review']].head(3)}")
    
    return df

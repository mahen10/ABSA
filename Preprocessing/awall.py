# ============================
# FILE: 02_preprocessing_pra_absa.py
# Deskripsi: Preprocessing Tahap 1 (Pra-ABSA)
# Tahapan: Case Folding → Cleaning → Normalisasi
# ============================

import pandas as pd
import re
import os

# ============================
# Load data hasil scraping
# ============================
input_path = os.path.join('Output', 'DataSet.xlsx')
df = pd.read_excel(input_path, engine='openpyxl')

# ============================
# 1. CASE FOLDING
# ============================
def case_folding(text):
    return str(text).lower()

df['case_folding'] = df['review'].apply(case_folding)

# ============================
# 2. CLEANING
# - Hapus tanda baca, angka, simbol
# ============================
def cleaning(text):
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaning'] = df['case_folding'].apply(cleaning)

# ============================
# 3. NORMALISASI (placeholder)
# - Akan diisi Ekphrasis / slang dictionary / SymSpell
# ============================
def normalisasi(text):
    return text  # sementara identik, nanti diganti

df['normalisasi'] = df['cleaning'].apply(normalisasi)

# ============================
# Simpan ke Excel
# ============================
output_path = os.path.join('Output', 'preprocessing_pra_absa.xlsx')
df.to_excel(output_path, index=False, engine='openpyxl')

print("Preprocessing Pra-ABSA selesai.")
print(df[['review', 'case_folding', 'cleaning', 'normalisasi']].head())

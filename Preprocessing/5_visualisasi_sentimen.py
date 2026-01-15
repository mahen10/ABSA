# ============================
# FILE: 06_visualisasi_sentimen.py
# Deskripsi: Visualisasi distribusi sentimen per aspek
# ============================

import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Untuk mencegah jendela kosong di Windows
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# Baca file Excel hasil klasifikasi
# ============================
input_path = os.path.join('output', 'absa_labeled_numeric.xlsx')

try:
    df = pd.read_excel(input_path, engine='openpyxl')
except FileNotFoundError:
    print(f"❌ File tidak ditemukan di: {input_path}")
    exit()

# ============================
# Validasi kolom yang dibutuhkan
# ============================
if 'aspect' not in df.columns or 'label_text' not in df.columns:
    print("❌ Pastikan file memiliki kolom 'aspect' dan 'label_text'")
    exit()

# ============================
# Visualisasi Bar Chart
# ============================
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

ax = sns.countplot(data=df, x='aspect', hue='label_text', palette='Set2')

plt.title('Distribusi Sentimen Positif dan Negatif per Aspek')
plt.xlabel('Aspek')
plt.ylabel('Jumlah')
plt.xticks(rotation=45)
plt.legend(title='Kategori Sentimen', loc='upper right')
plt.tight_layout()

# Simpan grafik ke file PNG
output_img = os.path.join('output', 'distribusi_sentimen_per_aspek.png')
plt.savefig(output_img)

# Tampilkan ke layar
plt.show()

print(f"✅ Grafik berhasil disimpan ke: {output_img}")

import pandas as pd

# Baca data dari file Excel
df = pd.read_excel('output/absa_labeled_numeric.xlsx')

# Hanya gunakan data yang sudah diberi label
df = df.dropna(subset=['label_text'])

# Hitung persentase sentimen per aspek
persentase = df.groupby(['aspect', 'label_text']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).round(2)

# Ubah jadi DataFrame
persentase_df = persentase.unstack().fillna(0)

# Tampilkan
print(persentase_df)

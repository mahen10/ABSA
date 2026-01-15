# ============================
# FILE: 04_post_absa_preprocessing_excel_split.py
# Deskripsi: Pisahkan hasil Tokenisasi, Stopword Removal, dan Stemming ke kolom sendiri
# ============================

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os

# Download resource (cukup sekali saja)
nltk.download('punkt')
nltk.download('stopwords')

# Load data hasil ABSA (dari Excel, bukan CSV)
df = pd.read_excel('output/absa_output.xlsx', engine='openpyxl')

# Siapkan stopwords dan stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Fungsi preprocessing bertahap
def preprocess_steps(text):
    # Bersihkan spasi berlebih
    text = re.sub(r'\s+', ' ', text.strip())

    # Tokenisasi
    tokens = word_tokenize(text.lower()) 

    # Stopword Removal
    tokens_no_stop = [t for t in tokens if t.lower() not in stop_words]

    # Stemming
    stemmed = [stemmer.stem(t) for t in tokens_no_stop]

    return tokens, tokens_no_stop, stemmed

# Terapkan ke kolom opinion_context
df[['token', 'no_stopword', 'stem']] = df['opinion_context'].apply(
    lambda x: pd.Series(preprocess_steps(str(x)))
)

# Kolom tambahan
df['processed_opinion'] = df['stem'].apply(lambda x: ' '.join(x))
df['label_sentimen'] = ''

# Simpan ke Excel
output_path = os.path.join('output', 'absa_processed.xlsx')
df.to_excel(output_path, index=False, engine='openpyxl')

print(f"✅ File selesai disimpan di: {output_path}")
print(df[['original_review', 'aspect', 'opinion_word', 'token', 'no_stopword', 'stem']].head(5))

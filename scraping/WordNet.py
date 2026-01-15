# ============================
# FILE: 05_label_sentimen.py
# Deskripsi:
#  - Pemberian label sentimen berdasarkan Lexicon (Umigon)
#  - Deteksi sarkasme berbasis aturan (rule-based)
# ============================

import pandas as pd
import os
import re

# ============================
# 1. Load Umigon Lexicon
# ============================

lexicon_df = pd.read_csv("dict/umigon-lexicon.tsv.txt", sep="\t")
lexicon_df['term'] = lexicon_df['term'].astype(str).str.lower()

lexicon_dict = dict(zip(lexicon_df['term'], lexicon_df['valence']))

positive_words = set(lexicon_df[lexicon_df['valence'] == 'positive']['term'])
negative_words = set(lexicon_df[lexicon_df['valence'] == 'negative']['term'])


# ============================
# 2. Fungsi Deteksi Sarkasme
# ============================

def detect_sarcasm(opinion_context, original_text, literal_polarity):
    """
    Deteksi sarkasme berdasarkan dua sumber:
      - original_review (untuk mendeteksi frasa sarkasme asli)
      - opinion_context (untuk cek kontradiksi polaritas per-aspek)
    """

    ctx = str(opinion_context).lower().strip()
    full = str(original_text).lower().strip()

    # --- LIST KAMI PER-ASPEK (hasil literatur) ---
    sarcasm_markers = [
        "yeah right", "as if", "of course", "sure...", "sure ",
        "oh really", "great job", "nice job", "good job",
        "amazing work", "thanks for nothing"
    ]

    hyperbole = [
        "sooo", "soooo", "totally", "literally",
        "absolutely", "extremely", "completely"
    ]

    # ============================
    # RULE 1 — Explicit Sarcasm Marker
    # (frasa di original_review langsung menandakan sarkasme)
    # ============================
    if any(m in full for m in sarcasm_markers):
        return True

    # ============================
    # RULE 2 — Hyperbole + Contradiction (per-aspek)
    # ============================
    if any(h in full or h in ctx for h in hyperbole):

        # contoh: "literally perfect but it crashes"
        if literal_polarity == "positive" and any(n in ctx for n in negative_words):
            return True

        # contoh: "totally bad but looks amazing"
        if literal_polarity == "negative" and any(p in ctx for p in positive_words):
            return True

    # ============================
    # RULE 3 — Polarity Conflict (PER-ASPEK SAJA)
    # ============================
    positive_in_ctx = any(p in ctx for p in positive_words)
    negative_in_ctx = any(n in ctx for n in negative_words)

    if literal_polarity == "positive" and negative_in_ctx:
        return True

    if literal_polarity == "negative" and positive_in_ctx:
        return True

    return False



# ============================
# 3. Fungsi Label Sentimen Final
# ============================

def label_sentiment(opinion_word, opinion_context, original_text):
    """
    Langkah:
      1. Cek label literal berdasarkan lexicon (Umigon)
      2. Deteksi sarkasme menggunakan original_review + context
      3. Jika sarkas → balik polaritas
    """

    w = str(opinion_word).strip().lower()

    # --- Langkah 1: Polaritas literal menurut lexicon ---
    literal = lexicon_dict.get(w, "unlabeled")

    # Jika tidak ada di lexicon → tidak bisa dilabeli
    if literal not in ["positive", "negative"]:
        return literal

    # --- Langkah 2: Deteksi sarkasme ---
    is_sarcasm = detect_sarcasm(opinion_context, original_text, literal)

    # --- Langkah 3: Jika sarkasme → balik polaritas ---
    if is_sarcasm:
        return "negative" if literal == "positive" else "positive"

    return literal


# ============================
# 4. Load File ABSA yang Sudah Diproses
# ============================

path = "output/absa_processed.xlsx"
df = pd.read_excel(path, engine='openpyxl')

# Pastikan kolom original_review ada
if "original_review" not in df.columns:
    raise ValueError(
        "❌ Kolom 'original_review' harus ada di absa_processed.xlsx "
        "agar deteksi sarkasme bekerja dengan benar."
    )


# ============================
# 5. Terapkan Label per-Aspek
# ============================

df["label_sentimen"] = df.apply(
    lambda row: label_sentiment(
        row["opinion_word"],
        row["opinion_context"],
        row["original_review"]
    ),
    axis=1
)

# ============================
# 6. Simpan Hasil
# ============================

df.to_excel(path, index=False, engine="openpyxl")

print("✅ Label sentimen + deteksi sarkasme selesai!")
print("📁 File diperbarui:", path)

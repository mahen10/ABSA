# =====================================================
# FILE: app.py
# Deskripsi:
# Streamlit App – Pipeline ABSA Game Steam
# =====================================================

import streamlit as st
import os
import importlib.util

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="ABSA Steam Review",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRE_DIR = os.path.join(BASE_DIR, "Preprocessing")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# SAFE MODULE LOADER
# =====================================================
def load_module(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =====================================================
# LOAD PIPELINE MODULES (TANPA EKSEKUSI)
# =====================================================
step01 = load_module(
    os.path.join(PRE_DIR, "1_preprocessing_pra_absa.py"),
    "step01"
)

step02 = load_module(
    os.path.join(PRE_DIR, "2_absa_extraction.py"),
    "step02"
)

step03 = load_module(
    os.path.join(PRE_DIR, "3_post_absa_preprocessing.py"),
    "step03"
)


step04 = load_module(
    os.path.join(PRE_DIR, "auto_lebel.py"),
    "step04"
)

step05 = load_module(
    os.path.join(BASE_DIR, "4_sentiment_classification.py"),
    "step05"
)


# =====================================================
# UI
# =====================================================
st.title("🎮 Aspect-Based Sentiment Analysis")
st.write("Pipeline Analisis Sentimen Ulasan Game Steam")

uploaded_file = st.file_uploader(
    "Upload file Excel hasil scraping",
    type=["xlsx"]
)

if uploaded_file:
    input_path = os.path.join(OUTPUT_DIR, "01_raw.xlsx")
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File berhasil diupload")

    if st.button("🚀 Jalankan Pipeline"):
        try:
            with st.spinner("Preprocessing Pra-ABSA..."):
                out1 = os.path.join(OUTPUT_DIR, "02_pra_absa.xlsx")
                step01.run(input_path, out1)

            with st.spinner("Ekstraksi ABSA..."):
                out2 = os.path.join(OUTPUT_DIR, "03_absa.xlsx")
                step02.run(out1, out2)

            with st.spinner("Post-ABSA Preprocessing..."):
                out3 = os.path.join(OUTPUT_DIR, "04_post_absa.xlsx")
                step03.run(out2, out3)

            with st.spinner("Auto Label Sentimen..."):
                out4 = os.path.join(OUTPUT_DIR, "05_labeled.xlsx")
                step04.run(out3, out4)

            with st.spinner("Klasifikasi Sentimen..."):
                out5 = os.path.join(OUTPUT_DIR, "06_classification.xlsx")
                step05.run(out4, out5)

            st.success("✅ Pipeline selesai")

            with open(out5, "rb") as f:
                st.download_button(
                    label="⬇ Download Hasil Akhir",
                    data=f,
                    file_name="hasil_absa.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error("Terjadi error saat menjalankan pipeline")
            st.exception(e)


# =====================================================
# FILE: app.py
# Streamlit App – ABSA Steam Review (FINAL + VISUAL)
# =====================================================

import streamlit as st
import os
import sys
import traceback
import pandas as pd
import matplotlib.pyplot as plt

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
    import importlib.util

    if not os.path.exists(path):
        st.error(f"File tidak ditemukan: {path}")
        st.stop()

    module_dir = os.path.dirname(path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# =====================================================
# LOAD PIPELINE MODULES
# =====================================================
step01 = load_module(os.path.join(PRE_DIR, "1_preprocessing_pra_absa.py"), "step01")
step02 = load_module(os.path.join(PRE_DIR, "2_absa_extraction.py"), "step02")
step03 = load_module(os.path.join(PRE_DIR, "3_post_absa_preprocessing.py"), "step03")
step04 = load_module(os.path.join(PRE_DIR, "auto_lebel.py"), "step04")
step05 = load_module(os.path.join(BASE_DIR, "4_sentiment_classification.py"), "step05")

# =====================================================
# UI
# =====================================================
st.title("🎮 Aspect-Based Sentiment Analysis")
st.write("Analisis Sentimen Ulasan Game Steam")

uploaded_file = st.file_uploader(
    "Upload file Excel hasil scraping",
    type=["xlsx"]
)

if uploaded_file:
    input_path = os.path.join(OUTPUT_DIR, "01_raw.xlsx")
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("📁 File berhasil diupload")

    if st.button("🚀 Jalankan Pipeline"):

        # ============================
        # PATH OUTPUT
        # ============================
        out1 = os.path.join(OUTPUT_DIR, "02_pra_absa.xlsx")
        out2 = os.path.join(OUTPUT_DIR, "03_absa.xlsx")
        out3 = os.path.join(OUTPUT_DIR, "04_post_absa.xlsx")
        out4 = os.path.join(OUTPUT_DIR, "05_labeled.xlsx")

        progress = st.progress(0)
        status = st.empty()

        try:
            # ============================
            # STEP 1
            # ============================
            status.info("🔄 Step 1/5: Preprocessing Pra-ABSA...")
            progress.progress(10)
            step01.run(input_path, out1)
            status.success("✅ Step 1 selesai")

            # ============================
            # STEP 2
            # ============================
            status.info("🔄 Step 2/5: Ekstraksi ABSA...")
            progress.progress(30)
            step02.run(out1, out2)
            status.success("✅ Step 2 selesai")

            # ============================
            # STEP 3
            # ============================
            status.info("🔄 Step 3/5: Post-ABSA Preprocessing...")
            progress.progress(50)
            step03.run(out2, out3)
            status.success("✅ Step 3 selesai")

            # ============================
            # STEP 4
            # ============================
            status.info("🔄 Step 4/5: Auto Label Sentimen...")
            progress.progress(70)
            step04.run(out3, out4)
            status.success("✅ Step 4 selesai")

            # ============================
            # STEP 5 (Klasifikasi)
            # ============================
            status.info("🔄 Step 5/5: Klasifikasi Sentimen...")
            progress.progress(90)
            result = step05.run(out4, OUTPUT_DIR)
            progress.progress(100)
            status.success("🎉 Semua proses selesai!")

        except Exception:
            st.error("❌ Pipeline gagal")
            st.code(traceback.format_exc())
            st.stop()

        # =====================================================
        # VISUALISASI HASIL
        # =====================================================
        df = pd.read_excel(out4)
        df["label_text"] = df["label_text"].astype(str).str.lower()

        # ============================
        # RINGKASAN

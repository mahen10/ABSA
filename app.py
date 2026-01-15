# =====================================================
# FILE: app.py
# Streamlit App – Pipeline ABSA Game Steam
# =====================================================

import streamlit as st
import os
import sys
import traceback
import pandas as pd

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
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

        module_dir = os.path.dirname(path)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

        return module

    except Exception as e:
        st.error(f"❌ Gagal load module: {name}")
        st.exception(e)
        st.stop()

# =====================================================
# LOAD MODULES
# =====================================================
st.info("🔄 Loading pipeline modules...")

step01 = load_module(os.path.join(PRE_DIR, "1_preprocessing_pra_absa.py"), "step01")
step02 = load_module(os.path.join(PRE_DIR, "2_absa_extraction.py"), "step02")
step03 = load_module(os.path.join(PRE_DIR, "3_post_absa_preprocessing.py"), "step03")
step04 = load_module(os.path.join(PRE_DIR, "auto_lebel.py"), "step04")
step05 = load_module(os.path.join(BASE_DIR, "4_sentiment_classification.py"), "step05")

st.success("✅ Semua module berhasil dimuat")

# =====================================================
# UI
# =====================================================
st.title("🎮 Aspect-Based Sentiment Analysis")
st.write("Pipeline ABSA untuk Ulasan Game Steam")

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
        progress = st.progress(0)
        status = st.empty()

        try:
            # STEP 1
            status.text("Step 1/5: Preprocessing Pra-ABSA...")
            out1 = os.path.join(OUTPUT_DIR, "02_pra_absa.xlsx")
            step01.run(input_path, out1)
            progress.progress(0.2)

            # STEP 2
            status.text("Step 2/5: Ekstraksi ABSA...")
            out2 = os.path.join(OUTPUT_DIR, "03_absa.xlsx")
            step02.run(out1, out2)
            progress.progress(0.4)

            # STEP 3
            status.text("Step 3/5: Post-ABSA Preprocessing...")
            out3 = os.path.join(OUTPUT_DIR, "04_post_absa.xlsx")
            step03.run(out2, out3)
            progress.progress(0.6)

            # STEP 4
            status.text("Step 4/5: Auto Label Sentimen...")
            out4 = os.path.join(OUTPUT_DIR, "05_labeled.xlsx")
            step04.run(out3, out4)
            progress.progress(0.8)

            # STEP 5 (EVALUASI MODEL, TANPA FILE EXCEL)
            status.text("Step 5/5: Klasifikasi Sentimen...")
            result = step05.run(out4, OUTPUT_DIR)
            progress.progress(1.0)

            status.text("✅ Pipeline selesai!")

            # ============================
            # HASIL KLASIFIKASI
            # ============================
            st.subheader("📊 Hasil Klasifikasi Sentimen")

            st.metric(
                "Akurasi Model",
                f"{result['accuracy']:.2%}"
            )

            st.subheader("Classification Report")
            st.dataframe(
                pd.DataFrame(result["classification_report"]).transpose()
            )

            st.subheader("Confusion Matrix")
            st.write(result["confusion_matrix"])

            # ============================
            # DOWNLOAD DATA FINAL
            # ============================
            with open(out4, "rb") as f:
                st.download_button(
                    "⬇ Download Data ABSA + Label",
                    data=f,
                    file_name="hasil_absa_labeled.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error("❌ Terjadi error saat menjalankan pipeline")
            with st.expander("Detail Error"):
                st.code(traceback.format_exc())

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
# LOAD MODULES
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
        try:
            # ============================
            # PIPELINE
            # ============================
            out1 = os.path.join(OUTPUT_DIR, "02_pra_absa.xlsx")
            out2 = os.path.join(OUTPUT_DIR, "03_absa.xlsx")
            out3 = os.path.join(OUTPUT_DIR, "04_post_absa.xlsx")
            out4 = os.path.join(OUTPUT_DIR, "05_labeled.xlsx")

            step01.run(input_path, out1)
            step02.run(out1, out2)
            step03.run(out2, out3)
            step04.run(out3, out4)

            # ============================
            # LOAD DATA LABELED
            # ============================
            df = pd.read_excel(out4)
            df["label_text"] = df["label_text"].str.lower().str.strip()

            # ============================
            # RINGKASAN LABEL
            # ============================
            total = len(df)
            pos = (df["label_text"] == "positive").sum()
            neg = (df["label_text"] == "negative").sum()
            unlabeled = total - pos - neg

            st.subheader("📌 Ringkasan Pelabelan")

            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", pos)
            col2.metric("Negative", neg)
            col3.metric("Tidak Terlabel", unlabeled)

            # ============================
            # BAR CHART – SEMUA ASPEK
            # ============================
            st.subheader("📊 Perbandingan Sentimen (Semua Aspek)")

            overall = (
                df[df["label_text"].isin(["positive", "negative"])]
                .groupby("label_text")
                .size()
            )

            fig, ax = plt.subplots()
            overall.plot(kind="bar", ax=ax)
            ax.set_xlabel("Sentimen")
            ax.set_ylabel("Jumlah")
            ax.set_title("Positive vs Negative (Keseluruhan)")
            st.pyplot(fig)

            # ============================
            # PIE CHART – PER ASPEK
            # ============================
            st.subheader("🧩 Distribusi Sentimen per Aspek")

            aspects = df["aspect"].dropna().unique()

            for aspect in aspects:
                aspect_df = df[
                    (df["aspect"] == aspect) &
                    (df["label_text"].isin(["positive", "negative"]))
                ]

                if aspect_df.empty:
                    continue

                counts = aspect_df["label_text"].value_counts()

                fig, ax = plt.subplots()
                ax.pie(
                    counts,
                    labels=counts.index,
                    autopct="%1.1f%%",
                    startangle=90
                )
                ax.set_title(f"Aspek: {aspect}")
                st.pyplot(fig)

            # ============================
            # DOWNLOAD
            # ============================
            with open(out4, "rb") as f:
                st.download_button(
                    "⬇ Download Data ABSA + Label",
                    data=f,
                    file_name="hasil_absa_labeled.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            # ============================
            # MODEL EVALUATION
            # ============================
            st.subheader("🤖 Evaluasi Model Klasifikasi")
            result = step05.run(out4, OUTPUT_DIR)

            st.metric("Akurasi Model", f"{result['accuracy']:.2%}")
            st.dataframe(
                pd.DataFrame(result["classification_report"]).transpose()
            )
            st.write("Confusion Matrix:")
            st.write(result["confusion_matrix"])

        except Exception:
            st.error("❌ Terjadi error")
            st.code(traceback.format_exc())

# =====================================================
# FILE: app.py
# Streamlit App – ABSA Steam Review (RE-DESIGNED)
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style Matplotlib agar lebih bagus (tidak kaku)
plt.style.use('ggplot')

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
        st.error(f"❌ File tidak ditemukan: {path}")
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
# UI HEADER
# =====================================================
st.title("🎮 Steam Review Analysis")
st.markdown("---")

# Sidebar untuk Upload
with st.sidebar:
    st.header("📂 Data Input")
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
    
    run_btn = False
    if uploaded_file:
        input_path = os.path.join(OUTPUT_DIR, "01_raw.xlsx")
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File Ready!")
        run_btn = st.button("🚀 Jalankan Analisis", type="primary")

# =====================================================
# MAIN PROCESS
# =====================================================
if uploaded_file and run_btn:
    progress_bar = st.progress(0)
    status_text = st.empty()

    # output path
    out1 = os.path.join(OUTPUT_DIR, "02_pre_absa.xlsx")
    out2 = os.path.join(OUTPUT_DIR, "03_absa.xlsx")
    out3 = os.path.join(OUTPUT_DIR, "04_post_absa.xlsx")
    out4 = os.path.join(OUTPUT_DIR, "05_labeled.xlsx")

    try:
        # STEP 1-5 (Sama seperti sebelumnya)
        status_text.write("⏳ Sedang memproses: Preprocessing...")
        progress_bar.progress(10)
        step01.run(input_path, out1)

        status_text.write("⏳ Sedang memproses: Ekstraksi Aspek...")
        progress_bar.progress(30)
        step02.run(out1, out2)

        status_text.write("⏳ Sedang memproses: Cleaning Lanjutan...")
        progress_bar.progress(50)
        step03.run(out2, out3)

        status_text.write("⏳ Sedang memproses: Pelabelan Otomatis...")
        progress_bar.progress(70)
        step04.run(out3, out4)

        status_text.write("⏳ Sedang memproses: Klasifikasi Model...")
        progress_bar.progress(90)
        result = step05.run(out4, OUTPUT_DIR)

        progress_bar.progress(100)
        status_text.success("✅ Analisis Selesai!")

        # =====================================================
        # VISUALISASI DASHBOARD (DIPERBAIKI)
        # =====================================================
        df = pd.read_excel(out4)
        df["label_text"] = df["label_text"].str.lower().str.strip()

        # --- 1. RINGKASAN DATA (METRICS) ---
        st.subheader("📌 Ringkasan Hasil")
        
        pos = (df["label_text"] == "positive").sum()
        neg = (df["label_text"] == "negative").sum()
        total = len(df)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Data", total)
        col2.metric("Positive", pos, delta="Good", delta_color="normal")
        col3.metric("Negative", neg, delta="-Bad", delta_color="inverse")
        col4.download_button(
            "⬇ Download Excel",
            data=open(out4, "rb"),
            file_name="hasil_analisis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("---")

        # --- 2. GRAFIK SENTIMEN (Kiri) & CONFUSION MATRIX (Kanan) ---
        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.subheader("📊 Distribusi Sentimen Global")
            # Membuat grafik yang lebih kecil dan rapi
            fig1, ax1 = plt.subplots(figsize=(6, 3)) 
            counts = df["label_text"].value_counts()
            # Warna custom: Hijau untuk positif, Merah untuk negatif
            colors = ['#4CAF50' if x == 'positive' else '#F44336' for x in counts.index]
            
            counts.plot(kind="barh", ax=ax1, color=colors, width=0.6)
            ax1.set_xlabel("Jumlah")
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=False) # False agar tidak stretch terlalu besar

        with c_right:
            st.subheader("🤖 Performa Model")
            st.write(f"**Akurasi: {result['accuracy']:.2%}**")
            
            # Tampilkan Confusion Matrix dalam bentuk DataFrame agar rapi
            cm_df = pd.DataFrame(
                result["confusion_matrix"], 
                index=["Aktual Neg", "Aktual Pos"], 
                columns=["Prediksi Neg", "Prediksi Pos"]
            )
            st.table(cm_df)

        st.markdown("---")

        # --- 3. DISTRIBUSI PER ASPEK (DIGABUNG BIAR RAPI) ---
        st.subheader("🧩 Analisis Detail Per Aspek")
        
        # Pivot data untuk membuat Grouped Bar Chart
        # Ini menggantikan banyak Pie Chart menjadi SATU grafik besar yang informatif
        aspect_sentiment = df.groupby(['aspect', 'label_text']).size().unstack(fill_value=0)
        
        # Pastikan kolom positive/negative ada
        if 'positive' not in aspect_sentiment.columns: aspect_sentiment['positive'] = 0
        if 'negative' not in aspect_sentiment.columns: aspect_sentiment['negative'] = 0
        
        # Plotting
        fig2, ax2 = plt.subplots(figsize=(10, 4)) # Wide tapi pendek
        aspect_sentiment[['positive', 'negative']].plot(
            kind='bar', 
            ax=ax2, 
            color=['#4CAF50', '#F44336'], # Hijau & Merah
            width=0.7
        )
        
        ax2.set_title("Perbandingan Positif vs Negatif pada Setiap Aspek")
        ax2.set_ylabel("Jumlah Ulasan")
        ax2.set_xlabel("Aspek Game")
        ax2.legend(["Positive", "Negative"])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig2)

    except Exception:
        st.error("❌ Terjadi error sistem")
        st.code(traceback.format_exc())

elif not uploaded_file:
    st.info("👈 Silakan upload file Excel di menu sebelah kiri untuk memulai.")

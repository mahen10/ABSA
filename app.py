# =====================================================
# FILE: app.py
# Streamlit App – ABSA Steam Review (ROBUST & USER FRIENDLY)
# =====================================================

import streamlit as st
import os
import sys
import traceback
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# CONFIG & STYLE
# =====================================================
st.set_page_config(
    page_title="ABSA Steam Review",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style Matplotlib
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
# UI HEADER & SIDEBAR
# =====================================================
st.title("🎮 Steam Review Analysis")
st.markdown("---")

# Variabel flag global
start_process = False
uploaded_file = None 

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    input_mode = st.radio("Pilih Sumber Data:", ["📂 Upload Excel", "✍️ Input Teks Manual"])
    
    st.markdown("---")
    
    if input_mode == "📂 Upload Excel":
        uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
        if uploaded_file:
            input_path = os.path.join(OUTPUT_DIR, "01_raw.xlsx")
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("File Terupload!")
            if st.button("🚀 Jalankan Analisis", key="btn_upload"):
                start_process = True

    elif input_mode == "✍️ Input Teks Manual":
        st.info("Ketik ulasan game di bawah ini.")
        
        # --- FITUR BARU: PANDUAN KATA KUNCI ---
        with st.expander("ℹ️ Tips: Gunakan kata kunci ini agar terdeteksi!"):
            st.markdown("""
            **Sistem mencari kata-kata ini:**
            * 🎨 **Graphics:** graphics, visual, art, texture, animation, lighting...
            * ⚔️ **Gameplay:** gameplay, combat, mechanics, action, quest, level...
            * 📜 **Story:** story, plot, narrative, ending, character, dialogue...
            * 🚀 **Performance:** fps, lag, crash, bug, optimization, freeze...
            * 🎵 **Music:** music, sound, audio, soundtrack, voice, sfx...
            """)
            
        user_text = st.text_area("Masukkan Review Game:", height=150, placeholder="Contoh: The graphics are amazing but the gameplay is boring.")
        
        if st.button("🚀 Analisis Teks", key="btn_manual"):
            if user_text.strip():
                df_manual = pd.DataFrame({"review": [user_text]}) 
                input_path = os.path.join(OUTPUT_DIR, "01_raw.xlsx")
                df_manual.to_excel(input_path, index=False)
                start_process = True
            else:
                st.warning("Mohon isi teks terlebih dahulu.")

# =====================================================
# MAIN PROCESS LOGIC
# =====================================================
if start_process:
    progress_bar = st.progress(0)
    status_text = st.empty()

    out1 = os.path.join(OUTPUT_DIR, "02_pre_absa.xlsx")
    out2 = os.path.join(OUTPUT_DIR, "03_absa.xlsx")
    out3 = os.path.join(OUTPUT_DIR, "04_post_absa.xlsx")
    out4 = os.path.join(OUTPUT_DIR, "05_labeled.xlsx")

    try:
        # --- PIPELINE START ---
        status_text.write("⏳ Step 1/5: Preprocessing...")
        progress_bar.progress(10)
        step01.run(input_path, out1)

        status_text.write("⏳ Step 2/5: Ekstraksi Aspek (ABSA)...")
        progress_bar.progress(30)
        step02.run(out1, out2)

        status_text.write("⏳ Step 3/5: Cleaning Lanjutan...")
        progress_bar.progress(50)
        step03.run(out2, out3)

        status_text.write("⏳ Step 4/5: Pelabelan Otomatis...")
        progress_bar.progress(70)
        step04.run(out3, out4)

        status_text.write("⏳ Step 5/5: Klasifikasi & Evaluasi Model...")
        progress_bar.progress(90)
        result = step05.run(out4, OUTPUT_DIR)

        progress_bar.progress(100)
        status_text.success("✅ Analisis Selesai!")
        # --- PIPELINE END ---

        # =====================================================
        # BAGIAN: PREVIEW TABEL DATA
        # =====================================================
        st.markdown("### 🔍 Deteksi Aspek & Sentimen")
        
        if os.path.exists(out4):
            df_final = pd.read_excel(out4)
            
            # --- PERBAIKAN CRITICAL: CEK APAKAH HASIL KOSONG ---
            if df_final.empty:
                st.warning("⚠️ **Tidak ada aspek game yang terdeteksi dari teks Anda.**")
                st.info("Coba masukkan kalimat yang lebih spesifik, misal: 'The graphics are bad' atau 'Gameplay is fun'.")
                st.stop() # Hentikan proses agar tidak error grafik
            
            df_final["label_text"] = df_final["label_text"].str.lower().str.strip()
            
            if input_mode == "✍️ Input Teks Manual":
                st.info("Berikut adalah hasil pembedahan kalimat Anda:")
                cols_to_show = [col for col in df_final.columns if col in ['aspect', 'processed_opinion', 'label_text', 'sentiment_score']]
                st.table(df_final[cols_to_show])
            else:
                with st.expander("🏷️ Klik untuk melihat Data Final Terlabeli"):
                     st.dataframe(df_final, use_container_width=True)
        
        st.markdown("---")

        # =====================================================
        # VISUALISASI DASHBOARD
        # =====================================================
        st.subheader("📌 Ringkasan Hasil")
        
        pos = (df_final["label_text"] == "positive").sum()
        neg = (df_final["label_text"] == "negative").sum()
        total = len(df_final)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Data", total)
        col2.metric("Positive", pos, delta="Good", delta_color="normal")
        col3.metric("Negative", neg, delta="-Bad", delta_color="inverse")
        
        with open(out4, "rb") as f:
            col4.download_button(
                "⬇ Download Hasil",
                data=f,
                file_name="hasil_analisis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        st.markdown("---")

        # --- GRAFIK KIRI & KANAN ---
        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.subheader("📊 Distribusi Sentimen")
            fig1, ax1 = plt.subplots(figsize=(6, 3)) 
            counts = df_final["label_text"].value_counts()
            
            if not counts.empty:
                colors = ['#4CAF50' if x == 'positive' else '#F44336' for x in counts.index]
                counts.plot(kind="barh", ax=ax1, color=colors, width=0.6)
            else:
                st.write("Tidak ada data untuk ditampilkan.")

            ax1.set_xlabel("Jumlah")
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=False)

        with c_right:
            st.subheader("🤖 Evaluasi Model")
            # Jika result kosong (dummy), tampilkan pesan
            if result['accuracy'] == 0 and total < 5:
                st.info("Model tidak melakukan split test karena data terlalu sedikit (<5). Akurasi ditampilkan sebagai 0% atau dummy.")
            else:
                st.write(f"**Akurasi Dataset (Test Split): {result['accuracy']:.2%}**")
            
            cm_df = pd.DataFrame(
                result["confusion_matrix"], 
                index=["Aktual Neg", "Aktual Pos"], 
                columns=["Prediksi Neg", "Prediksi Pos"]
            )
            st.table(cm_df)
        
        # --- GRAFIK ASPEK (FIXED ERROR) ---
        st.markdown("---")
        st.subheader("🧩 Analisis Detail Per Aspek")
        
        # Pivot data untuk grafik
        aspect_sentiment = df_final.groupby(['aspect', 'label_text']).size().unstack(fill_value=0)
        
        # Cek apakah hasil pivot kosong
        if not aspect_sentiment.empty:
            # Pastikan kolom positive/negative ada
            if 'positive' not in aspect_sentiment.columns: aspect_sentiment['positive'] = 0
            if 'negative' not in aspect_sentiment.columns: aspect_sentiment['negative'] = 0
            
            # Plotting
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            aspect_sentiment[['positive', 'negative']].plot(
                kind='bar', 
                ax=ax2, 
                color=['#4CAF50', '#F44336'], 
                width=0.7
            )
            
            ax2.set_title("Sentimen per Aspek")
            ax2.set_ylabel("Jumlah")
            ax2.set_xlabel("Aspek")
            ax2.legend(["Positive", "Negative"])
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("Belum cukup data untuk menampilkan grafik per aspek.")

    except Exception:
        st.error("❌ Terjadi error sistem")
        st.code(traceback.format_exc())

elif not start_process and uploaded_file is None and input_mode == "📂 Upload Excel":
    st.info("👈 Silakan upload file Excel di menu sebelah kiri.")

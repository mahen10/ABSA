# =====================================================
# FILE: app.py
# Streamlit App – ABSA Steam Review (FINAL WITH MANUAL INPUT)
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

# Variabel flag untuk trigger proses
start_process = False

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    # PILIHAN MODE: UPLOAD atau MANUAL
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
        st.info("Ketik ulasan game di bawah ini untuk dianalisis langsung.")
        # Text Area untuk input manual
        user_text = st.text_area("Masukkan Review Game:", height=150, placeholder="Contoh: Game ini grafiknya bagus banget, tapi harganya terlalu mahal.")
        
        if st.button("🚀 Analisis Teks", key="btn_manual"):
            if user_text.strip():
                # --- TRIK: MEMBUAT EXCEL PALSU DARI INPUT USER ---
                # PENTING: Ganti 'content' di bawah ini dengan nama kolom 
                # yang biasa dibaca oleh file '1_preprocessing_pra_absa.py' Anda.
                # Misal: 'content', 'review', 'text', atau 'ulasan'.
                df_manual = pd.DataFrame({"content": [user_text]}) 
                
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

    # Define output paths
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
        
        # Cek hasil ekstraksi
        if os.path.exists(out4):
            df_final = pd.read_excel(out4)
            df_final["label_text"] = df_final["label_text"].str.lower().str.strip()
            
            # Jika mode manual, tampilkan tabel lebih sederhana
            if input_mode == "✍️ Input Teks Manual":
                st.info("Berikut adalah hasil pembedahan kalimat Anda:")
                # Tampilkan kolom penting saja agar user fokus
                cols_to_show = [col for col in df_final.columns if col in ['aspect', 'processed_opinion', 'label_text', 'sentiment_score']]
                st.table(df_final[cols_to_show])
            else:
                # Mode upload file (tampilkan expander seperti biasa)
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
        
        # Tombol download hanya relevan jika upload file banyak, tapi dibiarkan saja
        with open(out4, "rb") as f:
            col4.download_button(
                "⬇ Download Hasil",
                data=f,
                file_name="hasil_analisis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        st.markdown("---")

        # --- GRAFIK ---
        # Untuk input manual 1 kalimat, grafik mungkin terlihat 100% satu warna, 
        # tapi tetap kita tampilkan untuk konsistensi.
        
        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.subheader("📊 Distribusi Sentimen")
            fig1, ax1 = plt.subplots(figsize=(6, 3)) 
            counts = df_final["label_text"].value_counts()
            colors = ['#4CAF50' if x == 'positive' else '#F44336' for x in counts.index]
            
            if not counts.empty:
                counts.plot(kind="barh", ax=ax1, color=colors, width=0.6)
            
            ax1.set_xlabel("Jumlah")
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=False)

        with c_right:
            st.subheader("🤖 Evaluasi Model")
            st.write(f"**Akurasi Dataset (Test Split): {result['accuracy']:.2%}**")
            # Note: Confusion matrix ini dari test split dataset training terakhir,
            # bukan dari input manual Anda saat ini.
            st.caption("*Confusion Matrix ini berdasarkan performa model pada data latih/uji, bukan input barusan.*")
            
            cm_df = pd.DataFrame(
                result["confusion_matrix"], 
                index=["Aktual Neg", "Aktual Pos"], 
                columns=["Prediksi Neg", "Prediksi Pos"]
            )
            st.table(cm_df)

    except Exception:
        st.error("❌ Terjadi error sistem")
        st.code(traceback.format_exc())

elif not start_process and not uploaded_file and input_mode == "📂 Upload Excel":
    st.info("👈 Silakan upload file Excel di menu sebelah kiri.")

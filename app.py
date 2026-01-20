# =====================================================
# FILE: app.py
# Streamlit App – ABSA Steam Review (ULTIMATE VERSION)
# Fitur: Upload, Manual Input, Scraping, AI Insight
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

# Style Matplotlib agar grafik terlihat modern
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
# Load Scraper (Step 0)
step00 = load_module(os.path.join(PRE_DIR, "0_steam_scraper.py"), "step00")

# Load Pipeline Steps (1-5)
step01 = load_module(os.path.join(PRE_DIR, "1_preprocessing_pra_absa.py"), "step01")
step02 = load_module(os.path.join(PRE_DIR, "2_absa_extraction.py"), "step02")
step03 = load_module(os.path.join(PRE_DIR, "3_post_absa_preprocessing.py"), "step03")
step04 = load_module(os.path.join(PRE_DIR, "auto_lebel.py"), "step04")
step05 = load_module(os.path.join(BASE_DIR, "4_sentiment_classification.py"), "step05")

# =====================================================
# FITUR: AI INSIGHT GENERATOR (LOGIC BASED)
# =====================================================
def generate_smart_insight(df, accuracy):
    """
    Fungsi ini bertindak seperti 'Copilot' sederhana yang membaca data
    dan mengubahnya menjadi narasi kesimpulan yang mudah dibaca manusia.
    """
    if df.empty:
        return "Belum cukup data untuk menyimpulkan."

    # 1. Hitung Statistik Global
    total = len(df)
    pos = (df['label_text'] == 'positive').sum()
    neg = (df['label_text'] == 'negative').sum()
    
    dominance = "Positif" if pos >= neg else "Negatif"
    percent_dom = (max(pos, neg) / total) * 100 if total > 0 else 0

    # 2. Cari Aspek Paling Bermasalah & Terbaik
    aspect_stats = df.groupby('aspect')['label_text'].value_counts().unstack(fill_value=0)
    
    # Pastikan kolom ada
    if 'negative' not in aspect_stats.columns: aspect_stats['negative'] = 0
    if 'positive' not in aspect_stats.columns: aspect_stats['positive'] = 0

    # Cari aspek dengan jumlah negatif terbanyak
    worst_aspect = aspect_stats['negative'].idxmax() if not aspect_stats.empty else "N/A"
    worst_count = aspect_stats.loc[worst_aspect, 'negative'] if not aspect_stats.empty else 0
    
    # Cari aspek dengan jumlah positif terbanyak
    best_aspect = aspect_stats['positive'].idxmax() if not aspect_stats.empty else "N/A"
    best_count = aspect_stats.loc[best_aspect, 'positive'] if not aspect_stats.empty else 0

    # 3. Rangkai Kalimat (Prompt Engineering versi Python String)
    insight = f"""
    ### 🤖 AI Copilot Summary
    Berdasarkan analisis terhadap **{total} ulasan**, berikut adalah kesimpulan otomatis:
    
    1.  **Sentimen Dominan:** Mayoritas pengguna memberikan respon **{dominance}** ({percent_dom:.1f}%).
    2.  **Kekuatan Utama:** Aspek **{best_aspect.upper()}** paling banyak dipuji (mendapat {best_count} respon positif). Ini adalah fitur unggulan game ini.
    3.  **Kelemahan Kritis:** Pengguna paling banyak mengeluh soal **{worst_aspect.upper()}** (mendapat {worst_count} keluhan). Developer disarankan untuk segera memperbaiki sektor ini.
    4.  **Kualitas Model:** Analisis ini didukung oleh model AI dengan tingkat akurasi **{accuracy:.1f}%**, sehingga hasil prediksi cukup dapat dipercaya.
    """
    return insight

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
    
    # 3 Opsi Input: Upload, Manual, Scraping
    input_mode = st.radio("Pilih Sumber Data:", 
                          ["📂 Upload Excel", "✍️ Input Teks Manual", "🕷️ Scraping Steam ID"])
    
    st.markdown("---")
    
    # --- MODE 1: UPLOAD EXCEL ---
    if input_mode == "📂 Upload Excel":
        uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
        if uploaded_file:
            input_path = os.path.join(OUTPUT_DIR, "01_raw.xlsx")
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("File Terupload!")
            if st.button("🚀 Jalankan Analisis", key="btn_upload"):
                start_process = True

    # --- MODE 2: INPUT MANUAL ---
    elif input_mode == "✍️ Input Teks Manual":
        st.info("Ketik ulasan game di bawah ini.")
        
        # Contekan Keyword
        with st.expander("ℹ️ Tips: Gunakan kata kunci ini!"):
            st.markdown("""
            * 🎨 **Graphics:** graphics, visual, art, texture...
            * ⚔️ **Gameplay:** gameplay, combat, mechanics, action...
            * 📜 **Story:** story, plot, narrative, ending...
            * 🚀 **Performance:** fps, lag, crash, bug, optimization...
            * 🎵 **Music:** music, sound, audio, soundtrack...
            """)
            
        user_text = st.text_area("Masukkan Review Game:", height=150, placeholder="Contoh: The graphics are amazing but the gameplay is boring.")
        
        if st.button("🚀 Analisis Teks", key="btn_manual"):
            if user_text.strip():
                # Simpan ke Excel dummy dengan kolom 'review'
                df_manual = pd.DataFrame({"review": [user_text]}) 
                input_path = os.path.join(OUTPUT_DIR, "01_raw.xlsx")
                df_manual.to_excel(input_path, index=False)
                start_process = True
            else:
                st.warning("Mohon isi teks terlebih dahulu.")

    # --- MODE 3: SCRAPING STEAM ---
    elif input_mode == "🕷️ Scraping Steam ID":
        st.info("Masukkan App ID dari URL Steam Store.")
        st.caption("Contoh: https://store.steampowered.com/app/**1091500**/Cyberpunk_2077/")
        
        app_id = st.text_input("Steam App ID:", value="1091500")
        limit = st.slider("Jumlah Ulasan diambil:", 10, 500, 50)
        
        if st.button("🕷️ Mulai Scraping & Analisis", key="btn_scrape"):
            if app_id.isdigit():
                # Panggil Fungsi Scraper (Step 00)
                df_scraped = step00.scrape_steam_reviews(app_id, limit=limit)
                
                if not df_scraped.empty:
                    st.success(f"Berhasil mengambil {len(df_scraped)} ulasan!")
                    # Simpan ke Excel RAW
                    df_scraped.to_excel(os.path.join(OUTPUT_DIR, "01_raw.xlsx"), index=False)
                    start_process = True
                else:
                    st.error("Gagal mengambil data atau tidak ada ulasan relevan.")
            else:
                st.warning("App ID harus berupa angka.")

# =====================================================
# MAIN PROCESS LOGIC
# =====================================================
if start_process:
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Definisi Path
    out1 = os.path.join(OUTPUT_DIR, "02_pre_absa.xlsx")
    out2 = os.path.join(OUTPUT_DIR, "03_absa.xlsx")
    out3 = os.path.join(OUTPUT_DIR, "04_post_absa.xlsx")
    out4 = os.path.join(OUTPUT_DIR, "05_labeled.xlsx")
    input_path = os.path.join(OUTPUT_DIR, "01_raw.xlsx") # Path file input

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
            
            # --- CEK APAKAH HASIL KOSONG (Robustness) ---
            if df_final.empty:
                st.warning("⚠️ **Tidak ada aspek game yang terdeteksi.**")
                if input_mode == "✍️ Input Teks Manual":
                    st.info("Tips: Gunakan kata spesifik. Contoh: 'The **graphics** are bad', 'I love the **gameplay**'.")
                st.stop() # Hentikan proses agar grafik tidak error
            
            df_final["label_text"] = df_final["label_text"].str.lower().str.strip()
            
            # Tampilan Tabel
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
        
        # --- FITUR AI COPILOT ---
        # Panggil fungsi generate_smart_insight
        ai_summary = generate_smart_insight(df

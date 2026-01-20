# =====================================================
# FILE: app.py
# Streamlit App – ABSA Steam Review (FINAL VERSION)
# Fitur: Upload, Manual, Scraping, AI Insight, Game Name
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
step00 = load_module(os.path.join(PRE_DIR, "0_steam_scraper.py"), "step00")
step01 = load_module(os.path.join(PRE_DIR, "1_preprocessing_pra_absa.py"), "step01")
step02 = load_module(os.path.join(PRE_DIR, "2_absa_extraction.py"), "step02")
step03 = load_module(os.path.join(PRE_DIR, "3_post_absa_preprocessing.py"), "step03")
step04 = load_module(os.path.join(PRE_DIR, "auto_lebel.py"), "step04")
step05 = load_module(os.path.join(BASE_DIR, "4_sentiment_classification.py"), "step05")

# =====================================================
# AI INSIGHT GENERATOR
# =====================================================
def generate_smart_insight(df, accuracy, game_name=None):
    if df.empty: return "Belum cukup data."
    
    # Tambahkan nama game di insight jika ada
    intro = f"Berdasarkan analisis ulasan untuk **{game_name}**," if game_name else "Berdasarkan analisis ulasan,"
    
    total = len(df)
    pos = (df['label_text'] == 'positive').sum()
    neg = (df['label_text'] == 'negative').sum()
    dominance = "Positif" if pos >= neg else "Negatif"
    percent_dom = (max(pos, neg) / total) * 100 if total > 0 else 0

    aspect_stats = df.groupby('aspect')['label_text'].value_counts().unstack(fill_value=0)
    if 'negative' not in aspect_stats.columns: aspect_stats['negative'] = 0
    if 'positive' not in aspect_stats.columns: aspect_stats['positive'] = 0

    worst_aspect = aspect_stats['negative'].idxmax() if not aspect_stats.empty else "N/A"
    worst_count = aspect_stats.loc[worst_aspect, 'negative'] if not aspect_stats.empty else 0
    best_aspect = aspect_stats['positive'].idxmax() if not aspect_stats.empty else "N/A"
    best_count = aspect_stats.loc[best_aspect, 'positive'] if not aspect_stats.empty else 0

    insight = f"""
    ### 🤖 AI Copilot Summary
    {intro} dari total **{total} data**:
    
    1.  **Sentimen Dominan:** Respon pengguna cenderung **{dominance}** ({percent_dom:.1f}%).
    2.  **Kekuatan Utama:** Aspek **{best_aspect.upper()}** paling diapresiasi ({best_count} positif).
    3.  **Kelemahan Kritis:** Keluhan terbanyak ada pada aspek **{worst_aspect.upper()}** ({worst_count} negatif).
    4.  **Kualitas Model:** Akurasi prediksi: **{accuracy:.1f}%**.
    """
    return insight

# =====================================================
# UI HEADER
# =====================================================
st.title("🎮 Steam Review Analysis")

# --- LOGIC NAMA GAME ---
if 'game_title' not in st.session_state:
    st.session_state['game_title'] = None

if st.session_state['game_title']:
    st.markdown(f"### 🎯 Target Analisis: **{st.session_state['game_title']}**")
else:
    st.write("Analisis Sentimen Berbasis Aspek untuk Ulasan Game")

st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================
start_process = False
uploaded_file = None 

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    input_mode = st.radio("Pilih Sumber Data:", ["📂 Upload Excel", "✍️ Input Teks Manual", "🕷️ Scraping Steam ID"])
    st.markdown("---")
    
    # Mode 1: Upload
    if input_mode == "📂 Upload Excel":
        st.session_state['game_title'] = None
        uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
        if uploaded_file:
            with open(os.path.join(OUTPUT_DIR, "01_raw.xlsx"), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("File Ready!")
            if st.button("🚀 Jalankan Analisis"):
                start_process = True

    # Mode 2: Manual
    elif input_mode == "✍️ Input Teks Manual":
        st.session_state['game_title'] = None
        st.info("Ketik ulasan manual.")
        user_text = st.text_area("Review:", placeholder="Graphics are good but gameplay is bad.")
        if st.button("🚀 Analisis"):
            if user_text.strip():
                pd.DataFrame({"review": [user_text]}).to_excel(os.path.join(OUTPUT_DIR, "01_raw.xlsx"), index=False)
                start_process = True
            else: st.warning("Isi teks dulu.")

    # Mode 3: Scraping
    elif input_mode == "🕷️ Scraping Steam ID":
        st.info("Masukkan App ID Steam.")
        app_id = st.text_input("Steam App ID:", value="1091500")
        limit = st.slider("Target Ulasan:", 10, 500, 50)
        
        if st.button("🕷️ Mulai Scraping"):
            if app_id.isdigit():
                # 1. Ambil Nama Game
                with st.spinner("Mencari info game..."):
                    try:
                        # Pastikan 0_steam_scraper.py sudah punya fungsi get_game_name
                        game_name = step00.get_game_name(app_id)
                        st.session_state['game_title'] = game_name 
                    except AttributeError:
                        st.warning("Fungsi get_game_name tidak ditemukan di scraper. Update 0_steam_scraper.py dulu.")
                        game_name = None
                
                # 2. Scraping Review
                df_scraped = step00.scrape_steam_reviews(app_id, limit=limit)
                
                if not df_scraped.empty:
                    st.success(f"Berhasil mengambil {len(df_scraped)} ulasan!")
                    df_scraped.to_excel(os.path.join(OUTPUT_DIR, "01_raw.xlsx"), index=False)
                    start_process = True
                    st.rerun()
                else:
                    st.error("Gagal mengambil data.")
            else:
                st.warning("ID harus angka.")

# =====================================================
# MAIN PIPELINE
# =====================================================
if start_process:
    progress = st.progress(0)
    status = st.empty()
    
    # Paths
    raw_p = os.path.join(OUTPUT_DIR, "01_raw.xlsx")
    out1 = os.path.join(OUTPUT_DIR, "02_pre_absa.xlsx")
    out2 = os.path.join(OUTPUT_DIR, "03_absa.xlsx")
    out3 = os.path.join(OUTPUT_DIR, "04_post_absa.xlsx")
    out4 = os.path.join(OUTPUT_DIR, "05_labeled.xlsx")

    try:
        status.write("⏳ Preprocessing...")
        progress.progress(10); step01.run(raw_p, out1)
        
        status.write("⏳ Ekstraksi Aspek...")
        progress.progress(30); step02.run(out1, out2)
        
        status.write("⏳ Cleaning Lanjutan...")
        progress.progress(50); step03.run(out2, out3)
        
        status.write("⏳ Pelabelan...")
        progress.progress(70); step04.run(out3, out4)
        
        status.write("⏳ Klasifikasi AI...")
        progress.progress(90); result = step05.run(out4, OUTPUT_DIR)
        
        progress.progress(100); status.success("Selesai!")

        # --- HASIL ---
        if os.path.exists(out4):
            df = pd.read_excel(out4)
            if df.empty:
                st.warning("Tidak ada aspek terdeteksi.")
                st.stop()
            
            df["label_text"] = df["label_text"].str.lower().str.strip()
            
            # Tampilkan Tabel
            with st.expander("📄 Data Hasil Analisis"):
                st.dataframe(df, use_container_width=True)

            st.markdown("---")
            st.subheader("📌 Hasil Analisis")

            # AI Copilot Summary
            game_title = st.session_state.get('game_title', None)
            summary = generate_smart_insight(df, result.get('accuracy', 0)*100, game_title)
            st.info(summary)

            # Metrics
            pos = (df["label_text"] == "positive").sum()
            neg = (df["label_text"] == "negative").sum()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", len(df))
            c2.metric("Positif", pos, delta="Good")
            c3.metric("Negatif", neg, delta="-Bad", delta_color="inverse")
            with open(out4, "rb") as f:
                c4.download_button("⬇ Download Excel", f, "hasil.xlsx")

            st.markdown("---")

            # Grafik
            c_left, c_right = st.columns(2)
            with c_left:
                st.subheader("📊 Distribusi")
                fig1, ax1 = plt.subplots(figsize=(6, 3))
                counts = df["label_text"].value_counts()
                if not counts.empty:
                    counts.plot(kind="barh", ax=ax1, color=['#4CAF50' if x=='positive' else '#F44336' for x in counts.index])
                    st.pyplot(fig1, use_container_width=False)
            
            with c_right:
                st.subheader("🤖 Akurasi Model")
                if result['accuracy'] == 0: st.info("Data < 5 (Akurasi N/A)")
                else: st.write(f"**{result['accuracy']:.2%}**")
                st.table(pd.DataFrame(result["confusion_matrix"], columns=["P_Neg", "P_Pos"], index=["A_Neg", "A_Pos"]))

            # Grafik Aspek
            st.subheader("🧩 Detail Aspek")
            piv = df.groupby(['aspect', 'label_text']).size().unstack(fill_value=0)
            if not piv.empty:
                if 'positive' not in piv: piv['positive']=0
                if 'negative' not in piv: piv['negative']=0
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                piv[['positive', 'negative']].plot(kind='bar', ax=ax2, color=['#4CAF50', '#F44336'], width=0.7)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig2)

    except Exception:
        st.error("Error System")
        st.code(traceback.format_exc())

elif not start_process and not uploaded_file and input_mode == "📂 Upload Excel":
    st.info("Upload file di sidebar.")

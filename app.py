# =====================================================
# FILE: app.py
# VERSI FINAL FIXED: Indentasi & Duplikasi Dibersihkan
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
try:
    step00 = load_module(os.path.join(PRE_DIR, "0_steam_scraper.py"), "step00")
    step01 = load_module(os.path.join(PRE_DIR, "1_preprocessing_pra_absa.py"), "step01")
    step02 = load_module(os.path.join(PRE_DIR, "2_absa_extraction.py"), "step02")
    step03 = load_module(os.path.join(PRE_DIR, "3_post_absa_preprocessing.py"), "step03")
    step04 = load_module(os.path.join(PRE_DIR, "auto_lebel.py"), "step04")
    step05 = load_module(os.path.join(BASE_DIR, "4_sentiment_classification.py"), "step05")
except Exception as e:
    st.warning(f"⚠️ Modul belum lengkap: {e}")

# =====================================================
# FITUR: AI INSIGHT GENERATOR
# =====================================================
def generate_smart_insight(df, accuracy, game_name="Semua Game"):
    if df.empty:
        return "Belum cukup data untuk menyimpulkan."

    # 1. Hitung Statistik
    total = len(df)
    pos = (df['label_text'] == 'positive').sum()
    neg = (df['label_text'] == 'negative').sum()
    
    dominance = "Positif" if pos >= neg else "Negatif"
    percent_dom = (max(pos, neg) / total) * 100 if total > 0 else 0

    # 2. Cari Aspek Terbaik & Terburuk
    aspect_stats = df.groupby('aspect')['label_text'].value_counts().unstack(fill_value=0)
    
    if 'negative' not in aspect_stats.columns: aspect_stats['negative'] = 0
    if 'positive' not in aspect_stats.columns: aspect_stats['positive'] = 0

    worst_aspect = aspect_stats['negative'].idxmax() if not aspect_stats.empty and aspect_stats['negative'].sum() > 0 else "N/A"
    worst_count = aspect_stats.loc[worst_aspect, 'negative'] if worst_aspect != "N/A" else 0
    
    best_aspect = aspect_stats['positive'].idxmax() if not aspect_stats.empty and aspect_stats['positive'].sum() > 0 else "N/A"
    best_count = aspect_stats.loc[best_aspect, 'positive'] if best_aspect != "N/A" else 0

    # 3. Buat Narasi
    insight = f"""
    ### 🤖 AI Summary: {game_name}
    Analisis dari **{total} ulasan**:
    
    1.  **Sentimen Dominan:** Respon mayoritas **{dominance}** ({percent_dom:.1f}%).
    2.  **Kekuatan:** Aspek **{best_aspect.upper()}** paling disukai ({best_count} positif).
    3.  **Kelemahan:** Aspek **{worst_aspect.upper()}** paling banyak dikeluhkan ({worst_count} negatif).
    4.  **Akurasi Model:** **{accuracy:.1f}%**.
    """
    return insight

# =====================================================
# FUNGSI RESET
# =====================================================
def reset_state():
    st.session_state['do_analysis'] = False
    st.session_state['game_title'] = None

# =====================================================
# UI HEADER
# =====================================================
st.title("🎮 Steam Review Analysis Hub")

if 'game_title' not in st.session_state: st.session_state['game_title'] = None
if 'do_analysis' not in st.session_state: st.session_state['do_analysis'] = False

st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================
uploaded_file = None 

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    input_mode = st.radio(
        "Pilih Sumber Data:", 
        ["📂 Upload Excel", "✍️ Input Teks Manual", "🕷️ Scraping Steam ID"],
        on_change=reset_state
    )
    
    st.markdown("---")
    
    # --- MODE 1: UPLOAD EXCEL ---
    if input_mode == "📂 Upload Excel":
        uploaded_file = st.file_uploader("Upload file Excel/CSV", type=["xlsx", "csv"])
        if uploaded_file:
            file_path = os.path.join(OUTPUT_DIR, "01_raw.xlsx")
            if uploaded_file.name.endswith('.csv'):
                df_temp = pd.read_csv(uploaded_file)
                df_temp.to_excel(file_path, index=False)
            else:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            st.success("File Terupload!")
            if st.button("🚀 Jalankan Analisis", key="btn_upload"):
                reset_state()
                st.session_state['do_analysis'] = True

    # --- MODE 2: INPUT MANUAL ---
    elif input_mode == "✍️ Input Teks Manual":
        user_text = st.text_area("Masukkan Review:", height=150)
        if st.button("🚀 Analisis Teks", key="btn_manual"):
            if user_text.strip():
                # Dummy AppID 999999 untuk manual
                df_manual = pd.DataFrame({"review": [user_text], "appid": [999999]}) 
                df_manual.to_excel(os.path.join(OUTPUT_DIR, "01_raw.xlsx"), index=False)
                reset_state()
                st.session_state['do_analysis'] = True

    # --- MODE 3: SCRAPING STEAM ---
    elif input_mode == "🕷️ Scraping Steam ID":
        app_id = st.text_input("Steam App ID:", value="1091500")
        limit = st.slider("Jumlah:", 10, 2000, 50)
        
        if st.button("🕷️ Mulai Scraping", key="btn_scrape"):
            reset_state()
            with st.spinner("Mengambil data..."):
                try:
                    game_name = step00.get_game_name(app_id)
                    st.session_state['game_title'] = game_name
                except:
                    st.session_state['game_title'] = f"AppID {app_id}"

                df_scraped = step00.scrape_steam_reviews(app_id, limit=limit)
                if not df_scraped.empty:
                    st.success(f"Dapat {len(df_scraped)} ulasan!")
                    df_scraped.to_excel(os.path.join(OUTPUT_DIR, "01_raw.xlsx"), index=False)
                    st.session_state['do_analysis'] = True 
                    st.rerun()
                else:
                    st.error("Gagal mengambil data.")

# =====================================================
# MAIN PROCESS LOGIC
# =====================================================
if st.session_state['do_analysis']:
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Define Paths
    files = {
        "raw": os.path.join(OUTPUT_DIR, "01_raw.xlsx"),
        "pre": os.path.join(OUTPUT_DIR, "02_pre_absa.xlsx"),
        "absa": os.path.join(OUTPUT_DIR, "03_absa.xlsx"),
        "post": os.path.join(OUTPUT_DIR, "04_post_absa.xlsx"),
        "label": os.path.join(OUTPUT_DIR, "05_labeled.xlsx")
    }

    try:
        if not os.path.exists(files["raw"]):
            st.error("File input hilang.")
            st.stop()

        # EKSEKUSI PIPELINE
        status_text.write("⏳ Step 1/5: Cleaning Awal...")
        progress_bar.progress(10); step01.run(files["raw"], files["pre"])
        
        status_text.write("⏳ Step 2/5: Ekstraksi Aspek...")
        progress_bar.progress(30); step02.run(files["pre"], files["absa"])
        
        status_text.write("⏳ Step 3/5: Cleaning Lanjutan...")
        progress_bar.progress(50); step03.run(files["absa"], files["post"])
        
        status_text.write("⏳ Step 4/5: Auto Labeling...")
        progress_bar.progress(70); step04.run(files["post"], files["label"])
        
        status_text.write("⏳ Step 5/5: Evaluasi Model...")
        progress_bar.progress(90); result = step05.run(files["label"], OUTPUT_DIR)

        progress_bar.progress(100)
        status_text.success("✅ Analisis Selesai!")

        # =====================================================
        # BAGIAN: VISUALISASI DATA
        # =====================================================
        if os.path.exists(files["label"]):
            df_final = pd.read_excel(files["label"])
            df_final["label_text"] = df_final["label_text"].str.lower().str.strip()

            # --- 1. FITUR FILTER GAME ID (Multigame Support) ---
            # Cari kolom yang mengandung ID
            game_id_col = None
            possible_cols = ['appid', 'app_id', 'game_id', 'steam_appid']
            for c in possible_cols:
                if c in df_final.columns:
                    game_id_col = c
                    break
            
            # Default: Semua Data
            df_active = df_final.copy()
            selected_game_label = "Semua Game"

            # Jika ada lebih dari 1 game, munculkan filter
            if game_id_col and len(df_final[game_id_col].unique()) > 1:
                st.markdown("### 🎯 Filter Analisis")
                col_filter, _ = st.columns([1, 2])
                with col_filter:
                    unique_games = sorted(list(df_final[game_id_col].unique()))
                    selected_game_id = st.selectbox(
                        "Pilih Game ID:", 
                        ["Gabungan (Semua)"] + unique_games
                    )
                
                if selected_game_id != "Gabungan (Semua)":
                    df_active = df_final[df_final[game_id_col] == selected_game_id]
                    selected_game_label = f"Game ID {selected_game_id}"

            # --- 2. DASHBOARD ---
            st.markdown(f"## 📊 Laporan: {selected_game_label}")
            
            # AI Insight
            acc_score = result.get('accuracy', 0) * 100
            ai_summary = generate_smart_insight(df_active, acc_score, selected_game_label)
            st.info(ai_summary)

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Ulasan", len(df_active))
            col2.metric("Positif", (df_active["label_text"] == "positive").sum())
            col3.metric("Negatif", (df_active["label_text"] == "negative").sum())
            
            # Download Button
            with col4:
                st.write("") 
                from io import BytesIO
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_active.to_excel(writer, index=False)
                
                st.download_button(
                    label="⬇ Download Excel",
                    data=output.getvalue(),
                    file_name=f"analisis_{selected_game_label}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            st.markdown("---")
            
            # --- 3. CHART SENTIMEN PER ASPEK (DENGAN LABEL ANGKA) ---
            st.subheader(f"🧩 Detail Aspek: {selected_game_label}")
            
            aspect_sentiment = df_active.groupby(['aspect', 'label_text']).size().unstack(fill_value=0)
            
            if not aspect_sentiment.empty:
                if 'positive' not in aspect_sentiment.columns: aspect_sentiment['positive'] = 0
                if 'negative' not in aspect_sentiment.columns: aspect_sentiment['negative'] = 0
                
                # Plotting
                fig, ax = plt.subplots(figsize=(10, 5))
                aspect_sentiment[['positive', 'negative']].plot(
                    kind='bar', ax=ax, color=['#4CAF50', '#F44336'], width=0.7
                )
                
                # === LABEL ANGKA DI ATAS BATANG ===
                for container in ax.containers:
                    ax.bar_label(container, fmt='%d', padding=3, fontsize=10)
                
                # Atur jarak atas agar angka tidak kepotong
                ax.set_ylim(0, aspect_sentiment.values.max() * 1.2)
                
                ax.set_ylabel("Jumlah Ulasan")
                ax.set_xlabel("Aspek Game")
                ax.legend(["Positif", "Negatif"])
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig)
            else:
                st.warning("Data aspek kosong.")

            # --- 4. DATA RAW ---
            with st.expander("📄 Lihat Data Mentah"):
                st.dataframe(df_active, use_container_width=True)

    except Exception:
        st.error("❌ Terjadi error sistem")
        st.code(traceback.format_exc())
    
    if st.button("🔄 Reset / Analisis Baru"):
        reset_state()
        st.rerun()

elif not uploaded_file and input_mode == "📂 Upload Excel":
    st.info("👈 Silakan upload file Excel/CSV di menu sebelah kiri.")

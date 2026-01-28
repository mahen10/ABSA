# =====================================================
# FILE: app.py
# Streamlit App – ABSA Steam Review (AUTO-RESET FIXED)
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
step00 = load_module(os.path.join(PRE_DIR, "0_steam_scraper.py"), "step00")
step01 = load_module(os.path.join(PRE_DIR, "1_preprocessing_pra_absa.py"), "step01")
step02 = load_module(os.path.join(PRE_DIR, "2_absa_extraction.py"), "step02")
step03 = load_module(os.path.join(PRE_DIR, "3_post_absa_preprocessing.py"), "step03")
step04 = load_module(os.path.join(PRE_DIR, "auto_lebel.py"), "step04")
step05 = load_module(os.path.join(BASE_DIR, "4_sentiment_classification.py"), "step05")


# =====================================================
# DATABASE GAME ID -> NAMA GAME
# =====================================================
GAME_DATABASE = {
    269790: "DreadOut",
    854570: "Pamali: Indonesian Folklore Horror",
    1201270: "A Space for the Unbound",
    1337760: "Potion Permit",
    1356280: "Kitaria Fables"
}

def get_game_name_from_id(game_id):
    """Ambil nama game dari ID, fallback ke ID jika tidak ada di database"""
    try:
        game_id_int = int(game_id)
        return GAME_DATABASE.get(game_id_int, f"Game ID {game_id}")
    except (ValueError, TypeError):
        return f"Game ID {game_id}"

# =====================================================
# FITUR: AI INSIGHT GENERATOR (LOGIC BASED)
# =====================================================
def generate_smart_insight(df, accuracy, game_name=None):
    if df.empty:
        return "Belum cukup data untuk menyimpulkan."

    # Intro Custom
    intro = f"Berdasarkan analisis ulasan untuk **{game_name}**," if game_name else "Berdasarkan analisis ulasan,"

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

    # 3. Rangkai Kalimat
    insight = f"""
    ### Hasil Analisis
    {intro} dari total **{total} data**:
    
    1. **Sentimen Dominan:** Mayoritas pengguna memberikan respon **{dominance}** ({percent_dom:.1f}%).
    2. **Kekuatan Utama:** Aspek **{best_aspect.upper()}** paling banyak dipuji (mendapat {best_count} respon positif). Ini adalah fitur unggulan game ini.
    3. **Kelemahan Kritis:** Pengguna paling banyak mengeluh soal **{worst_aspect.upper()}** (mendapat {worst_count} keluhan). Developer disarankan untuk segera memperbaiki sektor ini.
    4. **Kualitas Model:** Analisis ini didukung oleh model Logistik Regresion dengan tingkat akurasi **{accuracy:.1f}%**.
    """
    return insight

# =====================================================
# FUNGSI RESET (SOLUSI MASALAH ANDA)
# =====================================================
def reset_state():
    """Fungsi ini dipanggil otomatis saat user ganti menu"""
    st.session_state['do_analysis'] = False
    st.session_state['game_title'] = None
    # Kita tidak menghapus file output agar tidak error, tapi kita reset trigger-nya

# =====================================================
# UI HEADER
# =====================================================
st.title("🎮 Steam Review Analysis")

# --- STATE MANAGEMENT ---
if 'game_title' not in st.session_state:
    st.session_state['game_title'] = None
if 'do_analysis' not in st.session_state:
    st.session_state['do_analysis'] = False

# Tampilkan Nama Game jika ada di memori
if st.session_state['game_title']:
    st.markdown(f"### 🎯 Target Analisis: **{st.session_state['game_title']}**")

st.markdown("---")

# =====================================================
# SIDEBAR (UPDATE INI)
# =====================================================
uploaded_file = None

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    # 1. Checkbox Class Weight (Yang baru ditambahkan)
    # ------------------------------------------------
    use_cw = st.checkbox("⚖️ Pakai Class Weight (Balanced)?", value=True)
    st.caption("Matikan untuk melihat hasil murni tanpa penyeimbangan data.")
    
    st.markdown("---")
    
    # 2. Radio Button (Pastikan formatnya persis seperti ini)
    # -----------------------------------------------------
    input_mode = st.radio(
        "Pilih Sumber Data:",
        ["📂 Upload Excel", "✍️ Input Teks Manual", "🕷️ Scraping Steam ID"],
        on_change=reset_state  # Pastikan ada koma di baris sebelumnya
    )
    
    st.markdown("---")
    
    # --- MODE 1: UPLOAD EXCEL ---
    if input_mode == "📂 Upload Excel":
        uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
        if uploaded_file:
            with open(os.path.join(OUTPUT_DIR, "01_raw.xlsx"), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("File Terupload!")
            
            # Tombol Run
            if st.button("🚀 Jalankan Analisis", key="btn_upload"):
                reset_state()
                st.session_state['do_analysis'] = True

    # --- MODE 2: INPUT MANUAL ---
    elif input_mode == "✍️ Input Teks Manual":
        st.info("Ketik ulasan game di bawah ini.")
        with st.expander("ℹ️ Tips: Gunakan kata kunci ini!"):
            st.markdown("""
            * 🎨 **Graphics:** graphics, graphic, visual, visuals, ui, gui, art, artstyle, resolution, texture, animation, lighting, shadow, design, scenery, environment
            * ⚔️ **Gameplay:** gameplay, control, controls, mechanic, mechanics, combat, movement, system, feature, features, action, battle, attack, defend, quest, quests, level, enemy, boss, difficulty
            * 📜 **Story:** story, plot, narrative, lore, writing, dialogue, ending, cutscene, mission, twist, script, character, development, storyline, arc, pacing
            * 🚀 **Performance:** performance, fps, frame, rate, optimization, memory, rendering, loading, server, connection, ping, latency, bug, glitch, crash, freeze
            * 🎵 **Music:** music, sound, audio, sfx, voice, acting, soundtrack, ost, bgm, volume, melody, song, noise, dubbing
            """)
        user_text = st.text_area("Masukkan Review Game:", height=150)
        
        if st.button("🚀 Analisis Teks", key="btn_manual"):
            if user_text.strip():
                df_manual = pd.DataFrame({"review": [user_text]})
                input_path = os.path.join(OUTPUT_DIR, "01_raw.xlsx")
                df_manual.to_excel(input_path, index=False)
                
                reset_state()
                st.session_state['do_analysis'] = True
            else:
                st.warning("Mohon isi teks terlebih dahulu.")

    # --- MODE 3: SCRAPING STEAM ---
    elif input_mode == "🕷️ Scraping Steam ID":
        st.info("Masukkan App ID dari URL Steam Store.")
        app_id = st.text_input("Steam App ID:", value="1091500")
        limit = st.slider("Jumlah Ulasan diambil:", 10, 2000, 50)
        
        if st.button("🕷️ Mulai Scraping & Analisis", key="btn_scrape"):
            if app_id.isdigit():
                reset_state()
                with st.spinner("Mencari info game..."):
                    try:
                        game_name = step00.get_game_name(app_id)
                        st.session_state['game_title'] = game_name
                    except AttributeError:
                        st.session_state['game_title'] = f"Game ID {app_id}"

                df_scraped = step00.scrape_steam_reviews(app_id, limit=limit)
                
                if not df_scraped.empty:
                    st.success(f"Berhasil mengambil {len(df_scraped)} ulasan!")
                    df_scraped.to_excel(os.path.join(OUTPUT_DIR, "01_raw.xlsx"), index=False)
                    
                    st.session_state['do_analysis'] = True
                    st.rerun()
                else:
                    st.error("Gagal mengambil data atau tidak ada ulasan relevan.")
            else:
                st.warning("App ID harus berupa angka.")
# =====================================================
# MAIN PROCESS LOGIC (JALAN JIKA STATE TRUE)
# =====================================================
if st.session_state['do_analysis']:
    progress_bar = st.progress(0)
    status_text = st.empty()

    out1 = os.path.join(OUTPUT_DIR, "02_pre_absa.xlsx")
    out2 = os.path.join(OUTPUT_DIR, "03_absa.xlsx")
    out3 = os.path.join(OUTPUT_DIR, "04_post_absa.xlsx")
    out4 = os.path.join(OUTPUT_DIR, "05_labeled.xlsx")
    input_path = os.path.join(OUTPUT_DIR, "01_raw.xlsx")

    try:
        if not os.path.exists(input_path):
            st.error("File input hilang. Silakan ulangi proses.")
            st.session_state['do_analysis'] = False
            st.stop()

        status_text.write("⏳ Step 1/5: Preprocessing...")
        progress_bar.progress(10); step01.run(input_path, out1)
        
        status_text.write("⏳ Step 2/5: Ekstraksi Aspek (ABSA)...")
        progress_bar.progress(30); step02.run(out1, out2)
        
        status_text.write("⏳ Step 3/5: Cleaning Lanjutan...")
        progress_bar.progress(50); step03.run(out2, out3)
        
        status_text.write("⏳ Step 4/5: Pelabelan Otomatis...")
        progress_bar.progress(70); step04.run(out3, out4)
        

        status_text.write("⏳ Step 5/5: Klasifikasi & Evaluasi Model...")
        
        # --- UBAH CARA PANGGIL step05 ---
        # Kita kirim nilai dari checkbox (use_cw) ke fungsi run
        progress_bar.progress(90)
        result = step05.run(out4, OUTPUT_DIR, use_balanced=use_cw) 

        progress_bar.progress(100)
        status_text.success("✅ Analisis Selesai!")

        # =====================================================
        # BAGIAN: VISUALISASI DATA DENGAN MULTI-GAME FILTER
        # =====================================================
        if os.path.exists(out4):
            df_final = pd.read_excel(out4)
            
            if df_final.empty:
                st.warning("⚠️ **Tidak ada aspek game yang terdeteksi.**")
                st.session_state['do_analysis'] = False
                st.stop()
            
            df_final["label_text"] = df_final["label_text"].str.lower().str.strip()
            
            # =====================================================
            # FITUR FILTER GAME ID (Multigame Support)
            # =====================================================
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
                st.markdown("### 🎯 Filter Analisis Multi-Game")
                
                col_filter, col_info = st.columns([1, 2])
                
                with col_filter:
                    unique_games = sorted([str(x) for x in df_final[game_id_col].unique()])
                    
                    # Buat dictionary mapping ID -> Label yang cantik
                    game_options_dict = {"all": "🌐 Gabungan (Semua Game)"}
                    for gid in unique_games:
                        game_name = get_game_name_from_id(gid)
                        game_options_dict[gid] = f"🎮 {game_name}"
                    
                    # Opsi untuk selectbox
                    game_options = list(game_options_dict.values())
                    
                    selected_option = st.selectbox(
                        "Pilih Game untuk Dianalisis:",
                        game_options,
                        key="game_filter"
                    )
                    
                    if selected_option != "🌐 Gabungan (Semua Game)":
                        # Cari game ID dari pilihan
                        selected_game_id = None
                        for gid, label in game_options_dict.items():
                            if label == selected_option:
                                selected_game_id = gid
                                break
                        
                        if selected_game_id and selected_game_id != "all":
                            df_active = df_final[df_final[game_id_col].astype(str) == selected_game_id]
                            selected_game_label = get_game_name_from_id(selected_game_id)
                
                with col_info:
                    # Tampilkan ringkasan game yang tersedia dengan nama
                    game_summary = df_final.groupby(game_id_col).size().reset_index(name='Jumlah Review')
                    game_summary['Nama Game'] = game_summary[game_id_col].apply(get_game_name_from_id)
                    game_summary = game_summary[[game_id_col, 'Nama Game', 'Jumlah Review']]
                    game_summary.columns = ['Game ID', 'Nama Game', 'Jumlah Review']
                    
                    with st.expander("📊 Lihat Ringkasan Semua Game"):
                        st.dataframe(game_summary, use_container_width=True)
                
                st.markdown("---")
            
            # =====================================================
            # PREVIEW TABEL DATA (Berdasarkan Filter)
            # =====================================================
            st.markdown(f"### 🔍 Deteksi Aspek & Sentimen - **{selected_game_label}**")
            
            # --- TAMPILKAN KOLOM PILIHAN ---
            desired_cols = ["original_review", "opinion_context", "aspect", "label_text"]
            if game_id_col:
                desired_cols.insert(0, game_id_col)
            
            display_cols = [col for col in desired_cols if col in df_active.columns]
            
            with st.expander("📄 Lihat Data Hasil Analisis"):
                st.dataframe(df_active[display_cols], use_container_width=True)
            
            st.markdown("---")

            # =====================================================
            # VISUALISASI DASHBOARD (Berdasarkan Filter)
            # =====================================================
            st.subheader(f"📌 Ringkasan Hasil - {selected_game_label}")
            
            game_title = st.session_state.get('game_title', None) or selected_game_label
            acc_score = result.get('accuracy', 0) * 100
            ai_summary = generate_smart_insight(df_active, acc_score, game_title)
            st.info(ai_summary)
            
            pos = (df_active["label_text"] == "positive").sum()
            neg = (df_active["label_text"] == "negative").sum()
            total = len(df_active)
            
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

            c_left, c_right = st.columns([1, 1])

            with c_left:
                st.subheader("📊 Distribusi Sentimen")
                fig1, ax1 = plt.subplots(figsize=(6, 3))
                counts = df_active["label_text"].value_counts()
                
                if not counts.empty:
                    colors = ['#4CAF50' if x == 'positive' else '#F44336' for x in counts.index]
                    counts.plot(kind="barh", ax=ax1, color=colors, width=0.6)
                    ax1.set_xlabel("Jumlah")
                    st.pyplot(fig1, use_container_width=False)
                else:
                    st.write("Tidak ada data.")

           
            
            # =====================================================
            # ANALISIS DETAIL PER ASPEK (Berdasarkan Filter)
            # =====================================================
            st.subheader(f"🧩 Analisis Detail Per Aspek - {selected_game_label}")
            
            aspect_sentiment = df_active.groupby(['aspect', 'label_text']).size().unstack(fill_value=0)
            
            if not aspect_sentiment.empty:
                # Pastikan kolom ada
                if 'positive' not in aspect_sentiment.columns: aspect_sentiment['positive'] = 0
                if 'negative' not in aspect_sentiment.columns: aspect_sentiment['negative'] = 0
                
                # Buat Plot
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                aspect_sentiment[['positive', 'negative']].plot(
                    kind='bar', ax=ax2, color=['#4CAF50', '#F44336'], width=0.7
                )

                # Menampilkan angka di atas bar
                for container in ax2.containers:
                    ax2.bar_label(container, fmt='%d', padding=3, fontsize=10)
                
                # Tambahkan ruang kosong di atas chart
                y_max = aspect_sentiment.values.max()
                ax2.set_ylim(0, y_max * 1.2)

                ax2.set_ylabel("Jumlah")
                ax2.set_xlabel("Aspek")
                ax2.legend(title="Sentimen")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig2)
            else:
                st.info("Belum cukup data aspek.")

    except Exception:
        st.error("❌ Terjadi error sistem")
        st.code(traceback.format_exc())
        
    # Tombol Reset di bawah
    if st.button("🔄 Reset / Analisis Baru"):
        st.session_state['do_analysis'] = False
        st.rerun()

elif not uploaded_file and input_mode == "📂 Upload Excel":
    st.info("👈 Silakan upload file Excel di menu sebelah kiri.")


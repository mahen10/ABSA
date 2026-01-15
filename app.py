# =====================================================
# FILE: app.py
# Deskripsi:
# Streamlit App – Pipeline ABSA Game Steam
# =====================================================

import streamlit as st
import os
import sys
import traceback

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
    """Load module dengan error handling yang lebih baik"""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File tidak ditemukan: {path}")
        
        # Tambahkan direktori ke sys.path agar import relatif bisa bekerja
        module_dir = os.path.dirname(path)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        
        # Import module sebagai regular Python module
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Tidak bisa load spec untuk {path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        
        return module
        
    except Exception as e:
        st.error(f"Error loading module {name} dari {path}")
        st.exception(e)
        raise


# =====================================================
# LOAD PIPELINE MODULES
# =====================================================
try:
    st.info("Loading modules...")
    
    step01 = load_module(
        os.path.join(PRE_DIR, "1_preprocessing_pra_absa.py"),
        "step01"
    )
    st.success("✓ Module 1 loaded")
    
    step02 = load_module(
        os.path.join(PRE_DIR, "2_absa_extraction.py"),
        "step02"
    )
    st.success("✓ Module 2 loaded")
    
    step03 = load_module(
        os.path.join(PRE_DIR, "3_post_absa_preprocessing.py"),
        "step03"
    )
    st.success("✓ Module 3 loaded")
    
    step04 = load_module(
        os.path.join(PRE_DIR, "auto_lebel.py"),
        "step04"
    )
    st.success("✓ Module 4 loaded")
    
    step05 = load_module(
        os.path.join(BASE_DIR, "4_sentiment_classification.py"),
        "step05"
    )
    st.success("✓ Module 5 loaded")
    
except Exception as e:
    st.error("⚠️ Error saat loading modules")
    st.exception(e)
    st.stop()


# =====================================================
# UI
# =====================================================
st.title("🎮 Aspect-Based Sentiment Analysis")
st.write("Pipeline Analisis Sentimen Ulasan Game Steam")

uploaded_file = st.file_uploader(
    "Upload file Excel hasil scraping",
    type=["xlsx"]
)

if uploaded_file:
    input_path = os.path.join(OUTPUT_DIR, "01_raw.xlsx")
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File berhasil diupload")

    if st.button("🚀 Jalankan Pipeline"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Preprocessing Pra-ABSA
            status_text.text("Step 1/5: Preprocessing Pra-ABSA...")
            progress_bar.progress(0.1)
            out1 = os.path.join(OUTPUT_DIR, "02_pra_absa.xlsx")
            step01.run(input_path, out1)
            progress_bar.progress(0.2)
            st.success("✓ Step 1 selesai")

            # Step 2: Ekstraksi ABSA
            status_text.text("Step 2/5: Ekstraksi ABSA...")
            progress_bar.progress(0.3)
            out2 = os.path.join(OUTPUT_DIR, "03_absa.xlsx")
            step02.run(out1, out2)
            progress_bar.progress(0.4)
            st.success("✓ Step 2 selesai")

            # Step 3: Post-ABSA Preprocessing
            status_text.text("Step 3/5: Post-ABSA Preprocessing...")
            progress_bar.progress(0.5)
            out3 = os.path.join(OUTPUT_DIR, "04_post_absa.xlsx")
            step03.run(out2, out3)
            progress_bar.progress(0.6)
            st.success("✓ Step 3 selesai")

            # Step 4: Auto Label Sentimen
            status_text.text("Step 4/5: Auto Label Sentimen...")
            progress_bar.progress(0.7)
            out4 = os.path.join(OUTPUT_DIR, "05_labeled.xlsx")
            step04.run(out3, out4)
            progress_bar.progress(0.8)
            st.success("✓ Step 4 selesai")

            # Step 5: Klasifikasi Sentimen
            status_text.text("Step 5/5: Klasifikasi Sentimen...")
            progress_bar.progress(0.9)
            out5 = os.path.join(OUTPUT_DIR, "06_classification.xlsx")
            step05.run(out4, out5)
            progress_bar.progress(1.0)
            st.success("✓ Step 5 selesai")

            status_text.text("✅ Pipeline selesai!")

            # Download button
            with open(out5, "rb") as f:
                st.download_button(
                    label="⬇ Download Hasil Akhir",
                    data=f,
                    file_name="hasil_absa.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error("❌ Terjadi error saat menjalankan pipeline")
            st.error(f"Error: {str(e)}")
            
            # Tampilkan traceback lengkap
            with st.expander("Lihat detail error"):
                st.code(traceback.format_exc())
            
            st.info("💡 Tip: Periksa format file Excel dan pastikan kolom 'review' ada")

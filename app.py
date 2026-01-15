import streamlit as st
import pandas as pd
import os
import importlib.util
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ===============================
# PATH
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRE_DIR = os.path.join(BASE_DIR, "Preprocessing")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATASET_PATH = os.path.join(OUTPUT_DIR, "DataSet.xlsx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# SAFE MODULE LOADER
# ===============================
def load_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="ABSA Game Review Dashboard",
    layout="wide"
)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("📊 Menu")
menu = st.sidebar.radio(
    "Navigasi",
    ["Dashboard", "Upload Data", "Data Tersimpan", "Analisis & Hasil"]
)

# ===============================
# DASHBOARD
# ===============================
if menu == "Dashboard":
    st.title("🎮 Dashboard ABSA Game Review")

    if os.path.exists(DATASET_PATH):
        df = pd.read_excel(DATASET_PATH)
        st.metric("Total Ulasan", len(df))
    else:
        st.metric("Total Ulasan", 0)

# ===============================
# UPLOAD DATA
# ===============================
elif menu == "Upload Data":
    st.title("📥 Upload Dataset")

    uploaded = st.file_uploader("Upload Excel (kolom: review)", type=["xlsx"])
    if uploaded:
        df = pd.read_excel(uploaded)
        if "review" not in df.columns:
            st.error("Kolom 'review' wajib ada")
        else:
            df.to_excel(DATASET_PATH, index=False)
            st.success("Dataset disimpan")
            st.dataframe(df.head())

# ===============================
# DATA TERSIMPAN
# ===============================
elif menu == "Data Tersimpan":
    if os.path.exists(DATASET_PATH):
        st.dataframe(pd.read_excel(DATASET_PATH))
    else:
        st.warning("Dataset belum ada")

# ===============================
# ANALISIS & HASIL
# ===============================
elif menu == "Analisis & Hasil":
    st.title("⚙️ Proses Analisis")

    if not os.path.exists(DATASET_PATH):
        st.error("Dataset belum diupload")
        st.stop()

    if st.button("🚀 Jalankan Pipeline"):
        with st.spinner("Menjalankan pipeline..."):

            # STEP 1
            step1 = load_module(os.path.join(PRE_DIR, "1_preprocessing_pra_absa.py"))
            step1.run(DATASET_PATH, OUTPUT_DIR)

            # STEP 2 (ABSA — AMAN)
            step2 = load_module(os.path.join(PRE_DIR, "2_absa_extraction.py"))
            step2.run(
                input_path=os.path.join(OUTPUT_DIR, "cleaned_reviews.xlsx"),
                output_dir=OUTPUT_DIR
            )

            # STEP 3
            step3 = load_module(os.path.join(PRE_DIR, "3_post_absa_preprocessing.py"))
            step3.run(OUTPUT_DIR)

            # STEP 4
            step4 = load_module(os.path.join(PRE_DIR, "auto_lebel.py"))
            step4.run(OUTPUT_DIR)

            # STEP 5
            step5 = load_module(os.path.join(PRE_DIR, "4_sentiment_classification.py"))
            step5.run(OUTPUT_DIR)

        st.success("✅ Analisis selesai")

    # ===== HASIL =====
    absa_path = os.path.join(OUTPUT_DIR, "absa_output.xlsx")
    if os.path.exists(absa_path):
        df = pd.read_excel(absa_path)
        st.dataframe(df.head(50))

        st.subheader("Distribusi Aspek")
        df["aspect"].value_counts().plot(kind="bar")
        st.pyplot(plt.gcf())

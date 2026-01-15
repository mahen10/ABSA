import streamlit as st
import pandas as pd
import os
import importlib.util
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import nltk

# ===============================
# SAFE NLTK DOWNLOAD
# ===============================
@st.cache_resource
def download_nltk():
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("stopwords")

download_nltk()

# ===============================
# LOAD MODULE DARI FILE (AMAN)
# ===============================
st.write("BASE_DIR:", os.getcwd())
st.write("Isi folder BASE_DIR:", os.listdir(os.getcwd()))

def load_module(path, name):
    if not os.path.exists(path):
        st.error(f"❌ File tidak ditemukan: {path}")
        st.stop()

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ===============================
# PATH
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRE_DIR = os.path.join(BASE_DIR, "Preprocessing")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATASET_PATH = os.path.join(OUTPUT_DIR, "DataSet.xlsx")


os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# LOAD PIPELINE
# ===============================
step02 = load_module(os.path.join(PRE_DIR, "1_preprocessing_pra_absa.py"), "step02")
step03 = load_module(os.path.join(PRE_DIR, "2_absa_extraction.py"), "step03")
step04 = load_module(os.path.join(PRE_DIR, "3_post_absa_preprocessing.py"), "step04")
step05 = load_module(os.path.join(PRE_DIR, "auto_lebel.py"), "step05")
step06 = load_module(os.path.join(PRE_DIR, "4_sentiment_classification.py"), "step06")

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

    st.markdown("""
    Aplikasi ini digunakan untuk melakukan **Aspect-Based Sentiment Analysis (ABSA)**
    pada ulasan game Steam menggunakan pendekatan **Rule-Based ABSA**
    dan **Logistic Regression**.
    """)

    col1, col2, col3 = st.columns(3)

    if os.path.exists(DATASET_PATH):
        df = pd.read_excel(DATASET_PATH)
        col1.metric("Total Ulasan", len(df))
    else:
        col1.metric("Total Ulasan", 0)

    col2.metric("Aspek", 5)
    col3.metric("Model", "Logistic Regression")

# ===============================
# UPLOAD DATA
# ===============================
elif menu == "Upload Data":
    st.title("📥 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload file Excel (harus ada kolom 'review')",
        type=["xlsx"]
    )

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        if "review" not in df.columns:
            st.error("❌ Kolom 'review' wajib ada")
        else:
            df.to_excel(DATASET_PATH, index=False)
            st.success("✅ Dataset berhasil disimpan")
            st.dataframe(df.head())

# ===============================
# DATA TERSIMPAN
# ===============================
elif menu == "Data Tersimpan":
    st.title("📂 Data Tersimpan")

    if os.path.exists(DATASET_PATH):
        df = pd.read_excel(DATASET_PATH)
        st.dataframe(df)
    else:
        st.warning("⚠️ Dataset belum tersedia")

# ===============================
# ANALISIS & HASIL
# ===============================
elif menu == "Analisis & Hasil":
    st.title("⚙️ Proses Analisis Sentimen")

    if not os.path.exists(DATASET_PATH):
        st.error("❌ Dataset belum diupload")
        st.stop()

    if st.button("🚀 Lakukan Analisis"):
        with st.spinner("Menjalankan pipeline penelitian..."):
            step02.main()
            step03.main()
            step04.main()
            step05.main()
            step06.main()

        st.success("✅ Analisis selesai")

    # ===============================
    # HASIL ABSA
    # ===============================
    absa_path = os.path.join(OUTPUT_DIR, "absa_output.xlsx")
    if os.path.exists(absa_path):
        st.subheader("📌 Hasil ABSA")
        df_absa = pd.read_excel(absa_path)
        st.dataframe(df_absa.head(50))

        st.subheader("📊 Distribusi Aspek")
        aspect_counts = df_absa["aspect"].value_counts()

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            aspect_counts.plot(kind="bar", ax=ax)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            aspect_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
            st.pyplot(fig)

    # ===============================
    # DISTRIBUSI SENTIMEN
    # ===============================
    labeled_path = os.path.join(OUTPUT_DIR, "absa_labeled_numeric.xlsx")
    if os.path.exists(labeled_path):
        st.subheader("📈 Distribusi Sentimen")
        df_label = pd.read_excel(labeled_path)
        sent_counts = df_label["label_text"].value_counts()

        fig, ax = plt.subplots()
        sent_counts.plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # ===============================
    # CONFUSION MATRIX
    # ===============================
    if os.path.exists(labeled_path):
        st.subheader("🧪 Confusion Matrix")

        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer

        df = pd.read_excel(labeled_path)
        df = df.dropna(subset=["processed_opinion", "label_text"])

        X = df["processed_opinion"]
        y = df["label_text"]

        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_df=0.95,
            min_df=2
        )

        X_tfidf = tfidf.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear"
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)



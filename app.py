import streamlit as st
import pandas as pd
import os
import subprocess
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import nltk
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")


st.set_page_config(
    page_title="ABSA Game Review Dashboard",
    layout="wide"
)

# ===============================
# PATH
# ===============================
OUTPUT_DIR = "output"
DATASET_PATH = os.path.join("Output", "DataSet.xlsx")

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
        "Upload file Excel (harus ada kolom 'review' dan 'voted')",
        type=["xlsx"]
    )

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        if "review" not in df.columns:
            st.error("❌ Kolom 'review' wajib ada")
        else:
            os.makedirs("Output", exist_ok=True)
            df.to_excel(DATASET_PATH, index=False)
            st.success("✅ File berhasil disimpan")

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
        st.warning("Belum ada data")

# ===============================
# ANALISIS & HASIL
# ===============================
elif menu == "Analisis & Hasil":
    st.title("⚙️ Proses Analisis Sentimen")

    if st.button("🚀 Lakukan Analisis"):
        with st.spinner("Menjalankan seluruh pipeline..."):

            scripts = [
                "Preprocessing/02_preprocessing_pra_absa.py",
                "Preprocessing/03_absa_extraction_excel_FINAL.py",
                "Preprocessing/04_post_absa_preprocessing_excel_split.py",
                "Preprocessing/05_auto_label_sentiment.py",
                "Preprocessing/4_sentiment_classification.py"
            ]

            for script in scripts:
                subprocess.run(["python", script], check=True)

        st.success("✅ Analisis selesai")

    # ===============================
    # TAMPILKAN HASIL ABSA
    # ===============================
    absa_path = os.path.join("output", "absa_output.xlsx")
    if os.path.exists(absa_path):
        st.subheader("📌 Hasil ABSA")
        df_absa = pd.read_excel(absa_path)
        st.dataframe(df_absa.head(50))

        # ===============================
        # DISTRIBUSI ASPEK
        # ===============================
        st.subheader("📊 Distribusi Aspek")

        aspect_counts = df_absa["aspect"].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            aspect_counts.plot(kind="bar", ax=ax)
            ax.set_ylabel("Jumlah")
            ax.set_title("Distribusi Aspek")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            aspect_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            ax.set_title("Proporsi Aspek")
            st.pyplot(fig)

    # ===============================
    # HASIL KLASIFIKASI
    # ===============================
    labeled_path = os.path.join("output", "absa_labeled_numeric.xlsx")
    if os.path.exists(labeled_path):
        st.subheader("📈 Evaluasi Klasifikasi")

        df_label = pd.read_excel(labeled_path)
        sent_counts = df_label["label_text"].value_counts()

        fig, ax = plt.subplots()
        sent_counts.plot(kind="bar", ax=ax)
        ax.set_title("Distribusi Sentimen")
        st.pyplot(fig)

    # ===============================
    # CONFUSION MATRIX
    # ===============================
    tfidf_path = os.path.join("output", "tfidf_output.xlsx")
    if os.path.exists(tfidf_path):
        st.subheader("🧪 Confusion Matrix")

        # Ambil ulang hasil klasifikasi
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
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
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
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

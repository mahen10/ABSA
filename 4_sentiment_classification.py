import pandas as pd
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

def run(input_path, output_dir):
    # ============================
    # 1. Load data
    # ============================
    if not os.path.exists(input_path):
        raise FileNotFoundError("File input tidak ditemukan")
    
    df = pd.read_excel(input_path)
    
    # Validasi Kolom
    required_cols = {"opinion_context", "label_text"}
    if not required_cols.issubset(df.columns):
        if "processed_opinion" in df.columns:
            st.warning("⚠️ Menggunakan 'processed_opinion'.")
            df["feature_text"] = df["processed_opinion"]
        else:
            raise ValueError("Kolom opinion_context tidak ditemukan!")
    else:
        df["feature_text"] = df["opinion_context"]

    # Cleaning Simple
    df = df.dropna(subset=["feature_text", "label_text"])
    df["label_text"] = df["label_text"].str.lower().str.strip()
    df = df[df["label_text"].isin(["positive", "negative"])]
    df = df[df["feature_text"].astype(str).str.strip() != ""]
    
    if df.empty: return create_dummy_result()

    # Cek jumlah kelas
    if len(df["label_text"].unique()) < 2:
        return create_dummy_result(accuracy=1.0)

    # ============================
    # 2. TF-IDF (KUNCI PERBAIKAN DISINI)
    # ============================
    X = df["feature_text"].astype(str)
    y = df["label_text"]
    
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),   # UBAH KE (1,3): Tangkap "not bad at all"
        max_df=0.90,          # Abaikan kata yang muncul di 90% dokumen
        min_df=2,             # Abaikan typo unik
        stop_words=None,      # JANGAN PAKAI "english". "No/Not" itu penting!
        sublinear_tf=True     # Memuluskan frekuensi kata yang meledak
    )
    
    X_tfidf = tfidf.fit_transform(X)
    
    # ============================
    # 3. Split & Train
    # ============================
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42
        )
    
    # Model Configuration
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced", # Tetap pakai balanced agar recall seimbang
        solver="lbfgs",
        C=2.0,                   # Naikkan C sedikit (Model lebih percaya pada kata-kata spesifik)
        random_state=42
    )
    
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["negative", "positive"])
        }

        # Analisis Fitur (Cek kata apa yang bikin error)
        feature_names = tfidf.get_feature_names_out()
        coefs = model.coef_[0]
        
        # Ambil Top 10 Positive & Negative words
        top_pos = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)[:10]
        top_neg = sorted(zip(feature_names, coefs), key=lambda x: x[1])[:10]
        
        results["top_features"] = {"positive": top_pos, "negative": top_neg}

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return create_dummy_result()
    
    display_results(results, y_test, y_pred)
    return results

# --- HELPER VISUALISASI ---
def create_dummy_result(accuracy=0.0):
    return {
        "accuracy": accuracy,
        "classification_report": {
            "weighted avg": {"precision": 0, "recall": 0, "f1-score": 0},
            "positive": {"precision": 0, "recall": 0, "f1-score": 0},
            "negative": {"precision": 0, "recall": 0, "f1-score": 0}
        },
        "confusion_matrix": np.array([[0, 0], [0, 0]])
    }

def display_results(results, y_test, y_pred):
    st.markdown("---")
    st.markdown("### 📊 Model Performance (Improved)")
    col1, col2, col3, col4 = st.columns(4)
    report = results["classification_report"]
    
    acc = results.get('accuracy', 0)
    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("Precision", f"{report['weighted avg']['precision']:.2%}")
    col3.metric("Recall", f"{report['weighted avg']['recall']:.2%}")
    col4.metric("F1-Score", f"{report['weighted avg']['f1-score']:.2%}")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🎯 Confusion Matrix")
        plot_confusion_matrix_elegant(results["confusion_matrix"])
    with c2:
        st.markdown("#### 📈 Class Performance")
        plot_class_metrics_elegant(report)
        
    if "top_features" in results:
        st.markdown("---")
        st.markdown("### 🔍 Kata Penentu Keputusan")
        cp, cn = st.columns(2)
        with cp:
            st.success("**Top Positive Words:**")
            st.write(", ".join([f"{w} ({c:.2f})" for w, c in results["top_features"]["positive"]]))
        with cn:
            st.error("**Top Negative Words:**")
            st.write(", ".join([f"{w} ({c:.2f})" for w, c in results["top_features"]["negative"]]))

def plot_confusion_matrix_elegant(cm):
    labels = ['Negative', 'Positive'] # Pastikan urutan label sesuai model
    cm_sum = cm.sum(axis=1)[:, None]
    cm_p = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm.astype('float')), where=cm_sum!=0) * 100
    
    text = [[f"{cm[i][j]}<br>({cm_p[i][j]:.1f}%)" for j in range(len(cm[i]))] for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels, text=text,
        texttemplate="%{text}", textfont={"size": 14, "color": "white"},
        colorscale=[[0, '#3D5A80'], [0.5, '#98C1D9'], [1, '#EE6C4D']], showscale=False
    ))
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=20,b=20), plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

def plot_class_metrics_elegant(report):
    classes = [c for c in ['negative', 'positive'] if c in report]
    metrics = ['precision', 'recall', 'f1-score']
    data = []
    colors = ['#3D5A80', '#98C1D9', '#EE6C4D']
    for i, metric in enumerate(metrics):
        vals = [report[cls][metric] for cls in classes]
        data.append(go.Bar(name=metric.capitalize(), x=[c.capitalize() for c in classes], y=vals, text=[f'{v:.2%}' for v in vals], textposition='auto', marker_color=colors[i]))
    
    fig = go.Figure(data=data)
    fig.update_layout(barmode='group', height=300, margin=dict(l=20,r=20,t=20,b=20), plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

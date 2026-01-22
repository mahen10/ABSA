import pandas as pd
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ==========================================
# 1. VISUALIZATION FUNCTIONS
# ==========================================
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

def plot_confusion_matrix_elegant(cm):
    labels = ['Negative', 'Positive']
    cm_sum = cm.sum(axis=1)[:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_p = np.divide(cm.astype('float'), cm_sum, where=cm_sum!=0) * 100
        cm_p = np.nan_to_num(cm_p)
    
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
        data.append(go.Bar(
            name=metric.capitalize(), 
            x=[c.capitalize() for c in classes], 
            y=vals, 
            text=[f'{v:.2%}' for v in vals], 
            textposition='auto',
            marker_color=colors[i]
        ))
    
    fig = go.Figure(data=data)
    fig.update_layout(barmode='group', height=300, margin=dict(l=20,r=20,t=20,b=20), plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

def display_results(results, y_test, y_pred):
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    report = results["classification_report"]
    
    acc = results.get('accuracy', 0)
    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("Precision", f"{report.get('weighted avg', {}).get('precision', 0):.2%}")
    col3.metric("Recall", f"{report.get('weighted avg', {}).get('recall', 0):.2%}")
    col4.metric("F1-Score", f"{report.get('weighted avg', {}).get('f1-score', 0):.2%}")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🎯 Confusion Matrix")
        plot_confusion_matrix_elegant(results["confusion_matrix"])
    with c2:
        st.markdown("#### 📈 Class Performance")
        plot_class_metrics_elegant(report)
        
    with st.expander("📋 Detailed Report"):
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

# ==========================================
# 2. MAIN LOGIC
# ==========================================

def run(input_path, output_dir):
    # Load Data
    if not os.path.exists(input_path):
        raise FileNotFoundError("File input tidak ditemukan")
    
    df = pd.read_excel(input_path)
    
    # Validasi Kolom
    required_cols = {"opinion_context", "label_text"}
    if not required_cols.issubset(df.columns):
        if "processed_opinion" in df.columns:
            st.warning("⚠️ Menggunakan 'processed_opinion' (Akurasi mungkin turun).")
            df["feature_text"] = df["processed_opinion"]
        else:
            raise ValueError("Kolom opinion_context tidak ditemukan!")
    else:
        df["feature_text"] = df["opinion_context"]

    # Cleaning
    df = df.dropna(subset=["feature_text", "label_text"])
    df["label_text"] = df["label_text"].str.lower().str.strip()
    df = df[df["label_text"].isin(["positive", "negative"])]
    df = df[df["feature_text"].astype(str).str.strip() != ""]
    
    # --- CEK JUMLAH DATA SETELAH FILTER ---
    n_samples = len(df)
    if n_samples < 2:
        st.warning("⚠️ Data terlalu sedikit (< 2 baris) setelah filtering. Tidak bisa training.")
        return create_dummy_result(accuracy=0.0)

    # Feature Extraction (TF-IDF)
    X = df["feature_text"].astype(str)
    y = df["label_text"]
    
    # --- PERBAIKAN DINAMIS (ANTI CRASH) ---
    # Jika data < 5 baris, paksa min_df=1 dan max_df=1.0 agar tidak error
    use_min_df = 2 if n_samples >= 5 else 1
    use_max_df = 0.90 if n_samples >= 5 else 1.0

    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),   
        max_df=use_max_df,     # <-- Pakai variabel dinamis
        min_df=use_min_df,     # <-- Pakai variabel dinamis
        stop_words="english",  
        sublinear_tf=True
    )
    
    try:
        X_tfidf = tfidf.fit_transform(X)
    except ValueError as e:
        # Fallback terakhir jika masih error
        st.warning(f"⚠️ TF-IDF Error: {e}. Mencoba mode fallback...")
        tfidf = TfidfVectorizer(min_df=1, max_df=1.0)
        X_tfidf = tfidf.fit_transform(X)

    # Split Data (70:30)
    # Jika data terlalu sedikit (<5), jangan split, pakai data training untuk test (hanya untuk mencegah crash)
    if n_samples < 5:
        X_train, X_test, y_train, y_test = X_tfidf, X_tfidf, y, y
        st.info("ℹ️ Data sangat sedikit, melewati Train-Test Split.")
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_tfidf, y, 
                test_size=0.3, 
                random_state=42, 
                stratify=y
            )
        except ValueError:
            # Fallback jika stratify gagal (misal cuma ada 1 kelas di salah satu split)
            X_train, X_test, y_train, y_test = train_test_split(
                X_tfidf, y, 
                test_size=0.3, 
                random_state=42
            )
    
    # Model Training
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced", 
        solver="lbfgs",
        C=2.0,                   
        random_state=42,
        n_jobs=-1
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

    except Exception as e:
        st.error(f"Error training: {str(e)}")
        return create_dummy_result()
    
    display_results(results, y_test, y_pred)
    return results

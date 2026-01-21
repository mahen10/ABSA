# =====================================================
# FILE: 4_sentiment_classification.py
# PERBAIKAN: Training menggunakan 'opinion_context'
# =====================================================
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
    # Load data
    # ============================
    if not os.path.exists(input_path):
        raise FileNotFoundError("File input tidak ditemukan")
    
    df = pd.read_excel(input_path)
    
    # KITA BUTUH OPINION CONTEXT UNTUK FITUR, DAN LABEL UNTUK TARGET
    required_cols = {"opinion_context", "label_text"}
    
    if not required_cols.issubset(df.columns):
        # Fallback jika context tidak ada, pakai processed_opinion
        if "processed_opinion" in df.columns:
            st.warning("⚠️ Menggunakan 'processed_opinion' karena 'opinion_context' tidak ada. Akurasi mungkin rendah.")
            df["feature_text"] = df["processed_opinion"]
        else:
            raise ValueError("Kolom opinion_context tidak ditemukan!")
    else:
        # PENGGUNAAN CONTEXT ADALAH KUNCI AKURASI TINGGI
        df["feature_text"] = df["opinion_context"]

    df = df.dropna(subset=["feature_text", "label_text"])
    df["label_text"] = df["label_text"].str.lower().str.strip()
    
    # Filter hanya positive/negative
    df = df[df["label_text"].isin(["positive", "negative"])]
    df = df[df["feature_text"].astype(str).str.strip() != ""]
    
    if df.empty:
        return create_dummy_result()
    
    unique_labels = df["label_text"].unique()
    if len(unique_labels) < 2:
        st.warning(f"⚠️ Data hanya 1 kelas: {unique_labels}. Butuh min 2 kelas.")
        return create_dummy_result(accuracy=1.0)

    # ============================
    # Split X dan y
    # ============================
    # X SEKARANG ADALAH KALIMAT UTUH (CONTEXT)
    X = df["feature_text"].astype(str)
    y = df["label_text"]
    
    # ============================
    # TF-IDF (Dinamis)
    # ============================
    n_samples = len(X)
    use_min_df = 2 if n_samples >= 10 else 1 # Turunkan min_df agar kata unik tetap terambil
    use_max_df = 0.95 

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), # Unigram & Bigram (penting untuk menangkap "not good")
        max_df=use_max_df,
        min_df=use_min_df,
    )
    X_tfidf = tfidf.fit_transform(X)
    
    # ============================
    # Train-Test Split
    # ============================
    if n_samples < 5:
        X_train, X_test, y_train, y_test = X_tfidf, X_tfidf, y, y
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_tfidf, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_tfidf, y, test_size=0.2, random_state=42
            )
    
    # ============================
    # Logistic Regression
    # ============================
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        C=1.0 # Default regularization
    )
    
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
    except Exception as e:
        st.error(f"Gagal melatih model: {str(e)}")
        return create_dummy_result()
    
    display_results(results, y_test, y_pred)
    return results

# --- (FUNGSI VISUALISASI DI BAWAH SAMA SEPERTI SEBELUMNYA) ---
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
    st.markdown("### 📊 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    report = results["classification_report"]
    
    def get_metric(metric_type, metric_name):
        try: return report[metric_type][metric_name]
        except KeyError: return 0.0

    acc = results.get('accuracy', 0)
    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("Precision (Avg)", f"{get_metric('weighted avg', 'precision'):.2%}")
    col3.metric("Recall (Avg)", f"{get_metric('weighted avg', 'recall'):.2%}")
    col4.metric("F1-Score (Avg)", f"{get_metric('weighted avg', 'f1-score'):.2%}")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🎯 Confusion Matrix")
        plot_confusion_matrix_elegant(results["confusion_matrix"])
    with c2:
        st.markdown("#### 📈 Class Performance")
        plot_class_metrics_elegant(report)
    
    with st.expander("📋 Detailed Classification Report"):
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

def plot_confusion_matrix_elegant(cm):
    labels = ['Negative', 'Positive']
    cm_p = cm.astype('float') / cm.sum(axis=1)[:, None] * 100 if cm.sum() > 0 else cm
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
    for metric in metrics:
        vals = [report[cls][metric] for cls in classes]
        data.append(go.Bar(name=metric.capitalize(), x=[c.capitalize() for c in classes], y=vals, text=[f'{v:.2%}' for v in vals], textposition='auto'))
    
    fig = go.Figure(data=data)
    fig.update_layout(barmode='group', height=300, margin=dict(l=20,r=20,t=20,b=20), plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1))
    colors = ['#3D5A80', '#98C1D9', '#EE6C4D']
    for i, trace in enumerate(fig.data): trace.marker.color = colors[i]
    st.plotly_chart(fig, use_container_width=True)

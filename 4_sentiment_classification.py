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
# 1. HELPER FUNCTIONS (Visualisasi & Dummy)
# ==========================================

def create_dummy_result(accuracy=0.0):
    """Membuat hasil kosong jika terjadi error atau data kosong."""
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
    """Plot Confusion Matrix yang estetik menggunakan Plotly."""
    labels = ['Negative', 'Positive']
    # Hitung persentase
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
    """Plot Metrics (Precision, Recall, F1) per kelas."""
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
    fig.update_layout(
        barmode='group', 
        height=300, 
        margin=dict(l=20,r=20,t=20,b=20), 
        plot_bgcolor='rgba(0,0,0,0)', 
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

def display_results(results, y_test, y_pred):
    """Menampilkan hasil analisis di Streamlit."""
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    
    # Metrics Utama
    col1, col2, col3, col4 = st.columns(4)
    report = results["classification_report"]
    
    def get_metric(metric_type, metric_name):
        try:
            return report[metric_type][metric_name]
        except KeyError:
            return 0.0
    
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
    
    # Feature Importance
    if "top_features" in results:
        st.markdown("---")
        st.markdown("### 🔍 Top Influential Features")
        st.caption("Kata-kata yang paling mempengaruhi keputusan model.")
        col_pos, col_neg = st.columns(2)
        
        with col_pos:
            st.success("**Positive Indicators (Cenderung Positif):**")
            pos_df = pd.DataFrame(results["top_features"]["positive"], columns=["Feature", "Weight"])
            st.dataframe(pos_df.style.format({"Weight": "{:.3f}"}), use_container_width=True)
        
        with col_neg:
            st.error("**Negative Indicators (Cenderung Negatif):**")
            neg_df = pd.DataFrame(results["top_features"]["negative"], columns=["Feature", "Weight"])
            st.dataframe(neg_df.style.format({"Weight": "{:.3f}"}), use_container_width=True)
    
    with st.expander("📋 Lihat Detailed Classification Report"):
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

# ==========================================
# 2. MAIN LOGIC FUNCTION
# ==========================================

def run(input_path, output_dir):
    if not os.path.exists(input_path):
        raise FileNotFoundError("File input tidak ditemukan")
    
    # 1. Load Data
    df = pd.read_excel(input_path)
    
    # 2. Setup Kolom
    required_cols = {"opinion_context", "label_text"}
    if not required_cols.issubset(df.columns):
        if "processed_opinion" in df.columns:
            st.warning("⚠️ Menggunakan 'processed_opinion' karena 'opinion_context' tidak ada.")
            df["feature_text"] = df["processed_opinion"]
        else:
            raise ValueError("Kolom opinion_context tidak ditemukan!")
    else:
        df["feature_text"] = df["opinion_context"]
    
    # 3. Cleaning Dasar
    df = df.dropna(subset=["feature_text", "label_text"])
    df["label_text"] = df["label_text"].str.lower().str.strip()
    df["feature_text"] = df["feature_text"].astype(str).str.strip()
    
    # Filter hanya 2 kelas
    df = df[df["label_text"].isin(["positive", "negative"])]
    df = df[df["feature_text"] != ""]
    
    if df.empty:
        return create_dummy_result()
    
    # Cek Distribusi
    class_counts = df["label_text"].value_counts()
    st.info(f"📊 Original Class Distribution: {dict(class_counts)}")
    
    if len(class_counts) < 2:
        st.warning("⚠️ Data hanya memiliki 1 jenis label. Training dibatalkan.")
        return create_dummy_result(accuracy=1.0)
    
    # 4. Split Data (Stratified)
    # Penting: Split dilakukan SEBELUM Vectorizer untuk mencegah data leakage
    X = df["feature_text"]
    y = df["label_text"]
    
    try:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback jika stratify gagal (misal data terlalu sedikit)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # 5. TF-IDF Vectorization (Optimized)
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),    # Unigram dan Bigram
        max_df=0.85,           # Abaikan kata yang muncul di >85% dokumen
        min_df=2,              # Abaikan kata yang muncul <2 kali (typo/unik)
        max_features=5000,     # Batasi fitur agar tidak terlalu noise
        sublinear_tf=True,     # Scaling logaritmik
        analyzer='word'
    )
    
    # Fit hanya pada TRAIN, Transform pada TEST
    X_train = tfidf.fit_transform(X_train_raw)
    X_test = tfidf.transform(X_test_raw)
    
    # 6. Model Training (Single Balancing Strategy)
    # Kita menggunakan class_weight='balanced' tanpa oversampling manual
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # Penyeimbangan otomatis
        solver="lbfgs",
        C=0.8,                    # Regularization (sedikit dikurangi dari 1.0)
        random_state=42,
        n_jobs=-1
    )
    
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 7. Compile Results
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["negative", "positive"])
        }
        
        # Feature Importance
        feature_names = tfidf.get_feature_names_out()
        coefficients = model.coef_[0]
        
        top_positive = sorted(zip(feature_names, coefficients), key=lambda x: x[1], reverse=True)[:10]
        top_negative = sorted(zip(feature_names, coefficients), key=lambda x: x[1])[:10]
        
        results["top_features"] = {
            "positive": top_positive,
            "negative": top_negative
        }
        
    except Exception as e:
        st.error(f"❌ Gagal melatih model: {str(e)}")
        return create_dummy_result()
    
    # 8. Tampilkan Output
    display_results(results, y_test, y_pred)
    return results

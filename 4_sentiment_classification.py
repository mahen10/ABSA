# =====================================================
# FILE: 4_sentiment_classification.py
# Klasifikasi Sentimen – Logistic Regression + TF-IDF
# Fokus evaluasi (TANPA output Excel berat)
# =====================================================
import pandas as pd
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
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
    required_cols = {"processed_opinion", "label_text"}
    
    if not required_cols.issubset(df.columns):
        raise ValueError("Kolom processed_opinion / label_text tidak ditemukan")
    
    df = df[["processed_opinion", "label_text"]].dropna()
    df["label_text"] = df["label_text"].str.lower().str.strip()
    df = df[df["label_text"].isin(["positive", "negative"])]
    df = df[df["processed_opinion"].astype(str).str.strip() != ""]
    
    if df.empty:
        raise ValueError("Data kosong setelah preprocessing")
    
    # ============================
    # Split X dan y
    # ============================
    X = df["processed_opinion"]
    y = df["label_text"]
    
    # ============================
    # TF-IDF (RINGAN)
    # ============================
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=3,
        stop_words="english"
    )
    X_tfidf = tfidf.fit_transform(X)
    
    # ============================
    # Train-Test Split
    # ============================
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # ============================
    # Logistic Regression
    # ============================
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear"
    )
    model.fit(X_train, y_train)
    
    # ============================
    # Evaluasi
    # ============================
    y_pred = model.predict(X_test)
    
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    
    # ============================
    # ELEGANT VISUALIZATION
    # ============================
    display_results(results, y_test, y_pred)
    
    return results


def display_results(results, y_test, y_pred):
    """Tampilan elegant untuk hasil klasifikasi"""
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    
    # ============================
    # Metrics Cards (Compact & Elegant)
    # ============================
    col1, col2, col3, col4 = st.columns(4)
    
    report = results["classification_report"]
    
    with col1:
        st.metric(
            label="Accuracy",
            value=f"{results['accuracy']:.2%}",
            delta=f"{results['accuracy']*100:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Precision (Avg)",
            value=f"{report['weighted avg']['precision']:.2%}"
        )
    
    with col3:
        st.metric(
            label="Recall (Avg)",
            value=f"{report['weighted avg']['recall']:.2%}"
        )
    
    with col4:
        st.metric(
            label="F1-Score (Avg)",
            value=f"{report['weighted avg']['f1-score']:.2%}"
        )
    
    st.markdown("---")
    
    # ============================
    # Charts Layout (2 columns, compact)
    # ============================
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 🎯 Confusion Matrix")
        plot_confusion_matrix_elegant(results["confusion_matrix"])
    
    with col_right:
        st.markdown("#### 📈 Class Performance")
        plot_class_metrics_elegant(report)
    
    # ============================
    # Detailed Report (Expandable)
    # ============================
    with st.expander("📋 Detailed Classification Report"):
        df_report = pd.DataFrame(report).transpose()
        df_report = df_report.round(3)
        st.dataframe(
            df_report.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']),
            use_container_width=True
        )


def plot_confusion_matrix_elegant(cm):
    """Confusion matrix dengan design elegant & compact"""
    
    labels = ['Negative', 'Positive']
    
    # Hitung percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, None] * 100
    
    # Custom text dengan angka dan persen
    text = [[f"{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)" 
             for j in range(len(cm[i]))] 
            for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        colorscale=[
            [0, '#3D5A80'],      # Dark blue
            [0.5, '#98C1D9'],    # Light blue
            [1, '#EE6C4D']       # Coral
        ],
        showscale=False,
        hoverongaps=False
    ))
    
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(size=11),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_class_metrics_elegant(report):
    """Bar chart metrics per class - elegant & compact"""
    
    # Extract metrics untuk positive dan negative saja
    classes = ['negative', 'positive']
    metrics = ['precision', 'recall', 'f1-score']
    
    data = []
    for metric in metrics:
        values = [report[cls][metric] for cls in classes]
        data.append(go.Bar(
            name=metric.capitalize(),
            x=['Negative', 'Positive'],
            y=values,
            text=[f'{v:.2%}' for v in values],
            textposition='auto',
            textfont=dict(size=11),
            hovertemplate='%{y:.2%}<extra></extra>'
        ))
    
    fig = go.Figure(data=data)
    
    fig.update_layout(
        barmode='group',
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        font=dict(size=11),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            tickformat='.0%',
            gridcolor='rgba(128,128,128,0.2)'
        ),
        xaxis=dict(
            showgrid=False
        )
    )
    
    # Warna elegant
    colors = ['#3D5A80', '#98C1D9', '#EE6C4D']
    for i, trace in enumerate(fig.data):
        trace.marker.color = colors[i]
    
    st.plotly_chart(fig, use_container_width=True)

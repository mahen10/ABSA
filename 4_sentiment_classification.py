# =====================================================
# FILE: 4_sentiment_classification.py
# Klasifikasi Sentimen – Logistic Regression + TF-IDF
# Dengan Visualisasi Elegant
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


def create_metric_card(label, value, delta=None):
    """Create elegant metric display"""
    if isinstance(value, float):
        value_str = f"{value:.2%}" if value <= 1 else f"{value:.2f}"
    else:
        value_str = str(value)
    
    st.metric(label=label, value=value_str, delta=delta)


def plot_confusion_matrix(cm, labels):
    """Create compact confusion matrix heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hoverongaps=False,
        showscale=False
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=11)
    )
    
    return fig


def plot_classification_metrics(report_dict):
    """Create compact classification metrics bar chart"""
    metrics_data = []
    
    for label in ['positive', 'negative']:
        if label in report_dict:
            metrics_data.append({
                'Class': label.capitalize(),
                'Precision': report_dict[label]['precision'],
                'Recall': report_dict[label]['recall'],
                'F1-Score': report_dict[label]['f1-score']
            })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    fig = go.Figure()
    
    colors = {'Precision': '#FF6B6B', 'Recall': '#4ECDC4', 'F1-Score': '#45B7D1'}
    
    for metric in ['Precision', 'Recall', 'F1-Score']:
        fig.add_trace(go.Bar(
            name=metric,
            x=df_metrics['Class'],
            y=df_metrics[metric],
            marker_color=colors[metric],
            text=df_metrics[metric].round(3),
            texttemplate='%{text:.2f}',
            textposition='outside'
        ))
    
    fig.update_layout(
        title="Classification Metrics by Class",
        barmode='group',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[0, 1.1]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(size=11)
    )
    
    return fig


def plot_class_distribution(y_train, y_test):
    """Create compact class distribution chart"""
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Train',
        x=train_counts.index,
        y=train_counts.values,
        marker_color='#667eea',
        text=train_counts.values,
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Test',
        x=test_counts.index,
        y=test_counts.values,
        marker_color='#764ba2',
        text=test_counts.values,
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Data Distribution",
        barmode='group',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(size=11)
    )
    
    return fig


def run(input_path, output_dir):
    """Main pipeline with elegant Streamlit visualization"""
    
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
    # Streamlit UI Header
    # ============================
    st.markdown("---")
    st.markdown("### 📊 Sentiment Classification Results")
    
    # Dataset Info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        positive_count = (df["label_text"] == "positive").sum()
        st.metric("Positive", positive_count)
    with col3:
        negative_count = (df["label_text"] == "negative").sum()
        st.metric("Negative", negative_count)
    with col4:
        balance = min(positive_count, negative_count) / max(positive_count, negative_count)
        st.metric("Balance Ratio", f"{balance:.2f}")
    
    # ============================
    # Split X dan y
    # ============================
    X = df["processed_opinion"]
    y = df["label_text"]
    
    # ============================
    # TF-IDF
    # ============================
    with st.spinner("Extracting TF-IDF features..."):
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=3,
            stop_words="english"
        )
        X_tfidf = tfidf.fit_transform(X)
    
    st.info(f"✓ TF-IDF Features: {X_tfidf.shape[1]} dimensions")
    
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
    with st.spinner("Training Logistic Regression model..."):
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear"
        )
        model.fit(X_train, y_train)
    
    st.success("✓ Model trained successfully")
    
    # ============================
    # Evaluasi
    # ============================
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # ============================
    # Display Results - Compact Layout
    # ============================
    st.markdown("---")
    st.markdown("### 🎯 Model Performance")
    
    # Main Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_metric_card("Accuracy", accuracy)
    with col2:
        create_metric_card("Precision (Avg)", report_dict['weighted avg']['precision'])
    with col3:
        create_metric_card("Recall (Avg)", report_dict['weighted avg']['recall'])
    with col4:
        create_metric_card("F1-Score (Avg)", report_dict['weighted avg']['f1-score'])
    
    # Charts Row - 2 columns for better space utilization
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            plot_confusion_matrix(cm, ['Negative', 'Positive']),
            use_container_width=True
        )
        
        st.plotly_chart(
            plot_class_distribution(y_train, y_test),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            plot_classification_metrics(report_dict),
            use_container_width=True
        )
        
        # Detailed Metrics Table
        st.markdown("#### 📋 Detailed Metrics")
        metrics_df = pd.DataFrame({
            'Class': ['Positive', 'Negative'],
            'Precision': [
                report_dict['positive']['precision'],
                report_dict['negative']['precision']
            ],
            'Recall': [
                report_dict['positive']['recall'],
                report_dict['negative']['recall']
            ],
            'F1-Score': [
                report_dict['positive']['f1-score'],
                report_dict['negative']['f1-score']
            ],
            'Support': [
                int(report_dict['positive']['support']),
                int(report_dict['negative']['support'])
            ]
        })
        
        st.dataframe(
            metrics_df.style.format({
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1-Score': '{:.3f}'
            }).background_gradient(subset=['Precision', 'Recall', 'F1-Score'], cmap='Blues'),
            use_container_width=True,
            height=150
        )
    
    # ============================
    # Return results
    # ============================
    return {
        "accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix": cm
    }

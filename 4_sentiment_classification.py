import pandas as pd
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def run(input_path, output_dir):
    if not os.path.exists(input_path):
        raise FileNotFoundError("File input tidak ditemukan")
    
    df = pd.read_excel(input_path)
    
    # Setup feature text
    required_cols = {"opinion_context", "label_text"}
    if not required_cols.issubset(df.columns):
        if "processed_opinion" in df.columns:
            st.warning("⚠️ Menggunakan 'processed_opinion' karena 'opinion_context' tidak ada.")
            df["feature_text"] = df["processed_opinion"]
        else:
            raise ValueError("Kolom opinion_context tidak ditemukan!")
    else:
        df["feature_text"] = df["opinion_context"]
    
    df = df.dropna(subset=["feature_text", "label_text"])
    df["label_text"] = df["label_text"].str.lower().str.strip()
    df = df[df["label_text"].isin(["positive", "negative"])]
    df = df[df["feature_text"].astype(str).str.strip() != ""]
    
    if df.empty:
        return create_dummy_result()
    
    # Check class distribution
    class_counts = df["label_text"].value_counts()
    st.info(f"📊 Class Distribution BEFORE balancing: {dict(class_counts)}")
    
    unique_labels = df["label_text"].unique()
    if len(unique_labels) < 2:
        st.warning(f"⚠️ Data hanya 1 kelas: {unique_labels}.")
        return create_dummy_result(accuracy=1.0)
    
    # ✅ PERBAIKAN 1: Manual Oversampling (tanpa imbalanced-learn)
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    minority_count = class_counts[minority_class]
    majority_count = class_counts[majority_class]
    imbalance_ratio = majority_count / minority_count
    
    if imbalance_ratio > 2.0 and minority_count >= 5:
        st.info(f"🔄 Applying Manual Oversampling (ratio: {imbalance_ratio:.1f}:1)")
        
        # Separate classes
        df_majority = df[df["label_text"] == majority_class]
        df_minority = df[df["label_text"] == minority_class]
        
        # Oversample minority class
        df_minority_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=majority_count,
            random_state=42
        )
        
        # Combine back
        df = pd.concat([df_majority, df_minority_upsampled])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        st.success(f"✅ Balanced classes: {dict(df['label_text'].value_counts())}")
    
    # Feature extraction
    X = df["feature_text"].astype(str)
    y = df["label_text"]
    
    # ✅ PERBAIKAN 2: TF-IDF tanpa stop words (keep negations!)
    n_samples = len(X)
    use_min_df = 1 if n_samples < 20 else 2
    
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_df=0.95,
        min_df=use_min_df,
        sublinear_tf=True,
        analyzer='word',
        stop_words=None  # Keep "not", "no", etc.
    )
    X_tfidf = tfidf.fit_transform(X)
    
    # Train-test split
    if n_samples < 5:
        X_train, X_test, y_train, y_test = X_tfidf, X_tfidf, y, y
        st.warning("⚠️ Dataset terlalu kecil, tidak ada train-test split.")
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_tfidf, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError as e:
            st.warning(f"⚠️ Stratified split gagal: {e}")
            X_train, X_test, y_train, y_test = train_test_split(
                X_tfidf, y, test_size=0.2, random_state=42
            )
    
    # ✅ PERBAIKAN 3: Model dengan hyperparameter tuning
    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="saga",
        C=0.5,
        penalty="l2",
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
        
        # Feature importance
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
    
    display_results(results, y_test, y_pred)
    return results

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
    
    # Feature importance
    if "top_features" in results:
        st.markdown("---")
        st.markdown("### 🔍 Top Influential Features")
        col_pos, col_neg = st.columns(2)
        
        with col_pos:
            st.markdown("**Positive Indicators:**")
            pos_df = pd.DataFrame(results["top_features"]["positive"], columns=["Feature", "Weight"])
            st.dataframe(pos_df.style.format({"Weight": "{:.3f}"}), use_container_width=True)
        
        with col_neg:
            st.markdown("**Negative Indicators:**")
            neg_df = pd.DataFrame(results["top_features"]["negative"], columns=["Feature", "Weight"])
            st.dataframe(neg_df.style.format({"Weight": "{:.3f}"}), use_container_width=True)
    
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
        data.append(go.Bar(
            name=metric.capitalize(), 
            x=[c.capitalize() for c in classes], 
            y=vals, 
            text=[f'{v:.2%}' for v in vals], 
            textposition='auto'
        ))
    
    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='group', 
        height=300, 
        margin=dict(l=20,r=20,t=20,b=20), 
        plot_bgcolor='rgba(0,0,0,0)', 
        legend=dict(orientation="h", y=1.1)
    )
    colors = ['#3D5A80', '#98C1D9', '#EE6C4D']
    for i, trace in enumerate(fig.data):
        trace.marker.color = colors[i]
    st.plotly_chart(fig, use_container_width=True)

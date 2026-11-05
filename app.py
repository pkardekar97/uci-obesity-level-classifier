import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Obesity Risk Level Classifier",
    page_icon="⚖️",
    layout="wide"
)

# ---------------------------------------------------------
# Header & Assignment Information
# ---------------------------------------------------------
st.title("Obesity Level Classification System")
st.markdown("**BITS Pilani M.Tech AI/ML | Machine Learning Assignment 2**")
st.caption("Student Name: Pranay Kardekar | BITS ID: 2026AK05670")
st.markdown("---")

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.header("1. Model Selection")
model_options = [
    "Support Vector Machine",
    "KNN",
    "Decision Tree",
    "Logistic Regression",
    "Naive Bayes"
]
selected_model_name = st.sidebar.selectbox("Choose Classifier", model_options)

st.sidebar.header("2. Test Data Input")
data_mode = st.sidebar.radio(
    "Select Evaluation Mode:",
    ("Upload Custom Test CSV", "Use Sample Held-out Test Data"),
    index=0
)

uploaded_file = None
if data_mode == "Upload Custom Test CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload test dataset (.csv)", 
        type=["csv"],
        help="Upload a test CSV containing the 16 UCI Obesity features."
    )

# ---------------------------------------------------------
# Model Mapping & Asset Loading
# ---------------------------------------------------------
model_file_map = {
    "Support Vector Machine": "svm_model.pkl",
    "KNN": "knn_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

@st.cache_resource
def load_model(model_name):
    filename = model_file_map.get(model_name)
    if filename and os.path.exists(filename):
        return joblib.load(filename)
    return None

model = load_model(selected_model_name)

# ---------------------------------------------------------
# Data Acquisition & Gating
# ---------------------------------------------------------
df = None

if data_mode == "Upload Custom Test CSV":
    if uploaded_file is None:
        st.info("👈 **Awaiting Data Input:** Please upload a test dataset CSV file from the sidebar to begin model evaluation.")
        st.markdown(
            """
            > **Evaluation Note:**  
            > If you do not have an external test dataset on hand, switch the sidebar mode to **"Use Sample Held-out Test Data"** to evaluate the trained models on the held-out test split.
            """
        )
        st.stop()
    else:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
            st.stop()
else:
    # Fallback / sample mode
    sample_path = "test_data.csv"
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        st.sidebar.success("Loaded sample held-out test dataset.")
    else:
        st.error(f"Sample test dataset '{sample_path}' not found in repository.")
        st.stop()

# ---------------------------------------------------------
# Evaluation Dashboard (Renders only after data is loaded)
# ---------------------------------------------------------
if df is not None:
    st.subheader("Dataset Sample (First 5 Rows)")
    st.dataframe(df.head(), use_container_width=True)

    if model is None:
        st.error(f"Serialized model file for '{selected_model_name}' could not be loaded. Ensure .pkl files are in the repository.")
        st.stop()

    # Target variable handling
    target_col = "NObeyesdad"
    if target_col in df.columns:
        X_test = df.drop(columns=[target_col])
        y_test = df[target_col]
        has_ground_truth = True
    else:
        X_test = df
        y_test = None
        has_ground_truth = False

    # Inference
    try:
        y_pred = model.predict(X_test)
        has_proba = hasattr(model, "predict_proba")
        y_proba = model.predict_proba(X_test) if has_proba else None
    except Exception as e:
        st.error(f"Inference failed on input data: {e}")
        st.stop()

    # Display Metrics if Ground Truth is Available
    if has_ground_truth:
        st.subheader(f"Performance Metrics: {selected_model_name}")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Multi-class AUC
        auc_score = "N/A"
        if has_proba and len(np.unique(y_test)) > 1:
            try:
                auc_score = f"{roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted'):.4f}"
            except Exception:
                auc_score = "N/A"

        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
        m_col1.metric("Accuracy", f"{acc:.4f}")
        m_col2.metric("AUC", auc_score)
        m_col3.metric("Precision", f"{prec:.4f}")
        m_col4.metric("Recall", f"{rec:.4f}")
        m_col5.metric("F1 Score", f"{f1:.4f}")
        m_col6.metric("MCC", f"{mcc:.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        labels = np.unique(np.concatenate([y_test.unique(), np.unique(y_pred)]))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix: {selected_model_name}")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        plt.close()

    else:
        st.subheader("Batch Predictions")
        pred_summary = df.copy()
        pred_summary["Predicted_Obesity_Level"] = y_pred
        st.dataframe(pred_summary.head(10), use_container_width=True)
        st.info("Uploaded dataset does not contain ground-truth target 'NObeyesdad'. Showing predictions.")

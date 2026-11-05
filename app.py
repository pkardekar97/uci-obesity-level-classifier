import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from metrics_utils import compute_metrics

st.set_page_config(page_title="Obesity Level Classifier", page_icon="🩺", layout="wide")

# Custom lightweight styling
st.markdown("""
<style>
    .reportview-container .main .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #1e3a8a; }
</style>
""", unsafe_allow_html=True)

st.title("Obesity Level Classification System")
st.markdown("**BITS Pilani M.Tech AI/ML | Machine Learning Assignment 2**")
st.caption("Student Name: Pranay Kardekar | BITS ID: 2026AK05670")
st.markdown(
    "Interactive evaluation dashboard evaluating 5 distinct classifiers trained on the UCI Obesity dataset. "
    "Features include 14 lifestyle and demographic attributes (Height and Weight removed to avoid target leakage)."
)

# Load metadata
@st.cache_resource
def load_meta():
    meta_path = "model/meta.joblib"
    if os.path.exists(meta_path):
        return joblib.load(meta_path)
    return None

meta = load_meta()
if meta is None:
    st.error("Metadata file 'model/meta.joblib' not found. Please ensure it is present in the repository.")
    st.stop()

target_col = meta["target"]
class_names = meta["classes"]
readable_classes = [c.replace("_", " ") for c in class_names]

model_options = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib",
}

# Sidebar controls
st.sidebar.header("Model Configuration")
selected_model_name = st.sidebar.selectbox("Choose Classifier", list(model_options.keys()))

@st.cache_resource
def load_selected_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_selected_model(model_options[selected_model_name])
if model is None:
    st.error(f"Model file '{model_options[selected_model_name]}' could not be loaded.")
    st.stop()

# Data upload controls
st.sidebar.markdown("---")
st.sidebar.subheader("Test Data Input")

data_mode = st.sidebar.radio(
    "Select Evaluation Mode:",
    ("Upload Custom Test CSV", "Use Sample Held-out Test Data"),
    index=0
)

df_eval = None

if data_mode == "Upload Custom Test CSV":
    uploaded_file = st.sidebar.file_uploader("Upload custom CSV test set", type=["csv"])
    if uploaded_file is not None:
        try:
            df_eval = pd.read_csv(uploaded_file)
            st.sidebar.success("Custom CSV uploaded successfully.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()
    else:
        st.info("👈 **Awaiting Data Input:** Please upload a test dataset CSV file from the sidebar to begin model evaluation.")
        st.markdown(
            """
            > **Note:** If you do not have an external test CSV ready, switch the sidebar mode to **"Use Sample Held-out Test Data"** to evaluate the trained model on the benchmark test split.
            """
        )
        st.stop()
else:
    sample_path = "test_data.csv"
    if os.path.exists(sample_path):
        df_eval = pd.read_csv(sample_path)
        st.sidebar.info("Using default held-out test_data.csv.")
    else:
        st.error(f"Sample test dataset '{sample_path}' not found.")
        st.stop()

# ---------------------------------------------------------
# Evaluation Dashboard (Renders only after data is available)
# ---------------------------------------------------------
st.subheader("Dataset Sample")
st.dataframe(df_eval.head(8), width="stretch")

if target_col in df_eval.columns:
    X_eval = df_eval.drop(columns=[target_col])
    y_raw = df_eval[target_col]
    label_map = {name: idx for idx, name in enumerate(class_names)}
    y_eval = y_raw.map(label_map)

    # Compute metrics
    metrics = compute_metrics(model, X_eval, y_eval)
    preds = model.predict(X_eval)

    st.markdown("---")
    st.subheader(f"Performance Metrics: {selected_model_name}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    c2.metric("AUC (OvR)", f"{metrics['AUC']:.4f}")
    c3.metric("Precision", f"{metrics['Precision']:.4f}")
    c4.metric("Recall", f"{metrics['Recall']:.4f}")
    c5.metric("F1-Score", f"{metrics['F1']:.4f}")
    c6.metric("MCC", f"{metrics['MCC']:.4f}")

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_eval, preds)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=readable_classes,
            yticklabels=readable_classes,
            cbar=False,
            ax=ax,
        )
        ax.set_xlabel("Predicted Class", fontweight="bold")
        ax.set_ylabel("True Class", fontweight="bold")
        plt.xticks(rotation=40, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    with col_right:
        st.subheader("Per-Class Classification Report")
        report_dict = classification_report(
            y_eval,
            preds,
            target_names=readable_classes,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.round(4), width="stretch")
else:
    st.subheader(f"Model Predictions: {selected_model_name}")
    preds = model.predict(df_eval)
    pred_labels = [readable_classes[p] for p in preds]
    results_df = df_eval.copy()
    results_df["Predicted_Obesity_Level"] = pred_labels
    st.dataframe(results_df, width="stretch")

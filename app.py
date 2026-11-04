import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from metrics_utils import compute_metrics

st.set_page_config(page_title="Obesity Level Classifier", layout="wide")

st.title("Obesity Level Classification System")
st.markdown(
    "Interactive multi-class classification system trained on the UCI Obesity dataset "
    "using 14 behavioral and demographic features (Height and Weight omitted to prevent target leakage)."
)

# Load metadata
@st.cache_resource
def load_meta():
    return joblib.load("model/meta.joblib")

meta = load_meta()
target_col = meta["target"]
class_names = meta["classes"]

model_options = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib",
}

# Sidebar controls
st.sidebar.header("Configuration")
selected_model_name = st.sidebar.selectbox("Select Classifier", list(model_options.keys()))

@st.cache_resource
def load_selected_model(path):
    return joblib.load(path)

model = load_selected_model(model_options[selected_model_name])

# Data upload
st.sidebar.subheader("Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df_eval = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom CSV uploaded successfully.")
else:
    st.sidebar.info("Using default held-out test_data.csv.")
    df_eval = pd.read_csv("test_data.csv")

st.subheader("Data Preview")
st.dataframe(df_eval.head(10), width="stretch")

if target_col in df_eval.columns:
    X_eval = df_eval.drop(columns=[target_col])
    y_raw = df_eval[target_col]
    label_map = {name: idx for idx, name in enumerate(class_names)}
    y_eval = y_raw.map(label_map)

    # Compute metrics
    metrics = compute_metrics(model, X_eval, y_eval)
    preds = model.predict(X_eval)

    st.subheader(f"Evaluation Metrics: {selected_model_name}")
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
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[c.replace("_", " ") for c in class_names],
            yticklabels=[c.replace("_", " ") for c in class_names],
            ax=ax,
        )
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        st.pyplot(fig)

    with col_right:
        st.subheader("Per-Class Classification Report")
        report_dict = classification_report(
            y_eval,
            preds,
            target_names=[c.replace("_", " ") for c in class_names],
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.round(4), width="stretch")
else:
    st.subheader(f"Model Predictions: {selected_model_name}")
    preds = model.predict(df_eval)
    pred_labels = [class_names[p] for p in preds]
    results_df = df_eval.copy()
    results_df["Predicted_Obesity_Level"] = pred_labels
    st.dataframe(results_df, width="stretch")
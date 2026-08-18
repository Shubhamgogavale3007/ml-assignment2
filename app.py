
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Bank Marketing Classification")
st.caption("ML Assignment 2 — UCI Bank Marketing Dataset")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def evaluate_model(model, data):
    y_true = data["y"].astype(str).str.lower().eq("yes").astype(int)
    X = data.drop(columns=["y"])
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    return {
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": roc_auc_score(y_true, prob),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
    }

# The assignment requires a CSV upload option.
uploaded = st.sidebar.file_uploader(
    "Upload test CSV",
    type=["csv"],
)

default_path = ROOT / "test_data.csv"

if uploaded is not None:
    data = pd.read_csv(uploaded)
elif default_path.exists():
    data = pd.read_csv(default_path)
    st.sidebar.info("Using the generated test_data.csv.")
else:
    st.warning("Upload test_data.csv to continue.")
    st.stop()

if "y" not in data.columns:
    st.error("The uploaded CSV must contain the target column `y`.")
    st.stop()

selected = st.sidebar.selectbox(
    "Select classification model",
    list(MODEL_FILES.keys()),
)

model_path = MODEL_DIR / MODEL_FILES[selected]

if not model_path.exists():
    st.error(
        f"{model_path.name} is missing. "
        "Run `python model/train_models.py` first."
    )
    st.stop()

model = load_model(model_path)

# Model comparison using the same held-out test CSV.
rows = []
for name, filename in MODEL_FILES.items():
    path = MODEL_DIR / filename
    if path.exists():
        m = load_model(path)
        metrics = evaluate_model(m, data)
        rows.append({"ML Model Name": name, **metrics})

comparison = pd.DataFrame(rows)

st.header("Model Comparison")
st.dataframe(
    comparison.round(4),
    use_container_width=True,
    hide_index=True,
)

st.header(f"Selected Model: {selected}")

metrics = evaluate_model(model, data)

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
c2.metric("AUC", f"{metrics['AUC']:.4f}")
c3.metric("Precision", f"{metrics['Precision']:.4f}")

c4, c5, c6 = st.columns(3)
c4.metric("Recall", f"{metrics['Recall']:.4f}")
c5.metric("F1 Score", f"{metrics['F1']:.4f}")
c6.metric("MCC", f"{metrics['MCC']:.4f}")

y_true = data["y"].astype(str).str.lower().eq("yes").astype(int)
X = data.drop(columns=["y"])
pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

st.header("Confusion Matrix")

cm = confusion_matrix(y_true, pred, labels=[0, 1])
cm_df = pd.DataFrame(
    cm,
    index=["Actual no", "Actual yes"],
    columns=["Predicted no", "Predicted yes"],
)
st.dataframe(cm_df, use_container_width=False)

st.header("Classification Report")

report = classification_report(
    y_true,
    pred,
    target_names=["no", "yes"],
    output_dict=True,
    zero_division=0,
)
st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

st.header("Prediction Results")

prediction_table = X.copy()
prediction_table["Actual"] = np.where(y_true == 1, "yes", "no")
prediction_table["Predicted"] = np.where(pred == 1, "yes", "no")
prediction_table["Probability of yes"] = np.round(prob, 4)

st.dataframe(
    prediction_table.head(100),
    use_container_width=True,
)

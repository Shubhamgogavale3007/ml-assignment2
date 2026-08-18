
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import requests

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def load_dataset_from_url():
    print("Downloading Bank Marketing dataset from:")
    print(DATA_URL)
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as outer_zip:
        bank_zip_name = next(
            n for n in outer_zip.namelist() if n.endswith("bank.zip")
        )
        bank_zip_bytes = outer_zip.read(bank_zip_name)

    with ZipFile(BytesIO(bank_zip_bytes)) as bank_zip:
        csv_name = next(
            n for n in bank_zip.namelist() if n.endswith("bank-full.csv")
        )
        with bank_zip.open(csv_name) as f:
            return pd.read_csv(f, sep=";")


def build_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical)
    ])


def main():
    df = load_dataset_from_url()
    print(f"\nDataset shape: {df.shape}")

    X = df.drop(columns=["y"])
    y = (df["y"].str.lower() == "yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42
        ),
        "kNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        ),
    }

    results = []

    for name, classifier in models.items():
        print(f"Training {name}...")

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", classifier)
        ])
        pipeline.fit(X_train, y_train)

        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]

        results.append({
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, prob),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred),
        })

        filename = name.lower().replace(" ", "_").replace("-", "") + ".pkl"
        joblib.dump(pipeline, MODEL_DIR / filename)

    # Required root-level test_data.csv
    test_data = X_test.copy()
    test_data["y"] = np.where(y_test.to_numpy() == 1, "yes", "no")
    test_data.to_csv(PROJECT_ROOT / "test_data.csv", index=False)

    comparison = pd.DataFrame(results)
    comparison.to_csv(PROJECT_ROOT / "model_comparison.csv", index=False)

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nGenerated:")
    print("  test_data.csv")
    print("  model/*.pkl")
    print("  model_comparison.csv")


if __name__ == "__main__":
    main()

"""
model/quick_train.py
Quick trainer that saves model.pkl and scaler.pkl without MLflow (for local testing).
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "data.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "model_artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
MODEL_PATH  = os.path.join(ARTIFACT_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")


def quick_train():
    csv_path = os.path.normpath(DATA_PATH)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"data.csv not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "id" in df.columns:
        df = df.drop(columns=["id"]) 
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    feature_cols = [c for c in df.columns if c != "diagnosis"]
    X = df[feature_cols].values
    y = df["diagnosis"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = LogisticRegression(C=1.0, max_iter=10000, solver="lbfgs", random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"[INFO] Model saved -> {MODEL_PATH}")
    print(f"[INFO] Scaler saved -> {SCALER_PATH}")
    print("[INFO] Classification report:")
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    quick_train()

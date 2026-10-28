"""
train_model.py
---------------
Trains the JobGuard AI fake-job-detection model.

Pipeline:
  1. Load job_dataset.csv (run dataset_generator.py first if missing)
  2. TF-IDF vectorize the combined text (title + description + company
     profile + requirements)
  3. Extract engineered "red flag" numeric features (URL + text signals)
  4. Concatenate TF-IDF (sparse) + scaled numeric features
  5. Train a RandomForestClassifier
  6. Evaluate on a held-out test split
  7. Save vectorizer, scaler and model to model/artifacts.joblib

Run:  python model/train_model.py
"""

import os
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
import joblib

from feature_engineering import extract_all_features, FEATURE_ORDER

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "job_dataset.csv")
ARTIFACT_PATH = os.path.join(HERE, "artifacts.joblib")


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "job_dataset.csv not found. Run `python dataset_generator.py` first."
        )
    return pd.read_csv(DATASET_PATH).fillna("")


def build_features(df: pd.DataFrame):
    full_text = (
        df["title"].astype(str) + " " +
        df["description"].astype(str) + " " +
        df["company_profile"].astype(str) + " " +
        df["requirements"].astype(str)
    )

    vectorizer = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 2), stop_words="english", min_df=2
    )
    X_text = vectorizer.fit_transform(full_text)

    engineered_rows = [
        extract_all_features(row.title, row.description, row.company_profile,
                              row.requirements, row.url)
        for row in df.itertuples()
    ]
    eng_df = pd.DataFrame(engineered_rows)[FEATURE_ORDER]

    scaler = StandardScaler()
    X_num = scaler.fit_transform(eng_df.values)

    X = hstack([X_text, csr_matrix(X_num)])
    return X, vectorizer, scaler


def main():
    print("Loading dataset...")
    df = load_dataset()
    print(f"Loaded {len(df)} rows. Class balance:\n{df['fraudulent'].value_counts()}")

    print("\nBuilding features (TF-IDF + engineered red-flag signals)...")
    X, vectorizer, scaler = build_features(df)
    y = df["fraudulent"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=42, n_jobs=-1, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== Evaluation on held-out test set =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted) [Real, Fake]:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    joblib.dump({
        "model": model,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
    }, ARTIFACT_PATH)
    print(f"\nSaved trained artifacts to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()

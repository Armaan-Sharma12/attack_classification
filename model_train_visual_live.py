import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import os
import time
from datetime import datetime
import argparse

# ======================================================
# GPU-Optimized IoT-23 Model Trainer with Live Tracking + Visualizations
# ======================================================
def train_xgboost_gpu_live(input_csv, model_output, result_root="results"):
    # Create timestamped results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(result_root, f"run_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    print(f"\n Results will be saved in: {result_dir}")

    print(f"[INFO] Loading dataset: {input_csv}")
    df = pd.read_csv(input_csv)

    if "Label" not in df.columns:
        raise ValueError(" 'Label' column not found in dataset!")

    print("[INFO] Preparing features and labels...")
    X = df.drop(columns=["Label"])
    y = df["Label"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Detect GPU availability
    use_gpu = xgb.core._has_cuda_support()
    print(f"[INFO] GPU detected: {use_gpu}")

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist" if use_gpu else "hist",
        "predictor": "gpu_predictor" if use_gpu else "cpu_predictor",
        "learning_rate": 0.05,
        "max_depth": 8,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 1
    }

    model = xgb.XGBClassifier(**params)

    print("[INFO] Starting training with live progress tracking...\n")
    start_time = time.time()

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        eval_metric=["logloss", "auc"],
        verbose=True  # shows progress every iteration
    )

    elapsed = time.time() - start_time
    print(f"\n Training complete in {elapsed:.2f} seconds")

    print("\n[INFO] Evaluating model on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_prob)

    print("\n========== RESULTS ==========")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=============================\n")

    # Save model
    model_path = os.path.join(result_dir, model_output)
    model.save_model(model_path)
    print(f"[INFO] 💾 Model saved to: {model_path}")

    # ---------- Visualizations ----------
    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "roc_curve.png"))
    plt.close()

    # Feature Importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10, importance_type="gain")
    plt.title("Top 10 Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "feature_importance.png"))
    plt.close()

    print(f"[INFO]  All visualizations saved inside: {result_dir}")
    print("[INFO]  Training and evaluation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IoT-23 XGBoost model (GPU + Live Progress + Visualizations)")
    parser.add_argument("--input", type=str, required=True, help="Path to engineered features CSV")
    parser.add_argument("--output", type=str, default="xgb_iot23_model.json", help="Output model filename")
    parser.add_argument("--result_root", type=str, default="results", help="Root folder for saving results")
    args = parser.parse_args()

    train_xgboost_gpu_live(args.input, args.output, args.result_root)

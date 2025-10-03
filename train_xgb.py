import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import argparse

def train_xgb(data_file, save_model):
    print("[INFO] Loading dataset...")
    df = pd.read_csv(data_file)

    X = df.drop("Label", axis=1)
    y = df["Label"]

    print("[INFO] Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("[INFO] Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="gpu_hist",  # GPU acceleration
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)

    print("[INFO] Evaluating...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"[INFO] Saving model to {save_model}")
    joblib.dump(model, save_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="CSV file from feature_engineer.py")
    parser.add_argument("--save", type=str, required=True, help="Path to save trained model")
    args = parser.parse_args()

    train_xgb(args.data, args.save)

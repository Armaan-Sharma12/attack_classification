# feature_engineer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

DATASET = "data/iot23_light/conn_dataset.csv"  # your IoT-23 CSV
SAVE_PATH = "data/processed/"

df = pd.read_csv(DATASET)

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

# Save label encoder
joblib.dump(le, SAVE_PATH + "label_encoder.pkl")

# Select numeric features only
numeric_cols = df.select_dtypes(include=['int64','float64']).columns
X = df[numeric_cols]
y = df["label"]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SAVE_PATH + "scaler.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Save
joblib.dump((X_train, X_test, y_train, y_test), SAVE_PATH + "tabular_data.pkl")

print(f"Processed data saved in {SAVE_PATH}")

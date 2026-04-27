import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from imblearn.over_sampling import SMOTE

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("survey lung cancer.csv")

# -----------------------------
# DATA CLEANING
# -----------------------------
df = df.drop_duplicates()

# -----------------------------
# TRANSFORMATION
# -----------------------------
df['GENDER'] = df['GENDER'].map({'M':1, 'F':0})
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES':1, 'NO':0})

# -----------------------------
# SPLIT FEATURES
# -----------------------------
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# -----------------------------
# NORMALIZATION
# -----------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# SMOTE (IMPORTANT)
# -----------------------------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# -----------------------------
# MODEL TRAINING (FIXED RF)
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train_smote, y_train_smote)

# -----------------------------
# TEST CHECK (IMPORTANT)
# -----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nSample Predictions:")
print(model.predict(X_test[:10]))   # 🔥 MUST NOT be all 1

# -----------------------------
# SAVE MODEL + SCALER
# -----------------------------
joblib.dump(model, "lung_cancer_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully")

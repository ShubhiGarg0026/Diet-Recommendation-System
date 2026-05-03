import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def run_models():
    print("\nSTARTING MODEL TRAINING")

    data = pd.read_csv("processed_dataset.csv")

    target_cols = data.columns[-3:]
    X = data.drop(columns=target_cols)
    y = data[target_cols].idxmax(axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # TRAIN MODELS
    # -------------------------
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)

    # -------------------------
    # SAVE MODELS (AFTER TRAINING)
    # -------------------------
    joblib.dump(rf, "FastAPI_Backend/rf_model.pkl")
    joblib.dump(svm, "FastAPI_Backend/svm_model.pkl")

    print("✅ Models trained and saved successfully")

if __name__ == "__main__":
    run_models()
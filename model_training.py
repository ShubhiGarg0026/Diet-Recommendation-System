import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVR

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

def prepare_data(df):
    df = df.dropna()

    X = df[[
        "Age", "BMI", "Blood_Pressure_Systolic",
        "Cholesterol_Level", "Blood_Sugar_Level"
    ]]

    y = df[[
        "Recommended_Calories",
        "Recommended_Protein",
        "Recommended_Carbs",
        "Recommended_Fats"
    ]]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_all_models(df):
    X_train, X_test, y_train, y_test = prepare_data(df)

    rf = RandomForestRegressor().fit(X_train, y_train)
    svm = SVR().fit(X_train, y_train["Recommended_Calories"])
    knn = NearestNeighbors(metric='cosine').fit(X_train)

    X = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
    y = np.array(y_train)

    lstm = Sequential([
        Input(shape=(1, X.shape[2])),
        LSTM(50, activation='relu'),
        Dense(4)
    ])
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X, y, epochs=10, verbose=0)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    joblib.dump(rf, os.path.join(BASE_DIR, "rf_model.pkl"))
    joblib.dump(svm, os.path.join(BASE_DIR, "svm_model.pkl"))
    joblib.dump(knn, os.path.join(BASE_DIR, "knn_model.pkl"))
    lstm.save(os.path.join(BASE_DIR, "lstm_model.keras"))

    print("Models trained & saved")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR, "Personalized_Diet_Recommendations.csv"))
    train_all_models(df)
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("Personalized_Diet_Recommendations.csv")

features = [
    "Age",
    "BMI",
    "Blood_Pressure_Systolic",
    "Cholesterol_Level",
    "Blood_Sugar_Level"
]

targets = [
    "Recommended_Calories",
    "Recommended_Protein",
    "Recommended_Carbs",
    "Recommended_Fats"
]

X = df[features].values
y = df[targets].values

# -------------------------
# SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# ✅ NORMALIZED METRICS (CORRECT)
# -------------------------
def normalized_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)

    max_val = np.max(y_true)

    # normalize MSE
    mse_norm = (mse / (max_val ** 2)) * 100

    # derive RMSE FROM normalized MSE
    rmse_norm = np.sqrt(mse_norm)

    return mse_norm, rmse_norm


# =========================
# RANDOM FOREST
# =========================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

rf.fit(X_train_s, y_train)
rf_pred = rf.predict(X_test_s)

rf_mse, rf_rmse = normalized_metrics(y_test, rf_pred)


# =========================
# KNN
# =========================
knn = KNeighborsRegressor(n_neighbors=7, weights='distance')
knn.fit(X_train_s, y_train)

knn_pred = knn.predict(X_test_s)

knn_mse, knn_rmse = normalized_metrics(y_test, knn_pred)


# =========================
# SVM (Calories only)
# =========================
svm = SVR(C=100, gamma=0.01, epsilon=0.1)

svm.fit(X_train_s, y_train[:, 0])
svm_pred = svm.predict(X_test_s)

svm_mse, svm_rmse = normalized_metrics(y_test[:, 0], svm_pred)


# =========================
# LSTM
# =========================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# reshape
X_train_l = X_train_l.reshape((X_train_l.shape[0], 1, X_train_l.shape[1]))
X_test_l = X_test_l.reshape((X_test_l.shape[0], 1, X_test_l.shape[1]))

# model
lstm = Sequential([
    LSTM(128, input_shape=(1, X_train_l.shape[2])),
    Dense(64, activation='relu'),
    Dense(4)
])

lstm.compile(optimizer='adam', loss='mse')

lstm.fit(X_train_l, y_train_l, epochs=50, batch_size=16, verbose=0)

y_pred_l = lstm.predict(X_test_l)

# inverse scaling
y_test_actual = scaler_y.inverse_transform(y_test_l)
y_pred_actual = scaler_y.inverse_transform(y_pred_l)

lstm_mse, lstm_rmse = normalized_metrics(y_test_actual, y_pred_actual)


# =========================
# RESULTS
# =========================
print("\nMODEL PERFORMANCE (Normalized 0–100)\n")

print("Random Forest")
print(f"MSE  : {rf_mse:.2f}")
print(f"RMSE : {rf_rmse:.2f}\n")

print("KNN")
print(f"MSE  : {knn_mse:.2f}")
print(f"RMSE : {knn_rmse:.2f}\n")

print("SVM (Calories)")
print(f"MSE  : {svm_mse:.2f}")
print(f"RMSE : {svm_rmse:.2f}\n")

print("LSTM")
print(f"MSE  : {lstm_mse:.2f}")
print(f"RMSE : {lstm_rmse:.2f}\n")


# =========================
# BEST MODEL
# =========================
results = {
    "RF": rf_rmse,
    "KNN": knn_rmse,
    "SVM": svm_rmse,
    "LSTM": lstm_rmse
}

best_model = min(results, key=results.get)

print(f"BEST MODEL: {best_model}")
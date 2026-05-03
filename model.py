import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

FEATURE_COLUMNS = [
    "Age",
    "BMI",
    "Blood_Pressure_Systolic",
    "Cholesterol_Level",
    "Blood_Sugar_Level"
]

TARGET_COLUMNS = [
    "Recommended_Calories",
    "Recommended_Protein",
    "Recommended_Carbs",
    "Recommended_Fats"
]


# -------------------------------
# FILTER
# -------------------------------
def apply_filters(df, allergies=None, dietary=None, cuisine=None):
    filtered = df.copy()

    for col in ["Allergies", "Dietary_Habits", "Preferred_Cuisine"]:
        if col in filtered.columns:
            filtered[col] = filtered[col].astype(str)

    if allergies:
        filtered = filtered[
            ~filtered["Allergies"].str.lower().str.contains(allergies.lower(), na=False)
        ]

    if dietary:
        filtered = filtered[
            filtered["Dietary_Habits"].str.lower().str.contains(dietary.lower(), na=False)
        ]

    if cuisine:
        filtered = filtered[
            filtered["Preferred_Cuisine"].str.lower().str.contains(cuisine.lower(), na=False)
        ]

    return filtered


# -------------------------------
# KNN MODEL
# -------------------------------
def knn_recommend(df, user_input):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLUMNS])

    model = NearestNeighbors(metric="cosine")
    model.fit(X)

    user = scaler.transform(np.array(user_input).reshape(1, -1))
    index = model.kneighbors(user, n_neighbors=1, return_distance=False)[0][0]

    return df.iloc[index]


# -------------------------------
# RANDOM FOREST
# -------------------------------
def rf_predict(df, user_input):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]

    model = RandomForestRegressor()
    model.fit(X, y)

    pred = model.predict([user_input])[0]

    return pred


# -------------------------------
# SVM
# -------------------------------
def svm_predict(df, user_input):
    X = df[FEATURE_COLUMNS]
    y = df["Recommended_Calories"]  # single target

    model = SVR()
    model.fit(X, y)

    pred = model.predict([user_input])[0]

    return pred


# -------------------------------
# FINAL FORMAT
# -------------------------------
def format_output(row):
    return {
        "Meal_Plan": str(row["Recommended_Meal_Plan"]),
        "Calories": int(row["Recommended_Calories"]),
        "Protein": int(row["Recommended_Protein"]),
        "Carbs": int(row["Recommended_Carbs"]),
        "Fats": int(row["Recommended_Fats"]),
        "Dietary_Habits": str(row["Dietary_Habits"]),
        "Preferred_Cuisine": str(row["Preferred_Cuisine"]),
        "Reason": f"Best match for Age {int(row['Age'])}, BMI {float(row['BMI'])}"
    }
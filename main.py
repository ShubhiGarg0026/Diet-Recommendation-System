from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
from FastAPI_Backend.model_visualization import (
    plot_single_model,
    plot_all_models_comparison,
    choose_best_model
)
from FastAPI_Backend.model import (
    knn_recommend,
    rf_predict,
    svm_predict,
    apply_filters,
    format_output
)
from FastAPI_Backend.lstm_model import build_lstm, predict_future
from FastAPI_Backend.visualization import plot_comparison

app = FastAPI()

# -------------------------------
# LOAD DATA
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "Personalized_Diet_Recommendations.csv")

dataset = pd.read_csv(dataset_path)

# -------------------------------
# INIT LSTM
# -------------------------------
lstm_model = build_lstm((3, 5))


# -------------------------------
# INPUT SCHEMA
# -------------------------------
class InputData(BaseModel):
    age: float
    bmi: float
    systolic_bp: float
    cholesterol: float
    sugar: float

    allergies: str | None = None
    dietary_habits: str | None = None
    preferred_cuisine: str | None = None

    model_type: str = "knn"   


@app.get("/")
def home():
    return {"message": "LSTM + KNN Diet Recommendation API Running"}


@app.post("/predict")
def predict(data: InputData):

    user_vector = [
        data.age,
        data.bmi,
        data.systolic_bp,
        data.cholesterol,
        data.sugar
    ]

    df = dataset.copy()

    # Apply filters
    df = apply_filters(
        df,
        allergies=data.allergies,
        dietary=data.dietary_habits,
        cuisine=data.preferred_cuisine
    )

    if df.empty:
        df = dataset.copy()

    try:
        # ===============================
        # 🔹 SELECTED MODEL OUTPUT
        # ===============================
        if data.model_type.lower() == "knn":
            row = knn_recommend(df, user_vector)
            result = format_output(row)

        elif data.model_type.lower() == "rf":
            pred = rf_predict(df, user_vector)
            result = {
                "Calories": float(pred[0]),
                "Protein": float(pred[1]),
                "Carbs": float(pred[2]),
                "Fats": float(pred[3])
            }

        elif data.model_type.lower() == "svm":
            pred = svm_predict(df, user_vector)
            result = {
                "Calories": float(pred)
            }

        elif data.model_type.lower() == "lstm":
            future_values = predict_future(lstm_model, user_vector)
            row = knn_recommend(df, future_values)
            result = format_output(row)

        else:
            return {"error": "Invalid model_type"}

        # ===============================
        # 🔥 ADD THIS BLOCK (AFTER MODELS)
        # ===============================

        models_output = {}

        # KNN
        row_knn = knn_recommend(df, user_vector)
        models_output["KNN"] = [
            row_knn["Recommended_Calories"],
            row_knn["Recommended_Protein"],
            row_knn["Recommended_Carbs"],
            row_knn["Recommended_Fats"]
        ]

        # Random Forest
        rf_pred = rf_predict(df, user_vector)
        models_output["Random Forest"] = list(rf_pred)

        # SVM
        svm_pred = svm_predict(df, user_vector)
        models_output["SVM"] = [svm_pred, 0, 0, 0]

        # LSTM
        future_values = predict_future(lstm_model, user_vector)
        row_lstm = knn_recommend(df, future_values)
        models_output["LSTM"] = [
            row_lstm["Recommended_Calories"],
            row_lstm["Recommended_Protein"],
            row_lstm["Recommended_Carbs"],
            row_lstm["Recommended_Fats"]
        ]

        # ===============================
        # 📊 GENERATE GRAPHS
        # ===============================
        for model_name, values in models_output.items():
            plot_single_model(model_name, values)

        plot_all_models_comparison(models_output)

        # ===============================
        # 🏆 BEST MODEL
        # ===============================
        best_model = choose_best_model(models_output, user_vector)

        # ===============================
        # FINAL RESPONSE
        # ===============================
        return {
            "selected_model": data.model_type,
            "best_model": best_model,
            "graphs_generated": [
                "KNN_graph.png",
                "Random Forest_graph.png",
                "SVM_graph.png",
                "LSTM_graph.png",
                "comparison_all_models.png"
            ],
            "recommendation": result
        }

    except Exception as e:
        return {"error": str(e)}
    
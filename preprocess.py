import pandas as pd
from sklearn.preprocessing import StandardScaler

def run_preprocessing():

    print("="*50)
    print("STARTING PREPROCESSING")

    # Load dataset
    df = pd.read_csv("Personalized_Diet_Recommendations.csv")

    print("Original Shape:", df.shape)

    # Drop ID column if it exists
    if "Patient_ID" in df.columns:
        df = df.drop(columns=["Patient_ID"])

    # Fill missing values
    if "Chronic_Disease" in df.columns:
        df["Chronic_Disease"] = df["Chronic_Disease"].fillna("None")

    if "Allergies" in df.columns:
        df["Allergies"] = df["Allergies"].fillna("None")

    if "Food_Aversions" in df.columns:
        df["Food_Aversions"] = df["Food_Aversions"].fillna("None")

    # Encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Scale numeric columns
    scaler = StandardScaler()
    numeric_cols = df_encoded.select_dtypes(include=['int64','float64']).columns
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

    # Save processed dataset
    df_encoded.to_csv("processed_dataset.csv", index=False)

    print("Preprocessing Completed Successfully")
    print("New Shape:", df_encoded.shape)
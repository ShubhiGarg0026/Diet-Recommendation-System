import streamlit as st
import requests
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Diet AI", layout="centered")

# ---------------- CUSTOM CSS (ANIMATION) ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00ffcc;
    animation: fadeIn 1.5s ease-in;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    box-shadow: 0px 0px 15px rgba(0,255,204,0.2);
    animation: slideUp 0.7s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
@keyframes slideUp {
    from {transform: translateY(40px); opacity: 0;}
    to {transform: translateY(0); opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🥗 Smart Diet Recommendation</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered personalized nutrition guidance</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
st.markdown("### 🧾 Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, value=25)
    bmi = st.number_input("BMI", 10.0, 50.0, value=24.0)
    bp = st.number_input("Systolic BP", 80, 200, value=120)

with col2:
    chol = st.number_input("Cholesterol", 100, 400, value=180)
    sugar = st.number_input("Blood Sugar", 50, 300, value=100)

st.markdown("### 🍽 Preferences")

col3, col4 = st.columns(2)

with col3:
    allergies = st.text_input("Allergies")
    diet = st.selectbox("Dietary Habits", ["vegetarian", "non-vegetarian", "vegan"])

with col4:
    cuisine = st.selectbox("Preferred Cuisine", ["indian", "western", "asian"])

# ---------------- BUTTON ----------------
if st.button("🚀 Generate Diet Plan"):

    data = {
        "age": age,
        "bmi": bmi,
        "systolic_bp": bp,
        "cholesterol": chol,
        "sugar": sugar,
        "allergies": allergies,
        "dietary_habits": diet,
        "preferred_cuisine": cuisine
    }

    # LOADING ANIMATION
    with st.spinner("Analyzing your health profile..."):
        time.sleep(1.5)
        res = requests.post("http://127.0.0.1:8000/predict", json=data)

    if res.status_code == 200:
        result = res.json()["recommendation"]

        # ---------------- OUTPUT CARD ----------------
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.success("✅ Recommendation Ready")

        st.markdown("### 🥗 Meal Plan")
        st.write(f"**{result['Meal_Plan']}**")

        st.markdown("### 📊 Nutrition Breakdown")
        col5, col6, col7 = st.columns(3)

        col5.metric("Calories", result["Calories"])
        col6.metric("Protein", result["Protein"])
        col7.metric("Carbs", result["Carbs"])

        st.metric("Fats", result["Fats"])

        st.markdown("### 🧠 Why this diet?")
        st.info(result["Reason"])

        st.markdown("### 🎯 Matched Preferences")
        st.write(f"Diet: {result['Dietary_Habits']}")
        st.write(f"Cuisine: {result['Preferred_Cuisine']}")

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.error("⚠️ Something went wrong")
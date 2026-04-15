import streamlit as st
import joblib
import numpy as np
import os

# ---------------------------
# Load model and scaler
# ---------------------------
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "lung_cancer_rf_model.pkl")
scaler_path = os.path.join(base_path, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Lung Cancer Prediction App", page_icon="🫁", layout="centered")

# ---------------------------
# Session state
# ---------------------------
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

# ---------------------------
# Sidebar navigation
# ---------------------------
page = st.sidebar.radio("Navigation", ["Home", "Prediction", "Result"])

# ---------------------------
# HOME PAGE
# ---------------------------
if page == "Home":
    st.title("🫁 Lung Cancer Prediction App")
    st.subheader("Welcome")
    st.write(
        """
        This application predicts whether a patient is likely to have lung cancer
        based on clinical and lifestyle-related input features.
        """
    )

    st.markdown("### App Flow")
    st.write("1. Go to **Prediction** page")
    st.write("2. Enter patient details")
    st.write("3. Click **Predict**")
    st.write("4. Check the output in **Result** page")

    st.markdown("### Model Used")
    st.write("Random Forest Classifier")

# ---------------------------
# PREDICTION PAGE
# ---------------------------
elif page == "Prediction":
    st.title("📝 Prediction Page")

    name = st.text_input("Enter Patient Name")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=100, value=50)
    smoking = st.selectbox("Smoking", [0, 1])
    yellow_fingers = st.selectbox("Yellow Fingers", [0, 1])
    anxiety = st.selectbox("Anxiety", [0, 1])
    peer_pressure = st.selectbox("Peer Pressure", [0, 1])
    chronic_disease = st.selectbox("Chronic Disease", [0, 1])
    fatigue = st.selectbox("Fatigue", [0, 1])
    allergy = st.selectbox("Allergy", [0, 1])
    wheezing = st.selectbox("Wheezing", [0, 1])
    alcohol = st.selectbox("Alcohol Consuming", [0, 1])
    coughing = st.selectbox("Coughing", [0, 1])
    shortness_breath = st.selectbox("Shortness of Breath", [0, 1])
    swallowing_difficulty = st.selectbox("Swallowing Difficulty", [0, 1])
    chest_pain = st.selectbox("Chest Pain", [0, 1])

    # Encode gender
    gender_value = 1 if gender == "Male" else 0

    if st.button("Predict"):
        features = np.array([[
            gender_value, age, smoking, yellow_fingers, anxiety,
            peer_pressure, chronic_disease, fatigue, allergy,
            wheezing, alcohol, coughing, shortness_breath,
            swallowing_difficulty, chest_pain
        ]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        st.session_state.prediction_result = int(prediction)
        st.session_state.patient_name = name

        st.success("Prediction completed. Now open the Result page.")

# ---------------------------
# RESULT PAGE
# ---------------------------
elif page == "Result":
    st.title("📊 Result Page")

    if st.session_state.prediction_result is None:
        st.warning("No prediction found. Please go to Prediction page first.")
    else:
        patient_name = st.session_state.patient_name if st.session_state.patient_name else "Patient"

        if st.session_state.prediction_result == 1:
            st.error(f"{patient_name} - Lung Cancer Positive")
        else:
            st.success(f"{patient_name} - Lung Cancer Negative")


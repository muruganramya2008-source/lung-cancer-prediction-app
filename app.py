
import streamlit as st
import joblib
import numpy as np

# Load model & scaler
import os

base_path = os.path.dirname(__file__)

model_path = os.path.join(base_path, "lung_cancer_rf_model.pkl")
scaler_path = os.path.join(base_path, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.title("Lung Cancer Prediction App")

# 👉 Name input
name = st.text_input("Enter Patient Name")

st.write("Enter patient details below:")

# Input fields
gender = st.selectbox("Gender (Male=1, Female=0)", [0, 1])
age = st.number_input("Age", min_value=1, max_value=100, value=50)
smoking = st.selectbox("Smoking (Yes=1, No=0)", [0, 1])
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

if st.button("Predict"):
    features = np.array([[gender, age, smoking, yellow_fingers, anxiety,
                          peer_pressure, chronic_disease, fatigue, allergy,
                          wheezing, alcohol, coughing, shortness_breath,
                          swallowing_difficulty, chest_pain]])

    # Apply preprocessing
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)[0]

    if prediction == 1:
        st.error(f"{name} - Lung Cancer Positive")
    else:
        st.success(f"{name} - Lung Cancer Negative")

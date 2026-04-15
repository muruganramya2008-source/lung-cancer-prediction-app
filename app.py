import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Lung Cancer Prediction App",
    page_icon="🫁",
    layout="centered"
)

# -----------------------------
# Background image
# -----------------------------
def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1579684385127-1ef15d508118?auto=format&fit=crop&w=1600&q=80");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        .main-card {
            background-color: rgba(255, 255, 255, 0.82);
            padding: 2rem;
            border-radius: 18px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.2);
            margin-top: 2rem;
        }

        .center-btn {
            display: flex;
            justify-content: center;
            margin-top: 2rem;
        }

        h1, h2, h3, p, label {
            color: black !important;
        }

        .result-box {
            background-color: rgba(255, 255, 255, 0.86);
            padding: 2rem;
            border-radius: 18px;
            text-align: center;
            margin-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# -----------------------------
# Load model and scaler
# -----------------------------
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "lung_cancer_rf_model.pkl")
scaler_path = os.path.join(base_path, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# -----------------------------
# Session state
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

# -----------------------------
# HOME PAGE
# -----------------------------
if st.session_state.page == "Home":
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align:center;'>Lung Cancer Prediction</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; font-size:20px;'>Click the button below to start prediction</p>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Prediction", use_container_width=True):
            st.session_state.page = "Prediction"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif st.session_state.page == "Prediction":
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown(
        "<h2 style='text-align:center;'>Prediction Page</h2>",
        unsafe_allow_html=True
    )

    name = st.text_input("Enter Patient Name")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=100, value=50)

    smoking = st.selectbox("Smoking", ["No", "Yes"])
    yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
    anxiety = st.selectbox("Anxiety", ["No", "Yes"])
    peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
    chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
    fatigue = st.selectbox("Fatigue", ["No", "Yes"])
    allergy = st.selectbox("Allergy", ["No", "Yes"])
    wheezing = st.selectbox("Wheezing", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consuming", ["No", "Yes"])
    coughing = st.selectbox("Coughing", ["No", "Yes"])
    shortness_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
    swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
    chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

    # Encode inputs
    gender_value = 1 if gender == "Male" else 0
    smoking_value = 1 if smoking == "Yes" else 0
    yellow_fingers_value = 1 if yellow_fingers == "Yes" else 0
    anxiety_value = 1 if anxiety == "Yes" else 0
    peer_pressure_value = 1 if peer_pressure == "Yes" else 0
    chronic_disease_value = 1 if chronic_disease == "Yes" else 0
    fatigue_value = 1 if fatigue == "Yes" else 0
    allergy_value = 1 if allergy == "Yes" else 0
    wheezing_value = 1 if wheezing == "Yes" else 0
    alcohol_value = 1 if alcohol == "Yes" else 0
    coughing_value = 1 if coughing == "Yes" else 0
    shortness_breath_value = 1 if shortness_breath == "Yes" else 0
    swallowing_difficulty_value = 1 if swallowing_difficulty == "Yes" else 0
    chest_pain_value = 1 if chest_pain == "Yes" else 0

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back to Home", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()

    with col2:
        if st.button("Predict", use_container_width=True):
            features = np.array([[
                gender_value,
                age,
                smoking_value,
                yellow_fingers_value,
                anxiety_value,
                peer_pressure_value,
                chronic_disease_value,
                fatigue_value,
                allergy_value,
                wheezing_value,
                alcohol_value,
                coughing_value,
                shortness_breath_value,
                swallowing_difficulty_value,
                chest_pain_value
            ]])

            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]

            st.session_state.patient_name = name if name.strip() else "Patient"
            st.session_state.prediction_result = int(prediction)
            st.session_state.page = "Result"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# RESULT PAGE
# -----------------------------
elif st.session_state.page == "Result":
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown(
        "<h2>Result Page</h2>",
        unsafe_allow_html=True
    )

    patient_name = st.session_state.patient_name
    result = st.session_state.prediction_result

    st.markdown(
        f"<h3>Patient Name: {patient_name}</h3>",
        unsafe_allow_html=True
    )

    if result == 1:
        st.error(f"{patient_name} has Lung Cancer.")
    else:
        st.success(f"{patient_name} does not have Lung Cancer.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Go to Home", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()

    with col2:
        if st.button("New Prediction", use_container_width=True):
            st.session_state.page = "Prediction"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

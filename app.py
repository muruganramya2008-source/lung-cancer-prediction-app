import streamlit as st
import joblib
import numpy as np
import os
import base64

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Lung Cancer Prediction App",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Convert local image to base64
# -----------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# -----------------------------
# Background + styling
# -----------------------------
def set_bg():
    base_path = os.path.dirname(__file__)
    image_path = os.path.join(base_path, "lungs_bg.jpg")
    bg_image = get_base64_image(image_path)

    st.markdown(
        f"""
        <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        .stDeployButton {{display:none;}}

        .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            max-width: 900px;
        }}

        .stApp {{
            background-image: url("data:image/webp;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .home-box {{
            background: rgba(255,255,255,0.82);
            padding: 2.5rem 2rem;
            border-radius: 22px;
            text-align: center;
            margin-top: 4rem;
            box-shadow: 0 6px 24px rgba(0,0,0,0.25);
        }}

        .page-title {{
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            color: black;
            margin-bottom: 0.8rem;
        }}

        .page-subtitle {{
            text-align: center;
            font-size: 1.3rem;
            color: black;
            margin-bottom: 1.5rem;
        }}

        .section-title {{
            text-align: center;
            font-size: 2.2rem;
            font-weight: 800;
            color: black;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }}

        .result-title {{
            text-align: center;
            font-size: 2.2rem;
            font-weight: 800;
            color: black;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }}

        .result-box {{
            background: rgba(255,255,255,0.82);
            padding: 2rem;
            border-radius: 20px;
            margin-top: 1rem;
            box-shadow: 0 6px 24px rgba(0,0,0,0.25);
        }}

        h1, h2, h3, label, p, div {{
            color: black !important;
        }}

        div[data-testid="stButton"] > button {{
            border-radius: 12px;
            font-size: 18px;
            font-weight: 600;
        }}
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
    st.markdown(
        """
        <div class="home-box">
            <div class="page-title">Lung Cancer Prediction</div>
            <div class="page-subtitle">Click the button below to start prediction</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Prediction", use_container_width=True):
            st.session_state.page = "Prediction"
            st.rerun()

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif st.session_state.page == "Prediction":
    st.markdown('<div class="section-title">Prediction Page</div>', unsafe_allow_html=True)

    name = st.text_input("Enter Patient Name")

    gender = st.selectbox("Gender (Male=1, Female=0)", [0, 1])
    age = st.number_input("Age", min_value=1, max_value=100, value=50)
    smoking = st.selectbox("Smoking (Yes=1, No=0)", [0, 1])
    yellow_fingers = st.selectbox("Yellow Fingers (Yes=1, No=0)", [0, 1])
    anxiety = st.selectbox("Anxiety (Yes=1, No=0)", [0, 1])
    peer_pressure = st.selectbox("Peer Pressure (Yes=1, No=0)", [0, 1])
    chronic_disease = st.selectbox("Chronic Disease (Yes=1, No=0)", [0, 1])
    fatigue = st.selectbox("Fatigue (Yes=1, No=0)", [0, 1])
    allergy = st.selectbox("Allergy (Yes=1, No=0)", [0, 1])
    wheezing = st.selectbox("Wheezing (Yes=1, No=0)", [0, 1])
    alcohol = st.selectbox("Alcohol Consuming (Yes=1, No=0)", [0, 1])
    coughing = st.selectbox("Coughing (Yes=1, No=0)", [0, 1])
    shortness_breath = st.selectbox("Shortness of Breath (Yes=1, No=0)", [0, 1])
    swallowing_difficulty = st.selectbox("Swallowing Difficulty (Yes=1, No=0)", [0, 1])
    chest_pain = st.selectbox("Chest Pain (Yes=1, No=0)", [0, 1])

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back to Home", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()

    with col2:
        if st.button("Predict", use_container_width=True):
            features = np.array([[
                gender,
                age,
                smoking,
                yellow_fingers,
                anxiety,
                peer_pressure,
                chronic_disease,
                fatigue,
                allergy,
                wheezing,
                alcohol,
                coughing,
                shortness_breath,
                swallowing_difficulty,
                chest_pain
            ]])

            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]

            st.session_state.patient_name = name if name.strip() else "Patient"
            st.session_state.prediction_result = int(prediction)
            st.session_state.page = "Result"
            st.rerun()

# -----------------------------
# RESULT PAGE
# -----------------------------
elif st.session_state.page == "Result":
    st.markdown('<div class="result-title">Result Page</div>', unsafe_allow_html=True)

    patient_name = st.session_state.patient_name
    result = st.session_state.prediction_result

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown(f"### Patient Name: {patient_name}")

    if result == 1:
        st.error(f"{patient_name} Yes:lung Cancer.")
    else:
        st.success(f"{patient_name} No:Lung Cancer.")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Go to Home", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()

    with col2:
        if st.button("New Prediction", use_container_width=True):
            st.session_state.page = "Prediction"
            st.rerun()

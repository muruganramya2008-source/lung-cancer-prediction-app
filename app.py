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
# Background + full UI styling
# -----------------------------
def set_bg():
    base_path = os.path.dirname(__file__)
    image_path = os.path.join(base_path, "lungs_bg.webp")
    bg_image = get_base64_image(image_path)

    st.markdown(
        f"""
        <style>
        /* Hide default Streamlit elements */
        #MainMenu {{
            visibility: hidden;
        }}

        footer {{
            visibility: hidden;
        }}

        header {{
            visibility: hidden;
            height: 0px;
        }}

        .stDeployButton {{
            display: none;
        }}

        /* Remove unnecessary top spacing */
        .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
        }}

        /* App background */
        .stApp {{
            background-image: url("data:image/webp;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Home card */
        .home-card {{
            background-color: rgba(255, 255, 255, 0.82);
            padding: 2.5rem 2rem;
            border-radius: 22px;
            box-shadow: 0 6px 24px rgba(0,0,0,0.25);
            text-align: center;
            max-width: 850px;
            margin: 3rem auto 2rem auto;
        }}

        /* Prediction card */
        .main-card {{
            background-color: rgba(255, 255, 255, 0.84);
            padding: 2rem;
            border-radius: 22px;
            box-shadow: 0 6px 24px rgba(0,0,0,0.25);
            max-width: 900px;
            margin: 2rem auto;
        }}

        /* Result card */
        .result-card {{
            background-color: rgba(255, 255, 255, 0.86);
            padding: 2rem;
            border-radius: 22px;
            box-shadow: 0 6px 24px rgba(0,0,0,0.25);
            text-align: center;
            max-width: 850px;
            margin: 3rem auto;
        }}

        /* Typography */
        .title-text {{
            font-size: 3rem;
            font-weight: 800;
            color: black;
            margin-bottom: 1rem;
        }}

        .subtitle-text {{
            font-size: 1.4rem;
            color: black;
            margin-bottom: 1.5rem;
        }}

        h1, h2, h3, p, label {{
            color: black !important;
        }}

        /* Buttons */
        div[data-testid="stButton"] > button {{
            border-radius: 12px;
            font-weight: 600;
            font-size: 18px;
            padding: 0.6rem 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background
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
        <div class="home-card">
            <div class="title-text">Lung Cancer Prediction</div>
            <div class="subtitle-text">Click the button below to start prediction</div>
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
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown(
        "<h2 style='text-align:center;'>Prediction Page</h2>",
        unsafe_allow_html=True
    )

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

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# RESULT PAGE
# -----------------------------
elif st.session_state.page == "Result":
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("<h2>Result Page</h2>", unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)

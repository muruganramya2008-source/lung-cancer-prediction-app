import streamlit as st
import joblib
import numpy as np
import os
import base64

st.set_page_config(
    page_title="Lung Cancer Prediction App",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Background
# -----------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def set_bg():
    base_path = os.path.dirname(__file__)
    image_path = os.path.join(base_path, "lungs_bg.webp")

    if os.path.exists(image_path):
        bg_image = get_base64_image(image_path)

        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{bg_image}");
            background-size: cover;
        }}

        .top-card {{
            background: rgba(255,255,255,0.85);
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            margin-top: 2rem;
        }}
        </style>
        """, unsafe_allow_html=True)

set_bg()

# -----------------------------
# Load Model
# -----------------------------
base_path = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_path, "lung_cancer_rf_model.pkl"))
scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))

# -----------------------------
# Session State
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
    st.markdown("""
    <div class="top-card">
        <h1>Lung Cancer Prediction</h1>
        <p>Click below to start prediction</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Start Prediction"):
        st.session_state.page = "Prediction"
        st.rerun()

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif st.session_state.page == "Prediction":
    st.markdown("""
    <div class="top-card">
        <h2>Prediction Page</h2>
    </div>
    """, unsafe_allow_html=True)

    name = st.text_input("Enter Patient Name")

    gender = st.selectbox("Gender (Male=1, Female=0)", [0, 1])
    age = st.number_input("Age", 1, 100, 50)
    smoking = st.selectbox("Smoking", [0, 1])
    yellow_fingers = st.selectbox("Yellow Fingers", [0, 1])
    anxiety = st.selectbox("Anxiety", [0, 1])
    peer_pressure = st.selectbox("Peer Pressure", [0, 1])
    chronic_disease = st.selectbox("Chronic Disease", [0, 1])
    fatigue = st.selectbox("Fatigue", [0, 1])
    allergy = st.selectbox("Allergy", [0, 1])
    wheezing = st.selectbox("Wheezing", [0, 1])
    alcohol = st.selectbox("Alcohol", [0, 1])
    coughing = st.selectbox("Coughing", [0, 1])
    shortness_breath = st.selectbox("Shortness of Breath", [0, 1])
    swallowing_difficulty = st.selectbox("Swallowing Difficulty", [0, 1])
    chest_pain = st.selectbox("Chest Pain", [0, 1])

    # 🔥 MODEL SELECTION (NEW)
    st.markdown("### Select Model")
    selected_model = st.selectbox(
        "Choose Model",
        ["Random Forest"]   # Only RF
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back"):
            st.session_state.page = "Home"
            st.rerun()

    with col2:
        if st.button("Predict"):
            if selected_model == "Random Forest":

                features = np.array([[
                    gender, age, smoking, yellow_fingers, anxiety,
                    peer_pressure, chronic_disease, fatigue, allergy,
                    wheezing, alcohol, coughing, shortness_breath,
                    swallowing_difficulty, chest_pain
                ]])

                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]

                st.session_state.patient_name = name if name else "Patient"
                st.session_state.prediction_result = int(prediction)
                st.session_state.page = "Result"
                st.rerun()

# -----------------------------
# RESULT PAGE
# -----------------------------
elif st.session_state.page == "Result":
    name = st.session_state.patient_name
    result = st.session_state.prediction_result

    st.markdown(f"""
    <div class="top-card">
        <h2>Result Page</h2>
        <h3>Patient: {name}</h3>
    </div>
    """, unsafe_allow_html=True)

    if result == 1:
        st.error("Lung Cancer Detected")
    else:
        st.success("No Lung Cancer")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Home"):
            st.session_state.page = "Home"
            st.rerun()

    with col2:
        if st.button("New Prediction"):
            st.session_state.page = "Prediction"
            st.rerun()

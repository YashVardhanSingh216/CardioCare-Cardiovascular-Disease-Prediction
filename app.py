
import streamlit as st
import pandas as pd
import joblib


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="CardioCare | Health Analytics",
    page_icon="❤️",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #0B1120;
    color: #E2E8F0;
}

/* Main content */
[data-testid="stMain"] {
    background-color: #0B1120;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111C30;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

/* Main headings */
.main-title {
    font-size: 36px;
    font-weight: 750;
    color: #F8FAFC !important;
}

.subtitle {
    color: #A0AEC0 !important;
    font-size: 16px;
}

.section-title {
    font-size: 21px;
    font-weight: 650;
    color: #93C5FD !important;
    margin-bottom: 15px;
}

/* General text */
.stApp p,
.stApp label,
.stApp .stMarkdown {
    color: #E2E8F0;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #172338;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #293B55;
}

[data-testid="stMetricLabel"] {
    color: #A0AEC0 !important;
}

[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
}

/* Input fields */
.stNumberInput input {
    background-color: #172338;
    color: #F8FAFC;
    border: 1px solid #334155;
}

[data-baseweb="select"] > div {
    background-color: #172338;
    color: #F8FAFC;
    border-color: #334155;
}

/* Buttons */
div.stButton > button,
div.stFormSubmitButton > button {
    background: #0D9488;
    color: white;
    border-radius: 9px;
    border: none;
    height: 48px;
    font-weight: 600;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #0F766E;
    color: white;
}

/* Dividers */
hr {
    border-color: #293B55;
}

/* Info and warning boxes */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Progress bar */
.stProgress > div > div > div {
    background-color: #0D9488;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

import joblib
from pathlib import Path
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent


# Load preprocessing pipeline
preprocessor = joblib.load(
    BASE_DIR / "models" / "cardio_preprocessor.pkl"
)

# Load XGBoost model
model = xgb.XGBClassifier()
model.load_model(BASE_DIR / "models" / "cardio_xgb.json")

# Load threshold
threshold = joblib.load(
    BASE_DIR / "models" / "threshold.pkl"
)
# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("# ❤️ CardioCare")
    st.caption("HEALTH ANALYTICS PLATFORM")

    st.divider()

    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        ["Dashboard", "About Project"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Model Information")
    st.write("Algorithm: XGBoost")
    st.write(f"Threshold: {threshold}")
    st.write("ROC-AUC: 0.8009")

    st.divider()
    st.caption("Machine Learning Project | 2026")

# ---------------- ABOUT PAGE ----------------

if page == "About Project":

    st.markdown(
    '<div class="main-title">Cardiovascular Disease Prediction</div>',
    unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Predict whether the model classifies an '
        'individual as having cardiovascular disease (positive class) '
        'or not (negative class), using medical measurements and '
        'lifestyle information.</p>',
        unsafe_allow_html=True
    )

    st.info(
        "What does this application predict? "
        "It predicts the cardiovascular disease class (0 or 1) "
        "using a trained XGBoost machine learning model."
    )

    st.markdown("### Project Overview")
    st.write(
        "This application uses a trained XGBoost classification "
        "pipeline to predict the target class from patient-related "
        "measurements and lifestyle information."
    )

    st.markdown("### Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", "71.93%")
    c2.metric("Precision", "68.72%")
    c3.metric("Recall", "80.45%")
    c4.metric("ROC-AUC", "80.09%")


# ---------------- DASHBOARD PAGE ----------------

else:

    st.markdown(
        '<div class="main-title">Cardiovascular Health Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Enter the information below to generate '
        'a machine learning prediction.</p>',
        unsafe_allow_html=True
    )

    st.write("")

    # TOP METRICS

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction Model", "XGBoost")

    with col2:
        st.metric("Model ROC-AUC", "80.09%")

    with col3:
        st.metric("Decision Threshold", f"{threshold:.2f}")

    st.divider()

    # INPUT FORM

    with st.form("prediction_form"):

        st.markdown(
            '<div class="section-title">01. Personal Information</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input(
                "Age (years)", 18, 100, 45
            )

        with c2:
            gender = st.selectbox(
                "Gender",
                [1, 2],
                format_func=lambda x:
                "Female" if x == 1 else "Male"
            )

        with c3:
            height = st.number_input(
                "Height (cm)", 100, 230, 170
            )

        weight = st.number_input(
            "Weight (kg)", 30.0, 250.0, 70.0
        )

        bmi = weight / ((height / 100) ** 2)

        st.info(f"Calculated BMI: {bmi:.2f}")

        st.divider()

        st.markdown(
            '<div class="section-title">02. Medical Measurements</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:
            ap_hi = st.number_input(
                "Systolic Blood Pressure", 60, 250, 120
            )

            cholesterol = st.selectbox(
                "Cholesterol",
                [1, 2, 3],
                format_func=lambda x: {
                    1: "Normal",
                    2: "Above Normal",
                    3: "Well Above Normal"
                }[x]
            )

        with c2:
            ap_lo = st.number_input(
                "Diastolic Blood Pressure", 40, 200, 80
            )

            gluc = st.selectbox(
                "Glucose",
                [1, 2, 3],
                format_func=lambda x: {
                    1: "Normal",
                    2: "Above Normal",
                    3: "Well Above Normal"
                }[x]
            )

        st.divider()

        st.markdown(
            '<div class="section-title">03. Lifestyle Information</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            smoke = st.selectbox(
                "Smoking", [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

        with c2:
            alco = st.selectbox(
                "Alcohol Consumption", [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

        with c3:
            active = st.selectbox(
                "Physical Activity", [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

        st.write("")

        submitted = st.form_submit_button(
            "Generate Prediction",
            use_container_width=True
        )

    # ---------------- PREDICTION RESULT ----------------

    if submitted:

        if ap_hi <= ap_lo:
            st.error(
                "Please check your blood pressure values. "
                "Systolic pressure should normally be higher "
                "than diastolic pressure."
            )

        else:

            input_data = pd.DataFrame([{
                "age": age,
                "gender": gender,
                "height": height,
                "weight": weight,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "cholesterol": cholesterol,
                "gluc": gluc,
                "smoke": smoke,
                "alco": alco,
                "active": active,
                "bmi": bmi
            }])

            # Preprocess user input
            processed_data = preprocessor.transform(input_data)

            # Predict probability
            probability = model.predict_proba(processed_data)[0][1]

            # Apply threshold
            prediction = int(probability >= threshold)

            st.divider()
            st.markdown("## Prediction Results")

            c1, c2 = st.columns(2)

            with c1:
                st.metric("Calculated BMI", f"{bmi:.2f}")

            with c2:
                st.metric(
                    "Model Score for Positive Class",
                    f"{probability * 100:.2f}%"
                )

            st.progress(float(probability))

            if prediction == 1:
                st.error(
                    "Positive class predicted by the model."
                )
            else:
                st.success(
                    "Negative class predicted by the model."
                )

            st.caption(
                f"Classification threshold: {threshold:.2f}"
            )

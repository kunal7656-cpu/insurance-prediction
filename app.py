
from pathlib import Path
import __main__

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="CarePredict | Insurance AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# 2. PROJECT PATH
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_DIR / "insurance_prediction_pipeline.pkl"


# --------------------------------------------------
# 3. EXACT FEATURES USED DURING TRAINING
# --------------------------------------------------
feature_names = [
    "age",
    "is_felmale",
    "bmi",
    "children",
    "is_chainsmoker",
    "region_northeast",
    "region_northwest",
    "region_southeast",
    "region_southwest",
    "bmi_category_underweight",
    "bmi_category_Normal",
    "bmi_category_Overweight",
    "bmi_category_Obese"
]


# --------------------------------------------------
# 4. FEATURE PREPARATION FUNCTION
# IMPORTANT: Keep this function name unchanged.
# --------------------------------------------------
def prepare_features(data):
    if isinstance(data, pd.DataFrame):
        return data[feature_names]

    return pd.DataFrame(data, columns=feature_names)


# Register the function for compatibility with
# the saved pipeline's __main__ reference.
__main__.prepare_features = prepare_features


# --------------------------------------------------
# 5. LOAD SAVED PIPELINE
# --------------------------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)


try:
    insurance_pipeline = load_pipeline()

except FileNotFoundError:
    st.error(
        "Pipeline file not found. "
        "Please check insurance_prediction_pipeline.pkl."
    )
    st.stop()

except Exception as error:
    st.error("The saved pipeline could not be loaded.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# 6. FRONTEND DESIGN
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071b17, #102c25);
    color: #f0f7f3;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 30px;
    border: 1px solid #315b4b;
    border-radius: 20px;
    background: linear-gradient(120deg, #123b2f, #1c4d3c);
    margin-bottom: 25px;
}

.hero h1 {
    color: #eafff1;
    font-size: 38px;
    margin-bottom: 8px;
}

.hero p {
    color: #c5ded0;
    font-size: 16px;
}

div[data-testid="stForm"] {
    background: #102820;
    padding: 25px;
    border: 1px solid #315b4b;
    border-radius: 18px;
}

div[data-testid="stMetric"] {
    background: #123b2f;
    border: 1px solid #3a765b;
    padding: 22px;
    border-radius: 16px;
}

div[data-testid="stMetricLabel"] {
    color: #c5ded0;
}

div[data-testid="stMetricValue"] {
    color: #8ff0b0;
}

.stButton > button {
    width: 100%;
    background: #38a169;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #2f855a;
    color: white;
    border: 1px solid #8ff0b0;
}

.footer {
    color: #a7c4b4;
    text-align: center;
    padding-top: 30px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 7. HERO SECTION
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🌿 CarePredict</h1>
    <p>
        AI-Powered Medical Insurance Charges Prediction
    </p>
    <p>
        Enter your details below to estimate your
        medical insurance charges using a trained
        Machine Learning model.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 8. INPUT FORM
# --------------------------------------------------
st.subheader("Enter Your Details")

with st.form(key="carepredict_form_v2"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=25,
            step=1
        )

        sex = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        bmi = st.number_input(
            "BMI (Body Mass Index)",
            min_value=10.0,
            max_value=70.0,
            value=25.0,
            step=0.1,
            format="%.1f"
        )

    with col2:
        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        smoker = st.selectbox(
            "Smoker",
            ["No", "Yes"]
        )

        region = st.selectbox(
            "Region",
            ["Northeast", "Northwest", "Southeast", "Southwest"]
        )

    submitted = st.form_submit_button(
        "Predict Insurance Charges"
    )


# --------------------------------------------------
# 9. PREPARE INPUT FEATURES
# --------------------------------------------------
def make_features(age, sex, bmi, children, smoker, region):

    # Binary encoding
    is_felmale = 1 if sex == "Female" else 0
    is_chainsmoker = 1 if smoker == "Yes" else 0

    # Region one-hot encoding
    region_northeast = 1 if region == "Northeast" else 0
    region_northwest = 1 if region == "Northwest" else 0
    region_southeast = 1 if region == "Southeast" else 0
    region_southwest = 1 if region == "Southwest" else 0

    # BMI category encoding
    bmi_category_underweight = 1 if bmi < 18.5 else 0
    bmi_category_Normal = 1 if 18.5 <= bmi < 25 else 0
    bmi_category_Overweight = 1 if 25 <= bmi < 30 else 0
    bmi_category_Obese = 1 if bmi >= 30 else 0

    # Keep exact training feature order
    input_data = {
        "age": age,
        "is_felmale": is_felmale,
        "bmi": bmi,
        "children": children,
        "is_chainsmoker": is_chainsmoker,
        "region_northeast": region_northeast,
        "region_northwest": region_northwest,
        "region_southeast": region_southeast,
        "region_southwest": region_southwest,
        "bmi_category_underweight": bmi_category_underweight,
        "bmi_category_Normal": bmi_category_Normal,
        "bmi_category_Overweight": bmi_category_Overweight,
        "bmi_category_Obese": bmi_category_Obese
    }

    return pd.DataFrame(
        [input_data],
        columns=feature_names
    )


# --------------------------------------------------
# 10. PREDICTION
# --------------------------------------------------
if submitted:

    input_df = make_features(
        age=age,
        sex=sex,
        bmi=bmi,
        children=children,
        smoker=smoker,
        region=region
    )

    try:
        prediction = float(
            insurance_pipeline.predict(input_df)[0]
        )

        st.markdown("---")
        st.subheader("Your Prediction Result")

        st.success("Prediction completed successfully!")

        st.metric(
            label="Estimated Insurance Charges",
            value=f"₹{prediction:,.2f}"
        )

        st.caption(
            "This is a machine-learning estimate, "
            "not an official insurance quote."
        )

    except Exception as error:
        st.error("Prediction failed. Please check the error below.")
        st.exception(error)


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    CarePredict | Random Forest Machine Learning Project
</div>
""", unsafe_allow_html=True)

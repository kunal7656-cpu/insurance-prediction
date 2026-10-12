
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# IMPORTANT: Define this at the top level before loading the model
def prepare_features(X):
    columns = [
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
    return X[columns]


st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="🏥"
)

st.title("🏥 Insurance Charges Prediction")
st.write("Enter your details to predict insurance charges.")

# Load the saved pipeline
MODEL_PATH = Path(__file__).parent / "insurance_prediction_pipeline.pkl"

try:
    pipeline = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error("Model could not be loaded.")
    st.code(str(e))
    st.stop()

# Input form
with st.form("insurance_form"):
    age = st.number_input("Age", 18, 100, 30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    bmi = st.number_input("BMI", 10.0, 70.0, 28.5, step=0.1)
    children = st.number_input("Number of Children", 0, 10, 0)
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )

    submitted = st.form_submit_button("Predict Charges")

if submitted:
    # Encoding assumptions must match training
    is_felmale = int(sex == "Female")
    is_chainsmoker = int(smoker == "Yes")

    # Use the exact BMI category boundaries from your training code
    if bmi < 18.5:
        category = "underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    input_data = pd.DataFrame([{
        "age": age,
        "is_felmale": is_felmale,
        "bmi": bmi,
        "children": children,
        "is_chainsmoker": is_chainsmoker,
        "region_northeast": int(region == "northeast"),
        "region_northwest": int(region == "northwest"),
        "region_southeast": int(region == "southeast"),
        "region_southwest": int(region == "southwest"),
        "bmi_category_underweight": int(category == "underweight"),
        "bmi_category_Normal": int(category == "Normal"),
        "bmi_category_Overweight": int(category == "Overweight"),
        "bmi_category_Obese": int(category == "Obese")
    }])

    try:
        result = pipeline.predict(input_data)[0]
        st.success("Prediction completed!")
        st.metric("Estimated Insurance Charges", f"${result:,.2f}")
    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))


from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).parent / "best_random_forest.pkl"
model = joblib.load(MODEL_PATH)

print("Random Forest model loaded successfully!")

st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Insurance Charges Prediction")
st.write("Enter your details to predict your medical insurance charges.")

# User inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

sex = st.selectbox("Sex", ["Male", "Female"])

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=28.5,
    step=0.1
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox("Smoker", ["No", "Yes"])

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# Predict button
if st.button("Predict Insurance Charges"):

    # Encode categorical inputs
    is_felmale = 1 if sex == "Female" else 0
    is_chainsmoker = 1 if smoker == "Yes" else 0

    # BMI categories — use the SAME boundaries as training
    if bmi < 18.5:
        bmi_category = "underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    # Create all 13 features in the exact training order
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
        "bmi_category_underweight": int(
            bmi_category == "underweight"
        ),
        "bmi_category_Normal": int(
            bmi_category == "Normal"
        ),
        "bmi_category_Overweight": int(
            bmi_category == "Overweight"
        ),
        "bmi_category_Obese": int(
            bmi_category == "Obese"
        )
    }])

    try:
        prediction = pipeline.predict(input_data)[0]

        st.success("Prediction completed!")
        
        
        
        st.write("Estimated Insurance Charges")
        st.subheader(f"₹{prediction:,.2f}")


        st.caption(
            "This is a model estimate, not an actual insurance quote."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

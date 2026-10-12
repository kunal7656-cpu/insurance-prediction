
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Load your already-trained Random Forest
best_model = joblib.load("best_random_forest.pkl")

# Exact feature order used during training
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

# Validate input columns and order
def prepare_features(data):
    if isinstance(data, pd.DataFrame):
        return data[feature_names]

    return pd.DataFrame(data, columns=feature_names)

# Create prediction pipeline
insurance_pipeline = Pipeline([
    ("prepare_features", FunctionTransformer(
        prepare_features, validate=False
    )),
    ("model", best_model)
])

# Save complete pipeline
joblib.dump(
    insurance_pipeline,
    "insurance_prediction_pipeline.pkl"
)

print("Insurance prediction pipeline saved successfully!")


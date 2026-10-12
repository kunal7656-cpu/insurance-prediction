
from pathlib import Path
import joblib

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "best_random_forest.pkl"
)

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}"
        )

    loaded_model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
    return loaded_model

model = load_model()

def predict_charges(input_data):
    prepared_data = prepare_features(input_data)
    print("Prepared feature columns:", prepared_data.columns.tolist())
    print("Prepared feature shape:", prepared_data.shape)

    prediction = model.predict(prepared_data)
    return float(prediction[0])
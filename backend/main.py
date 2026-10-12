
from fastapi import FastAPI, HTTPException
from backend.schemas import InsuranceInput, PredictionResponse
from backend.predictor import predict_charges
import pandas as pd

app = FastAPI(
    title="Insurance Charges Prediction API",
    description="Predict insurance charges using a trained model",
    version="1.0.0"
)

@app.get("/")
def home():
    return {"message": "Insurance Backend is Running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: InsuranceInput):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        charges = predict_charges(input_df)

        return PredictionResponse(
            predicted_charges=round(charges, 2),
            currency="USD"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {str(e)}"
        )

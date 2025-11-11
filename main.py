
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Any

app = FastAPI(
    title="Chemotherapy Toxicity Predictor API",
    description="Serves a binary (0/1) prediction model for severe toxicity events.",
    version="1.0"
)


MODEL_FILE = 'chemo_toxicity_predictor_final.joblib'
try:
    loaded_model = joblib.load(MODEL_FILE)
    print(f"Model '{MODEL_FILE}' loaded successfully.")
except FileNotFoundError:
    print(f"--- FATAL ERROR: Model file '{MODEL_FILE}' not found. ---")
    print("Please make sure the .joblib file is in the same directory as main.py")
    loaded_model = None



class PatientVitals(BaseModel):
    feature_0: float
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float
    feature_10: float
    feature_11: float
    feature_12: float
    feature_13: float
    feature_14: float
    feature_15: float
    feature_16: float
    feature_17: float
    feature_18: float
    feature_19: float
    feature_20: float
    feature_21: float
    feature_22: float
    feature_23: float
    feature_24: float


    class Config:
        schema_extra = {
            "example": {
                'feature_0': 1.1, 'feature_1': -0.5, 'feature_2': 2.3, 'feature_3': 0.1,
                'feature_4': -1.2, 'feature_5': 0.8, 'feature_6': -0.2, 'feature_7': 1.4,
                'feature_8': 0.0, 'feature_9': -2.0, 'feature_10': 0.5, 'feature_11': 0.7,
                'feature_12': -1.1, 'feature_13': 0.3, 'feature_14': 1.9, 'feature_15': -0.8,
                'feature_16': 1.0, 'feature_17': -1.5, 'feature_18': 0.2, 'feature_19': -0.1,
                'feature_20': 0.9, 'feature_21': -1.3, 'feature_22': 0.6, 'feature_23': -0.4,
                'feature_24': 1.7
            }
        }



@app.post("/predict/")
async def predict_toxicity(vitals: PatientVitals):
    if loaded_model is None:
        return {"error": "Model is not loaded. Cannot make predictions."}


    patient_df = pd.DataFrame([vitals.dict()])


    prediction_array = loaded_model.predict(patient_df)
    probability_array = loaded_model.predict_proba(patient_df)


    prediction_result = int(prediction_array[0])
    probability_of_severe_event = float(probability_array[0][1])


    return {
        "prediction_code": prediction_result,
        "prediction_label": "Severe Toxicity Event is PREDICTED" if prediction_result == 1 else "Severe Toxicity Event is NOT PREDICTED",
        "confidence_score": f"{probability_of_severe_event:.2%}"
    }



@app.get("/")
async def read_root():
    return {"message": "Chemotherapy Toxicity Predictor API is running."}
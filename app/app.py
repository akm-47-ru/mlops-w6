import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Create the FastAPI app object
app = FastAPI(
    title="Iris Species Prediction API",
    description="A simple API to classify Iris flower species based on their measurements.",
    version="1.0.0",
)

class FlowerMeasurements(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

PRE_TRAINED_MODEL_PATH = "artifacts/model.joblib"

try:
    classifier = joblib.load(PRE_TRAINED_MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model artifact not found at {PRE_TRAINED_MODEL_PATH}")


@api.get("/", tags=["Status"])
def get_server_status():
    return {"status": "ok", "message": "Iris Classifier API is up and running."}


@api.post("/predict", tags=["Predictions"])
def classify_iris_species(measurements: FlowerMeasurements):
    input_features = measurements.dict()
    input_dataframe = pd.DataFrame([input_features])
    predicted_class = classifier.predict(input_dataframe)[0]
    prediction_probabilities = classifier.predict_proba(input_dataframe)[0]
    confidence_mapping = {
        species: float(probability)
        for species, probability in zip(classifier.classes_, prediction_probabilities)
    }
    return {
        "prediction_result": {"species_name": predicted_class},
        "confidence": confidence_mapping,
    }

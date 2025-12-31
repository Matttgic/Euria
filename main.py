from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np

app = FastAPI()

# Load model (XGBoost natif, PAS sklearn)
model = xgb.Booster()
model.load_model("model.json")

class PredictRequest(BaseModel):
    features: list[list[float]]

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array(req.features)
    dmatrix = xgb.DMatrix(X)
    preds = model.predict(dmatrix)
    return {"predictions": preds.tolist()}

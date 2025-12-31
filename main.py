from fastapi import FastAPI
import xgboost as xgb
import numpy as np

app = FastAPI()

# Load model
model = xgb.Booster()
model.load_model("model.json")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: list):
    X = np.array(features)
    dmatrix = xgb.DMatrix(X)
    preds = model.predict(dmatrix)
    return {"predictions": preds.tolist()}

from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import joblib  # pour sauvegarder/charger le modèle sklearn

app = FastAPI()

# Charger le modèle sklearn XGBoost
try:
    model = joblib.load("model_sklearn.pkl")  # ton modèle doit être sauvegardé via joblib.dump(model, "model_sklearn.pkl")
except Exception as e:
    print("Erreur lors du chargement du modèle:", e)

class PredictRequest(BaseModel):
    features: list[list[float]]

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        X = np.array(req.features)
        preds = model.predict(X)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}

from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import joblib
import requests

app = FastAPI()

# Charger le modèle sklearn XGBoost
try:
    model = joblib.load("model_sklearn.pkl")  # Assure-toi que le fichier est dans ton repo
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

@app.get("/fixtures")
def fetch_fixtures():
    API_KEY = "TA_CLE_API_FOOTBALL"  # Remplace par ta vraie clé
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": API_KEY}
    params = {
        "league": "39",  # Premier League
        "season": "2025",
        "next": 10       # récupère les 10 prochains matchs
    }
    r = requests.get(url, headers=headers, params=params)
    return r.json()

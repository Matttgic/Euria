from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import joblib
import requests

app = FastAPI()

# Charger le modèle sklearn XGBoost
try:
    model = joblib.load("model_sklearn.pkl")  # Assure-toi que ce fichier est dans le repo
except Exception as e:
    print("Erreur lors du chargement du modèle:", e)

# --- Config API-Football ---
API_FOOTBALL_KEY = "TA_CLE_API_FOOTBALL_ICI"  # Remplace par ta clé
LEAGUE_ID = 39  # Premier League
SEASON = 2025

# --- Classes pour les requêtes ---
class PredictRequest(BaseModel):
    features: list[list[float]]

# --- Endpoints ---
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

@app.get("/predict-fixtures")
def predict_fixtures():
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    url = f"https://v3.football.api-sports.io/fixtures?league={LEAGUE_ID}&season={SEASON}&next=5"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    matches = []

    for fixture in data.get("response", []):
        home_team = fixture["teams"]["home"]["name"]
        away_team = fixture["teams"]["away"]["name"]
        # Odds Bet365
        odds = fixture.get("odds", {}).get("Bet365", {})
        home_odd = odds.get("home", None)
        draw_odd = odds.get("draw", None)
        away_odd = odds.get("away", None)
        # Construire les features dans l’ordre attendu par ton modèle
        features = [home_odd, draw_odd, away_odd]  # + autres stats si nécessaire

        # Prédiction
        try:
            X = np.array([features])
            pred = model.predict(X)[0]
        except Exception as e:
            pred = None

        matches.append({
            "home": home_team,
            "away": away_team,
            "prediction": int(pred) if pred is not None else None
        })

    return {"fixtures": matches}

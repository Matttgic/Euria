import os
import requests
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ============================
# Charger le modèle
# ============================
MODEL_PATH = "model_sklearn.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print("Erreur lors du chargement du modèle:", e)
    model = None

# ============================
# API-Football
# ============================
API_KEY = os.environ.get("API_FOOTBALL_KEY")
API_BASE_URL = "https://v3.football.api-sports.io"
LEAGUE_ID = 39  # Premier League

class PredictRequest(BaseModel):
    match_id: int

def get_team_stats(team_id, season):
    url = f"{API_BASE_URL}/teams/statistics?team={team_id}&league={LEAGUE_ID}&season={season}"
    r = requests.get(url, headers={"x-apisports-key": API_KEY}).json()["response"]
    stats = {
        "gf": float(r["goals"]["for"]["average"]["total"]),
        "ga": float(r["goals"]["against"]["average"]["total"]),
        "cs": r["clean_sheet"]["total"],
        "fw": r["form"][-5:].count("W"),
        "fd": r["form"][-5:].count("D"),
        "fl": r["form"][-5:].count("L"),
    }
    return stats

def get_match_odds(match_id):
    url = f"{API_BASE_URL}/odds?fixture={match_id}&bookmaker=8"  # 8 = Bet365
    r = requests.get(url, headers={"x-apisports-key": API_KEY}).json()["response"]
    if not r or "bookmakers" not in r[0] or len(r[0]["bookmakers"]) == 0:
        return [1.0, 1.0, 1.0]  # fallback si pas de cote
    odds_data = r[0]["bookmakers"][0]["bets"][0]["values"]
    return [odds_data[0]["odd"], odds_data[1]["odd"], odds_data[2]["odd"]]

def fetch_match_features(match_id):
    url = f"{API_BASE_URL}/fixtures?id={match_id}"
    r = requests.get(url, headers={"x-apisports-key": API_KEY}).json()["response"][0]
    home_id = r["teams"]["home"]["id"]
    away_id = r["teams"]["away"]["id"]
    season = r["league"]["season"]

    h = get_team_stats(home_id, season)
    a = get_team_stats(away_id, season)

    features = [
        h["gf"], h["ga"], h["cs"], h["fw"], h["fd"], h["fl"],
        a["gf"], a["ga"], a["cs"], a["fw"], a["fd"], a["fl"]
    ]
    return np.array([features]), get_match_odds(match_id)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        return {"error": "Modèle non chargé"}
    try:
        X, odds = fetch_match_features(req.match_id)
        probs = model.predict_proba(X)[0]  # proba pour 3 issues
        value_bets = []
        for i, p in enumerate(probs):
            if odds[i] * p > 1.05:  # threshold Value Bet
                value_bets.append({
                    "outcome": i, 
                    "probability": round(p, 3),
                    "odds": odds[i],
                    "value": round(odds[i]*p, 3)
                })
        return {
            "predictions": probs.tolist(),
            "odds": odds,
            "value_bets": value_bets
        }
    except Exception as e:
        return {"error": str(e)}

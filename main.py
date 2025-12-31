import os
import requests
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Charger le modèle sklearn XGBoost
MODEL_PATH = "model_sklearn.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print("Erreur lors du chargement du modèle:", e)
    model = None

# Charger la clé API-football depuis les secrets Render / Lovable
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY")
API_BASE_URL = "https://v3.football.api-sports.io"  # mettre le vrai endpoint

class PredictRequest(BaseModel):
    match_id: int  # ID du match ou identifiant API-football

def fetch_match_data(match_id: int):
    """
    Récupère les données nécessaires pour construire les features pour un match donné.
    """
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    url = f"{API_BASE_URL}/fixtures?id={match_id}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Erreur API-Football: {response.status_code}")
    data = response.json()
    
    # TODO: reconstruire exactement les features du modèle ici
    # Exemple :
    # features = [
    #     data["home_team"]["form"],
    #     data["away_team"]["form"],
    #     data["home_team"]["goals_avg"],
    #     data["away_team"]["goals_avg"],
    #     data["odds"]["bet365"]["home_win"],
    #     data["odds"]["bet365"]["draw"],
    #     data["odds"]["bet365"]["away_win"],
    #     ... # remplir toutes les features que ton modèle attend
    # ]
    
    features = []  # <- remplacer par le vrai calcul des 12+ features du modèle
    return np.array([features])  # retourne 2D array pour sklearn

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        return {"error": "Modèle non chargé"}
    try:
        X = fetch_match_data(req.match_id)
        preds = model.predict(X)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}

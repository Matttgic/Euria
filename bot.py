import requests
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from xgboost import XGBClassifier

# --- CONFIGURATION VIA SECRETS ---
API_KEY = os.getenv('API_KEY_FOOTBALL')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

HEADERS = {'x-apisports-key': API_KEY}
LEAGUES = [39, 140, 135, 78, 61]
HISTORY_FILE = "global_history.json"
CSV_FILE = "suivi_paris.csv"
MODEL_FILE = "model.json"

# --- CHARGEMENT DU MODÈLE IA ---
model = XGBClassifier()
if os.path.exists(MODEL_FILE):
    model.load_model(MODEL_FILE)
    print("✅ Modèle IA model.json chargé.")
else:
    print("⚠️ Attention: model.json introuvable. Utilisation du mode secours.")

# --- FONCTIONS TECHNIQUES ---

def get_match_detailed_stats(fixture_id):
    url = f"https://v3.football.api-sports.io/fixtures/statistics?fixture={fixture_id}"
    try:
        r = requests.get(url, headers=HEADERS).json().get("response", [])
        if not r: return None
        results = {}
        for team_data in r:
            t_id = str(team_data["team"]["id"])
            s_dict = {s["type"]: s["value"] for s in team_data["statistics"]}
            possession = str(s_dict.get("Ball Possession", "50%")).replace("%", "")
            results[t_id] = {
                "shots": s_dict.get("Shots on Goal", 0) or 0,
                "corners": s_dict.get("Corner Kicks", 0) or 0,
                "possession": int(possession)
            }
        return results
    except: return None

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={message}&parse_mode=Markdown"
    requests.get(url)

# --- ROUTINE PRINCIPALE ---

def main():
    # 1. Initialisation correcte des fichiers
    if not os.path.exists(CSV_FILE):
        pd.DataFrame(columns=['Match', 'Pari', 'Cote', 'Value', 'Date']).to_csv(CSV_FILE, index=False)

    if not API_KEY or not TELEGRAM_TOKEN:
        print("❌ Clés manquantes.")
        return

    history = load_history()

    # 2. Mise à jour avec les matchs d'hier
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    for league_id in LEAGUES:
        url = f"https://v3.football.api-sports.io/fixtures?league={league_id}&date={yesterday}&status=FT"
        matches = requests.get(url, headers=HEADERS).json().get("response", [])
        for m in matches:
            f_id = m['fixture']['id']
            h_id, a_id = str(m['teams']['home']['id']), str(m['teams']['away']['id'])
            stats = get_match_detailed_stats(f_id)
            if stats:
                if h_id not in history: history[h_id] = []
                if a_id not in history: history[a_id] = []
                history[h_id].append([m['goals']['home'], m['goals']['away'], stats[h_id]['shots'], stats[h_id]['corners'], stats[h_id]['possession']])
                history[a_id].append([m['goals']['away'], m['goals']['home'], stats[a_id]['shots'], stats[a_id]['corners'], stats[a_id]['possession']])
    
    save_history(history)

    # 3. Prédictions avec le modèle IA
    predictions = []
    for league_id in LEAGUES:
        url_next = f"https://v3.football.api-sports.io/fixtures?league={league_id}&next=10"
        fixtures = requests.get(url_next, headers=HEADERS).json().get("response", [])
        
        for f in fixtures:
            h_id, a_id = str(f['teams']['home']['id']), str(f['teams']['away']['id'])
            if h_id in history and a_id in history and len(history[h_id]) >= 5:
                h_avg = np.mean(history[h_id][-5:], axis=0)
                a_avg = np.mean(history[a_id][-5:], axis=0)
                
                # Formatage des features pour l'IA
                features = np.array([np.hstack([h_avg, a_avg])])
                
                # Prédiction
                try:
                    proba_array = model.predict_proba(features)[0]
                    
                    url_o = f"https://v3.football.api-sports.io/odds?fixture={f['fixture']['id']}"
                    odds_res = requests.get(url_o, headers=HEADERS).json().get("response", [])
                    
                    if odds_res:
                        book = odds_res[0]['bookmakers'][0]['bets'][0]['values']
                        current_odds = {v['value']: float(v['odd']) for v in book}
                        
                        # Mapping : model.classes_ (souvent 0:Draw, 1:Home, 2:Away)
                        outcomes = {1: 'Home', 0: 'Draw', 2: 'Away'}
                        for idx, label in enumerate(model.classes_):
                            name = outcomes[label]
                            if name in current_odds:
                                odd = current_odds[name]
                                proba = proba_array[idx]
                                
                                # SEUIL DE VALUE (1.10 = 10%)
                                if proba * odd > 1.10:
                                    val = round(proba * odd, 2)
                                    predictions.append(f"⚽️ *{f['teams']['home']['name']} vs {f['teams']['away']['name']}*\n🎯 {name} @ {odd} (IA: {round(proba*100)}%) | Value: {val}")
                                    
                                    # Log dans le CSV
                                    new_row = pd.DataFrame([[f['teams']['home']['name'] + " vs " + f['teams']['away']['name'], name, odd, val, datetime.now()]], columns=['Match', 'Pari', 'Cote', 'Value', 'Date'])
                                    new_row.to_csv(CSV_FILE, mode='a', header=False, index=False)
                except: continue

    # 4. Envoi
    if predictions:
        for i in range(0, len(predictions), 5):
            msg = "🚀 *NOUVEAUX VALUE BETS (IA)* 🚀\n\n" + "\n\n".join(predictions[i:i+5])
            send_telegram_alert(msg)
    
    print(f"✅ Terminé. {len(predictions)} alertes.")

if __name__ == "__main__":
    main()

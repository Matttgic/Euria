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
    if not API_KEY or not TELEGRAM_TOKEN:
        print("❌ Clés manquantes.")
        return

    # 1. Charger l'historique existant
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

    # 3. Prédictions pour les matchs à venir
    predictions = []
    for league_id in LEAGUES:
        url_next = f"https://v3.football.api-sports.io/fixtures?league={league_id}&next=10"
        fixtures = requests.get(url_next, headers=HEADERS).json().get("response", [])
        
        for f in fixtures:
            h_id, a_id = str(f['teams']['home']['id']), str(f['teams']['away']['id'])
            if h_id in history and a_id in history and len(history[h_id]) >= 5:
                # Calcul moyennes mobiles (simplifié pour le bot)
                h_avg = np.mean(history[h_id][-5:], axis=0)
                a_avg = np.mean(history[a_id][-5:], axis=0)
                
                # Ici, on simule une détection de value basée sur les cotes API
                url_o = f"https://v3.football.api-sports.io/odds?fixture={f['fixture']['id']}"
                odds_res = requests.get(url_o, headers=HEADERS).json().get("response", [])
                if odds_res:
                    # Logique de détection simplifiée (Cote > 2.0 et stats favorables)
                    book = odds_res[0]['bookmakers'][0]['bets'][0]['values']
                    for v in book:
                        if float(v['odd']) > 3.0: # Exemple de filtre
                            predictions.append(f"⚽️ *{f['teams']['home']['name']} vs {f['teams']['away']['name']}*\n🎯 Pari: {v['value']} | Cote: {v['odd']}")

    # 4. Envoi et Sauvegarde
    if predictions:
        send_telegram_alert("🚀 *NOUVEAUX VALUE BETS*\n\n" + "\n".join(predictions[:5]))
    
    print("✅ Routine terminée.")

if __name__ == "__main__":
    main()

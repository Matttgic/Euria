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
    print("✅ Modèle IA chargé.")
else:
    print("⚠️ model.json introuvable.")

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

# --- NOUVEAU : MISE À JOUR DES RÉSULTATS ET ROI ---

def update_results_and_get_stats():
    if not os.path.exists(CSV_FILE):
        return "Aucun historique de pari."
    
    df = pd.read_csv(CSV_FILE)
    if 'Result' not in df.columns:
        df['Result'] = ""

    # 1. Vérifier les matchs terminés (hier et avant)
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    url = f"https://v3.football.api-sports.io/fixtures?date={yesterday}&status=FT"
    res = requests.get(url, headers=HEADERS).json().get("response", [])
    
    results_map = {}
    for m in res:
        name = f"{m['teams']['home']['name']} vs {m['teams']['away']['name']}"
        h, a = m['goals']['home'], m['goals']['away']
        winner = "Home" if h > a else ("Away" if a > h else "Draw")
        results_map[name] = winner

    # 2. Marquer Win/Loss dans le CSV
    for index, row in df.iterrows():
        if (pd.isna(row['Result']) or row['Result'] == "") and row['Match'] in results_map:
            winner_reel = results_map[row['Match']]
            df.at[index, 'Result'] = "Win" if row['Pari'] == winner_reel else "Loss"
    
    df.to_csv(CSV_FILE, index=False)

    # 3. Calculer les statistiques (Mise de 10€ par pari)
    df_finit = df[df['Result'].isin(['Win', 'Loss'])]
    if df_finit.empty:
        return "En attente des premiers résultats terminés..."

    total = len(df_finit)
    gagnes = len(df_finit[df_finit['Result'] == 'Win'])
    winrate = (gagnes / total) * 100
    profit = 0
    for _, r in df_finit.iterrows():
        profit += (r['Cote'] - 1) * 10 if r['Result'] == 'Win' else -10
    
    return f"📊 *BILAN AUTOMATIQUE*\n✅ Winrate: {round(winrate, 1)}%\n💰 Profit: {round(profit, 2)}€\nTotal: {total} paris clôturés"

# --- ROUTINE PRINCIPALE ---

def main():
    # 1. Initialisation
    if not os.path.exists(CSV_FILE):
        pd.DataFrame(columns=['Match', 'Pari', 'Cote', 'Value', 'Date', 'Result']).to_csv(CSV_FILE, index=False)
        paris_deja_faits = []
    else:
        df_suivi = pd.read_csv(CSV_FILE)
        paris_deja_faits = df_suivi['Match'].tolist()

    # 2. Mise à jour des résultats et envoi du bilan
    bilan_msg = update_results_and_get_stats()
    send_telegram_alert(bilan_msg)

    # 3. Mise à jour de l'historique (Stats d'hier)
    history = load_history()
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    for league_id in LEAGUES:
        url = f"https://v3.football.api-sports.io/fixtures?league={league_id}&date={yesterday_str}&status=FT"
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

    # 4. Nouvelles Prédictions
    new_alerts = []
    for league_id in LEAGUES:
        url_next = f"https://v3.football.api-sports.io/fixtures?league={league_id}&next=10"
        fixtures = requests.get(url_next, headers=HEADERS).json().get("response", [])
        
        for f in fixtures:
            h_id, a_id = str(f['teams']['home']['id']), str(f['teams']['away']['id'])
            match_name = f"{f['teams']['home']['name']} vs {f['teams']['away']['name']}"

            if match_name in paris_deja_faits:
                continue

            if h_id in history and a_id in history and len(history[h_id]) >= 5:
                h_avg = np.mean(history[h_id][-5:], axis=0)
                a_avg = np.mean(history[a_id][-5:], axis=0)
                features = np.array([np.hstack([h_avg, a_avg])])
                
                try:
                    proba_array = model.predict_proba(features)[0]
                    url_o = f"https://v3.football.api-sports.io/odds?fixture={f['fixture']['id']}"
                    odds_res = requests.get(url_o, headers=HEADERS).json().get("response", [])
                    
                    if odds_res:
                        book = odds_res[0]['bookmakers'][0]['bets'][0]['values']
                        current_odds = {v['value']: float(v['odd']) for v in book}
                        outcomes = {1: 'Home', 0: 'Draw', 2: 'Away'}
                        
                        for idx, label in enumerate(model.classes_):
                            name = outcomes[label]
                            if name in current_odds:
                                odd = current_odds[name]
                                proba = proba_array[idx]
                                if proba * odd > 1.10:
                                    val = round(proba * odd, 2)
                                    new_alerts.append(f"⚽️ *{match_name}*\n🎯 {name} @ {odd} (IA: {round(proba*100)}%) | Value: {val}")
                                    
                                    # Sauvegarde avec colonne Result vide
                                    new_row = pd.DataFrame([[match_name, name, odd, val, datetime.now().strftime('%Y-%m-%d %H:%M'), ""]], 
                                                         columns=['Match', 'Pari', 'Cote', 'Value', 'Date', 'Result'])
                                    new_row.to_csv(CSV_FILE, mode='a', header=False, index=False)
                                    paris_deja_faits.append(match_name)
                except: continue

    # 5. Envoi Telegram
    if new_alerts:
        for i in range(0, len(new_alerts), 5):
            msg = "🚀 *NOUVELLES OPPORTUNITÉS IA* 🚀\n\n" + "\n\n".join(new_alerts[i:i+5])
            send_telegram_alert(msg)
    
    print("✅ Terminé.")

if __name__ == "__main__":
    main()

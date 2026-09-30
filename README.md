# Euria

Bot de détection de *value bets* sur les 5 grands championnats européens (Premier League, LaLiga,
Serie A, Bundesliga, Ligue 1), avec une API FastAPI. Chaque jour, il :

1. règle les paris en attente avec les scores réels et envoie le bilan sur Telegram ;
2. calcule, pour chaque match des 7 prochains jours, les probabilités du modèle et les compare aux cotes ;
3. envoie une alerte quand probabilité × cote dépasse le seuil (1,10 par défaut).

> Outil d'analyse, pas une garantie de gain. Bilan réel au 30/09/2026 (240 paris, ancien modèle) : ROI −5,2 %.

## Architecture

```
bot.py / main.py            points d'entrée (GitHub Actions / uvicorn main:app)
euria/
  config.py                 réglages, championnats, durées de cache (aucune clé en dur)
  http.py                   appels HTTP : délai max 8 s, User-Agent, erreurs lisibles
  cache.py                  cache disque + dernière valeur connue
  fallback.py               principale -> secours -> dernière valeur connue (ne lève jamais)
  schema.py                 format commun renvoyé par toutes les sources
  sources/                  un fichier par API externe
  services.py               matchs, cotes, stades, météo prêts à l'emploi
  teams.py                  rapprochement des noms d'équipes entre sources
  stats.py                  classement et 12 variables du modèle, calculés à partir des scores
  predictor.py              modèle unique models/model_sklearn.pkl + calcul de la value
  analysis.py               analyse d'un match (partagée par le bot et l'API)
  betting.py / telegram.py  suivi des paris (data/suivi_paris.csv) et alertes
  bot.py / api.py           routine quotidienne et API
models/model_sklearn.pkl    modèle XGBoost (12 variables) ; models/archive/ = anciens modèles
data/                       suivi_paris.csv, predictions.csv (commités) ; cache/ (jamais commité)
archive/global_history.json historique utilisé par l’ancien modèle (archivé)
tests/                      tests hors ligne ; tests/live/ = un test réel par API
```

## Sources de données

| Donnée | Principale | Secours | Clé | Cache |
|---|---|---|---|---|
| Matchs et scores | football-data.org | football-data.co.uk (CSV), OpenLigaDB (Bundesliga) | `FOOTBALL_DATA_TOKEN` (gratuite) | 1 h |
| Cotes 1N2 | Parlay API (PMU, Unibet, bet365, Pinnacle) | API-Football (bet365) | `PARLAY_API_KEY`, `API_FOOTBALL_KEY` (gratuites) | 4 h |
| Météo au stade | MET Norway | Open-Meteo (non commercial) | aucune | 3 h |
| Stades | Wikidata | TheSportsDB + Nominatim | aucune | 30 j |

Le classement et les variables du modèle sont **calculés par Euria à partir des scores** : quelle que
soit la source des matchs, le calcul est identique. Si toutes les sources d'une donnée tombent, la
dernière valeur connue est utilisée, avec son heure et un message ; jamais de plantage.

Mentions obligatoires (ajoutées automatiquement là où la donnée est renvoyée : champ `attribution` de l’API, pied des alertes Telegram) :
« Football data provided by the Football-Data.org API », « Données météo : MET Norway (CC BY 4.0) »,
« Données météo : Open-Meteo.com (CC BY 4.0) », « © OpenStreetMap contributors ».

## Installation

```bash
pip install -r requirements-dev.txt
cp .env.example .env        # puis remplir les clés (le fichier .env n'est jamais commité)
python bot.py               # routine du bot
uvicorn main:app --reload   # API sur http://localhost:8000/docs
```

## API

Toutes les réponses : `{"data": ..., "meta": {"source", "updated_at", "stale", "message", "attribution", "errors"}}`.

| Route | Rôle |
|---|---|
| `GET /leagues` | championnats disponibles (`PL`, `PD`, `SA`, `BL1`, `FL1`) |
| `GET /matches/{league}?status=all\|upcoming\|finished` | calendrier et scores |
| `GET /standings/{league}` | classement calculé |
| `GET /odds/{league}` | cotes 1N2 par bookmaker |
| `GET /weather?home=Arsenal&kickoff=2026-10-10T11:30:00Z` | météo au stade de l'équipe à domicile |
| `POST /predict` `{"league": "PL", "home": "Arsenal", "away": "Leeds United"}` | probabilités, variables, cotes, value, météo, classement |

## Tests

```bash
pytest              # tests hors ligne (réponses réelles enregistrées, aucun réseau)
pytest -m live -rs  # un test réel par API : vérifie que les champs utilisés existent toujours
```

Les tests réels des sources à clé sont sautés si la clé est absente. Le workflow `Tests` lance les
tests hors ligne à chaque push et les tests réels chaque lundi.

## Modèle

`models/model_sklearn.pkl` attend 12 variables : pour chaque équipe, buts marqués et encaissés par
match sur la saison, nombre de matchs sans encaisser, victoires/nuls/défaites sur les 5 derniers.
Ordre des classes : **0 = Nul, 1 = Domicile, 2 = Extérieur** (déduit du modèle lui-même ; l'ancien
`main.py` utilisait un ordre faux). La météo et le classement sont enregistrés dans
`data/predictions.csv` pour un futur réentraînement, mais ne sont pas des entrées du modèle actuel.

"""Réglages centralisés. Aucune clé n'est écrite ici : tout vient de l'environnement
(fichier .env en local, secrets GitHub Actions ou variables de l'hébergeur en production)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

try:  # .env est optionnel : en production les variables viennent de l'hébergeur
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


# ---------------------------------------------------------------------------
# Clés et jetons (tous optionnels : une source sans clé est simplement sautée)
# ---------------------------------------------------------------------------
FOOTBALL_DATA_TOKEN = _env("FOOTBALL_DATA_TOKEN")
PARLAY_API_KEY = _env("PARLAY_API_KEY")
# Facultative : sans clé, 100 requêtes/jour par IP, et Noozra refuse les serveurs GitHub Actions (HTTP 403,
# constaté le 30/09/2026). Avec une clé gratuite : 5 000/jour, comptées par clé.
NOOZRA_API_KEY = _env("NOOZRA_API_KEY")
TELEGRAM_TOKEN = _env("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = _env("TELEGRAM_CHAT_ID")

# ---------------------------------------------------------------------------
# Réseau
# ---------------------------------------------------------------------------
HTTP_TIMEOUT_S = 8
# MET Norway, Nominatim et Wikidata exigent un User-Agent qui identifie l'application
# et permet de contacter son auteur.
CONTACT = _env("EURIA_CONTACT", "https://github.com/Matttgic/Euria")
USER_AGENT = f"Euria/2.0 (+{CONTACT})"

# ---------------------------------------------------------------------------
# Stockage
# ---------------------------------------------------------------------------
DATA_DIR = Path(_env("EURIA_DATA_DIR", str(ROOT_DIR / "data")))
CACHE_DIR = DATA_DIR / "cache"
BETS_FILE = DATA_DIR / "suivi_paris.csv"
PREDICTIONS_FILE = DATA_DIR / "predictions.csv"
MODEL_FILE = ROOT_DIR / "models" / "model_sklearn.pkl"

# ---------------------------------------------------------------------------
# Durées de cache (secondes)
# ---------------------------------------------------------------------------
MINUTE, HOUR, DAY = 60, 3600, 86400
TTL = {
    # Liste des matchs de la saison (calendrier + scores). Relue au plus toutes les heures :
    # le bot a besoin des scores de la veille à chaque exécution, et football-data.org
    # autorise 10 appels/minute (5 championnats = 5 appels).
    "matches": 1 * HOUR,
    # Cotes : 4 h. Budget Parlay gratuit = 1 000 crédits/mois ; 5 championnats x 6 rafraîchissements
    # par jour x 30 jours = 900 crédits dans le pire cas (API interrogée en continu).
    "odds": 4 * HOUR,
    # Météo : les modèles de prévision sont recalculés toutes les quelques heures.
    "weather": 3 * HOUR,
    # Stades : ils ne bougent pas.
    "venues": 30 * DAY,
    # Points attendus (Football Charts) : ne changent qu'après une journée de championnat.
    "xpoints": 12 * HOUR,
    # Deuxième avis (Bet Better) : leur modèle est recalculé plusieurs fois par jour.
    "second_opinion": 3 * HOUR,
    # Actus (Noozra) : 100 requêtes/jour sans clé, donc on ne relit pas la même équipe avant 6 h.
    "news": 6 * HOUR,
}
NEWS_MAX_AGE_DAYS = 7  # une actu blessure plus ancienne n'est plus pertinente
# Conditions Parlay : pas de conservation des cotes détaillées plus de 90 jours.
ODDS_MAX_RETENTION_S = 90 * DAY

# ---------------------------------------------------------------------------
# Modèle et paris
# ---------------------------------------------------------------------------
# Ordre des classes du modèle, déduit des données du modèle lui-même (base_score :
# classe 1 = 45-48 % = victoires à domicile) et vérifié sur des cas extrêmes.
# main.py (avant la refonte) utilisait l'ordre Domicile/Nul/Extérieur : c'était faux.
CLASS_LABELS = {0: "Draw", 1: "Home", 2: "Away"}
VALUE_THRESHOLD = float(_env("VALUE_THRESHOLD", "1.10"))
MIN_MATCHES_PLAYED = int(_env("MIN_MATCHES_PLAYED", "5"))
DAYS_AHEAD = int(_env("DAYS_AHEAD", "7"))
STAKE_EUR = float(_env("STAKE_EUR", "10"))
# Bookmakers pour la value, par ordre de préférence (clés Parlay). PMU est agréé ANJ en France.
ODDS_BOOKMAKERS = [b.strip() for b in _env("ODDS_BOOKMAKERS", "pmu,unibet,bet365,pinnacle").split(",") if b.strip()]


# ---------------------------------------------------------------------------
# Championnats : identifiants de chaque source, tous vérifiés pendant l'intégration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class League:
    code: str  # code football-data.org, sert d'identifiant interne
    name: str
    couk: str  # fichier CSV football-data.co.uk
    parlay: str  # sport_key Parlay
    wikidata: str  # entité Wikidata du championnat
    openligadb: str | None = None  # raccourci OpenLigaDB (Bundesliga seulement)
    football_charts: str = ""  # identifiant Football Charts (points attendus)
    betbetter: str = ""  # identifiant Bet Better (deuxième avis)


LEAGUES: dict[str, League] = {
    "PL": League("PL", "Premier League", "E0", "soccer_epl", "Q9448", football_charts="premier", betbetter="epl"),
    "PD": League("PD", "LaLiga", "SP1", "soccer_spain_la_liga", "Q324867", football_charts="spain1", betbetter="la-liga"),
    "SA": League("SA", "Serie A", "I1", "soccer_italy_serie_a", "Q15804", football_charts="italy1", betbetter="serie-a"),
    "BL1": League("BL1", "Bundesliga", "D1", "soccer_germany_bundesliga", "Q82595", openligadb="bl1",
                  football_charts="germany1", betbetter="bundesliga"),
    "FL1": League("FL1", "Ligue 1", "F1", "soccer_france_ligue_one", "Q13394", football_charts="france1", betbetter="ligue-1"),
}


def get_league(code: str) -> League:
    try:
        return LEAGUES[code.upper()]
    except KeyError as exc:
        raise KeyError(f"Championnat inconnu : {code}. Choix : {', '.join(LEAGUES)}") from exc


def season_for(day: date | datetime | None = None) -> int:
    """Année de début de la saison (2026 pour 2026/27). Une saison commence en juillet."""
    day = day or datetime.now(timezone.utc)
    return day.year if day.month >= 7 else day.year - 1

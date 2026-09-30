"""Rapprochement des noms d'équipes entre sources.

Chaque source écrit les noms à sa façon (« Man United », « Manchester United F.C. »,
« Paris SG », « Paris Saint-Germain FC »). On normalise, on applique une table d'alias
tirée des noms réellement renvoyés par les sources, puis on cherche la meilleure
correspondance. En cas de doute, on ne devine pas : la fonction renvoie None."""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Iterable

# Mots qui ne distinguent pas un club (formes juridiques, « club », articles).
_STOPWORDS = {
    "fc", "afc", "cf", "ac", "as", "sc", "ssc", "ss", "us", "ogc", "rc", "rcd", "ud", "cd", "sd", "bc",
    "cfc", "acf", "sv", "tsg", "fsv", "vfb", "vfl", "osc", "sco", "aj", "es", "club", "de", "del", "la",
    "calcio", "football", "futbol", "balompie", "and", "hove", "albion", "hotspur", "sad",
}

# Villes partagées par plusieurs clubs : « Paris » seul ne doit jamais être rapproché
# de « Paris Saint-Germain » (ce peut être le Paris FC).
_AMBIGUOUS_SHORT = {"paris", "milan", "madrid", "manchester", "barcelona", "sevilla", "bilbao"}

# Variantes observées dans les sources -> forme longue. Clés et valeurs déjà normalisées.
# Les noms courts de football-data.org (« Barça », « Atleti », « HSV »…) ont été relevés le
# 30/09/2026 par tests/live/test_live_team_names.py.
_ALIASES = {
    "barca": "barcelona",
    "atleti": "atletico madrid",
    "real oviedo": "oviedo",
    "wolves": "wolverhampton",
    "wolverhampton wanderers": "wolverhampton",
    "hsv": "hamburg",
    "bremen": "werder bremen",
    "frankfurt": "eintracht frankfurt",
    "olympique lyon": "lyon",
    "man city": "manchester city",
    "man united": "manchester united",
    "man utd": "manchester united",
    "nottm forest": "nottingham forest",
    "ein frankfurt": "eintracht frankfurt",
    "koln": "koln",
    "mgladbach": "borussia monchengladbach",
    "monchengladbach": "borussia monchengladbach",
    "bayern munchen": "bayern munich",
    "bayern": "bayern munich",
    "dortmund": "borussia dortmund",
    "leverkusen": "bayer 04 leverkusen",
    "bayer leverkusen": "bayer 04 leverkusen",
    "ath bilbao": "athletic",
    "athletic bilbao": "athletic",
    "ath madrid": "atletico madrid",
    "espanol": "espanyol",
    "espanyol barcelona": "espanyol",
    "sociedad": "real sociedad",
    "vallecano": "rayo vallecano",
    "betis": "real betis",
    "celta": "celta vigo",
    "coruna": "deportivo coruna",
    "deportivo a coruna": "deportivo coruna",
    "deportivo alaves": "alaves",
    "atletico osasuna": "osasuna",
    "santander": "racing santander",
    "paris sg": "paris saint germain",
    "psg": "paris saint germain",
    "inter": "inter milan",
    "internazionale milano": "inter milan",
    "internazionale": "inter milan",
    "milan": "milan",
    "real madrid futbol": "real madrid",
    "hull": "hull",
    "stade rennais": "rennes",
    "stade brestois 29": "brest",
    "stade brestois": "brest",
    "olympique lyonnais": "lyon",
    "olympique marseille": "marseille",
    "olympique de marseille": "marseille",
    "lille": "lille",
    "strasbourg alsace": "strasbourg",
    "monaco": "monaco",
    "nice": "nice",
    "angers": "angers",
    "havre": "le havre",
    "troyes": "troyes",
    "auxerre": "auxerre",
    "lens": "lens",
    "mainz 05": "mainz",
    "paderborn 07": "paderborn",
    "1899 hoffenheim": "hoffenheim",
    "werder bremen": "werder bremen",
    "hamburger": "hamburg",
    "schalke 04": "schalke",
    "union berlin": "union berlin",
    "fiorentina": "fiorentina",
    "bologna 1909": "bologna",
    "parma 1913": "parma",
    "como 1907": "como",
}


def normalize(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = text.replace("&", " and ").replace("'", "").replace("’", "").replace(".", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [t for t in text.split() if t not in _STOPWORDS and not (t.isdigit() and len(t) == 1)]
    key = " ".join(tokens) or text.strip()
    return _ALIASES.get(key, key)


def similarity(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return 1.0
    ta, tb = set(na.split()), set(nb.split())
    short, long_ = (na, nb) if len(ta) <= len(tb) else (nb, na)
    if ta and tb and (ta <= tb or tb <= ta) and short not in _AMBIGUOUS_SHORT and long_.startswith(short):
        # Un nom qui commence l'autre (« Leeds » / « Leeds United ») : fort mais pas certain.
        return 0.9
    return difflib.SequenceMatcher(None, na, nb).ratio()


def best_match(name: str, candidates: Iterable[str], threshold: float = 0.85) -> str | None:
    """Candidat le plus proche, ou None si aucun n'est assez proche ou si deux sont ex æquo."""
    scored = sorted(((similarity(name, c), c) for c in set(candidates)), reverse=True)
    if not scored or scored[0][0] < threshold:
        return None
    if len(scored) > 1 and scored[1][0] == scored[0][0] and normalize(scored[1][1]) != normalize(scored[0][1]):
        return None  # ambigu : on ne devine pas
    return scored[0][1]


def same_team(a: str, b: str) -> bool:
    return similarity(a, b) >= 0.85

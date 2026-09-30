"""Wikidata (requête SPARQL publique) : stade et coordonnées des clubs — source PRINCIPALE des stades.

Sans clé (l'annuaire indique OAuth : c'est pour l'écriture, la lecture SPARQL est publique).
Données CC0. User-Agent identifiant exigé. Une seule requête couvre les 5 championnats.
Adresse : GET https://query.wikidata.org/sparql?query=...  (Accept: application/sparql-results+json)
Champs : results.bindings[].clubLabel / venueLabel / coord ("Point(lon lat)").
Limite connue : quelques clubs n'y sont pas reliés à leur championnat (Chelsea, Barcelona, Augsburg
au 30/09/2026) ; le secours TheSportsDB + Nominatim prend alors le relais."""

from __future__ import annotations

import re

from .. import schema
from ..config import LEAGUES
from ..http import get_json, require

NAME = "Wikidata"
ATTRIBUTION = None  # CC0 : aucune mention exigée
URL = "https://query.wikidata.org/sparql"
_POINT = re.compile(r"Point\(([-0-9.eE]+) ([-0-9.eE]+)\)")


def build_query() -> str:
    leagues = " ".join(f"wd:{league.wikidata}" for league in LEAGUES.values())
    return (
        "SELECT ?clubLabel ?venueLabel ?coord WHERE { "
        f"VALUES ?league {{ {leagues} }} "
        "?club wdt:P118 ?league; wdt:P31 wd:Q476028; wdt:P115 ?venue. ?venue wdt:P625 ?coord. "
        'SERVICE wikibase:label { bd:serviceParam wikibase:language "en,fr". } }'
    )


def parse_venues(payload: dict) -> list[dict]:
    bindings = ((payload or {}).get("results") or {}).get("bindings")
    require(NAME, isinstance(bindings, list), "results.bindings absent")
    out, seen = [], set()
    for b in bindings:
        match = _POINT.match(b.get("coord", {}).get("value", ""))
        club = b.get("clubLabel", {}).get("value")
        if not match or not club or club in seen:
            continue
        seen.add(club)  # un club peut avoir plusieurs stades listés : on garde le premier
        lon, lat = float(match.group(1)), float(match.group(2))
        out.append(schema.venue(team=club, stadium=b.get("venueLabel", {}).get("value"), latitude=lat, longitude=lon))
    return out


def fetch_all_venues() -> list[dict]:
    payload = get_json(NAME, URL, params={"query": build_query()}, headers={"Accept": "application/sparql-results+json"})
    return parse_venues(payload)

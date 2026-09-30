"""Bet Better : pronostic 1N2 d'un modèle extérieur, pour un deuxième avis — sans clé.

Licence CC BY 4.0 (usage commercial autorisé), mention « Bet Better — https://betbetter.world ».
Réponses cachables 15 minutes ; modèle recalculé plusieurs fois par jour.
Adresse : GET https://betbetter.world/soccer/{epl|la-liga|serie-a|bundesliga|ligue-1}/picks?format=json
Champs utilisés : picks[].game (« Extérieur @ Domicile », vérifié sur le calendrier OpenLigaDB),
gameTimeUtc, market (« Head to Head »), selection, winProbabilityPct, fairOdds, confidence.
Une seule issue par match : celle que leur modèle choisit."""

from __future__ import annotations

from .. import schema
from ..config import League
from ..http import SourceError, get_json, require

NAME = "Bet Better"
ATTRIBUTION = "Bet Better — https://betbetter.world"
BASE_URL = "https://betbetter.world/soccer"


def parse_picks(payload: dict, league: League) -> list[dict]:
    require(NAME, isinstance(payload, dict) and isinstance(payload.get("picks"), list), "clé 'picks' absente")
    if payload.get("error"):  # en cas de problème, Bet Better renvoie un document valide avec un champ error
        raise SourceError(NAME, f"erreur signalée : {payload['error']}")
    out = []
    for pick in payload["picks"]:
        if pick.get("market") != "Head to Head" or " @ " not in pick.get("game", ""):
            continue
        away, home = pick["game"].split(" @ ", 1)
        outcome = {home: "Home", away: "Away", "Draw": "Draw"}.get(pick.get("selection"))
        if outcome is None or pick.get("winProbabilityPct") is None:
            continue
        out.append(
            schema.opinion(
                league=league.code,
                utc_date=schema.iso_utc(schema.parse_utc(pick["gameTimeUtc"])),
                home=home,
                away=away,
                outcome=outcome,
                probability=pick["winProbabilityPct"] / 100,
                fair_odds=pick.get("fairOdds"),
                confidence=pick.get("confidence"),
            )
        )
    return out


def fetch_picks(league: League) -> list[dict]:
    return parse_picks(get_json(NAME, f"{BASE_URL}/{league.betbetter}/picks", params={"format": "json"}), league)

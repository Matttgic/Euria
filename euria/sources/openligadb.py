"""OpenLigaDB : calendrier et scores de la Bundesliga — SECOURS (Bundesliga uniquement, sans clé).

Adresse : GET https://api.openligadb.de/getmatchdata/bl1/{année}
Champs utilisés : matchDateTimeUTC, team1.teamName, team2.teamName, matchIsFinished,
matchResults[resultTypeID == 2] (score final : pointsTeam1, pointsTeam2)."""

from __future__ import annotations

from .. import schema
from ..config import League
from ..http import SourceError, get_json, require

NAME = "OpenLigaDB"
ATTRIBUTION = "Bundesliga : OpenLigaDB"
BASE_URL = "https://api.openligadb.de"
FINAL_RESULT_TYPE = 2  # 1 = mi-temps, 2 = « Endergebnis » (score final)


def parse_matches(payload: list, league: League, season: int) -> list[dict]:
    require(NAME, isinstance(payload, list), "liste de matchs attendue")
    out = []
    for m in payload:
        final = next((r for r in m.get("matchResults") or [] if r.get("resultTypeID") == FINAL_RESULT_TYPE), None)
        finished = bool(m.get("matchIsFinished")) and final is not None
        out.append(
            schema.match(
                league=league.code,
                season=season,
                utc_date=m["matchDateTimeUTC"],
                home=m["team1"]["teamName"],
                away=m["team2"]["teamName"],
                status="FINISHED" if finished else "SCHEDULED",
                home_goals=final["pointsTeam1"] if finished else None,
                away_goals=final["pointsTeam2"] if finished else None,
            )
        )
    return out


def fetch_matches(league: League, season: int) -> list[dict]:
    if not league.openligadb:
        raise SourceError(NAME, f"{league.name} non couvert")
    payload = get_json(NAME, f"{BASE_URL}/getmatchdata/{league.openligadb}/{season}")
    return parse_matches(payload, league, season)

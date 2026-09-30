"""football-data.org (v4) : calendrier et scores des 5 championnats — source PRINCIPALE des matchs.

Clé gratuite (FOOTBALL_DATA_TOKEN), 10 appels/minute, les 5 championnats sont dans l'offre gratuite.
Adresse : GET https://api.football-data.org/v4/competitions/{code}/matches?season={année}
Format documenté : {"matches": [{"utcDate", "status", "venue", "homeTeam": {"name", "shortName"},
"awayTeam": {...}, "score": {"fullTime": {"home", "away"}}}]}"""

from __future__ import annotations

from .. import config, schema
from ..config import League
from ..http import SourceError, get_json, require

NAME = "football-data.org"
ATTRIBUTION = "Football data provided by the Football-Data.org API"
BASE_URL = "https://api.football-data.org/v4"

_STATUS = {
    "SCHEDULED": "SCHEDULED", "TIMED": "SCHEDULED",
    "IN_PLAY": "LIVE", "PAUSED": "LIVE", "LIVE": "LIVE",
    "FINISHED": "FINISHED", "AWARDED": "FINISHED",
    "POSTPONED": "POSTPONED", "SUSPENDED": "POSTPONED",
    "CANCELLED": "CANCELLED",
}


def _team_name(team: dict) -> str:
    return team.get("shortName") or team.get("name") or ""


def parse_matches(payload: dict, league: League, season: int) -> list[dict]:
    require(NAME, isinstance(payload, dict) and isinstance(payload.get("matches"), list), "clé 'matches' absente")
    out = []
    for m in payload["matches"]:
        full_time = (m.get("score") or {}).get("fullTime") or {}
        status = _STATUS.get(m.get("status", ""), "SCHEDULED")
        out.append(
            schema.match(
                league=league.code,
                season=season,
                utc_date=m["utcDate"],
                home=_team_name(m["homeTeam"]),
                away=_team_name(m["awayTeam"]),
                status=status,
                home_goals=full_time.get("home") if status == "FINISHED" else None,
                away_goals=full_time.get("away") if status == "FINISHED" else None,
                venue=m.get("venue"),
            )
        )
    return out


def fetch_matches(league: League, season: int) -> list[dict]:
    if not config.FOOTBALL_DATA_TOKEN:
        raise SourceError(NAME, "clé FOOTBALL_DATA_TOKEN absente")
    payload = get_json(
        NAME,
        f"{BASE_URL}/competitions/{league.code}/matches",
        params={"season": season},
        headers={"X-Auth-Token": config.FOOTBALL_DATA_TOKEN},
    )
    return parse_matches(payload, league, season)

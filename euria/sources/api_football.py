"""API-Football (v3) : cotes bet365 — SECOURS des cotes.

Clé gratuite (API_FOOTBALL_KEY) : 100 appels/jour. Coût : 1 appel pour la liste des matchs
+ 1 appel par match. On se limite donc aux matchs des DAYS_AHEAD prochains jours.
Adresses :
  GET https://v3.football.api-sports.io/fixtures?league={id}&season={année}&from={AAAA-MM-JJ}&to={AAAA-MM-JJ}
  GET https://v3.football.api-sports.io/odds?fixture={id}&bookmaker=8&bet=1   (8 = bet365, 1 = 1N2)
L'ancien code utilisait fixtures?next=10, refusé par l'offre gratuite (« Free plans do not have
access to the Next parameter », constaté le 30/09/2026) : d'où la plage de dates.
Le test tests/live/test_live_api_football.py appelle fetch_fixtures() et vérifie les champs utilisés."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .. import config, schema
from ..config import season_for
from ..config import League
from ..http import SourceError, get_json, require

NAME = "API-Football"
ATTRIBUTION = "Cotes : API-Football"
BASE_URL = "https://v3.football.api-sports.io"
BET365, MATCH_WINNER = 8, 1


def _headers() -> dict:
    if not config.API_FOOTBALL_KEY:
        raise SourceError(NAME, "clé API_FOOTBALL_KEY absente")
    return {"x-apisports-key": config.API_FOOTBALL_KEY}


def _response(payload) -> list:
    require(NAME, isinstance(payload, dict) and isinstance(payload.get("response"), list), "clé 'response' absente")
    if payload.get("errors"):  # API-Football renvoie HTTP 200 avec un champ errors en cas de refus
        raise SourceError(NAME, f"refus de l'API : {payload['errors']}")
    return payload["response"]


def parse_odds(payload, league: League, fixture: dict) -> dict | None:
    rows = _response(payload)
    if not rows:
        return None
    book = rows[0]["bookmakers"][0]
    values = {v["value"]: float(v["odd"]) for v in book["bets"][0]["values"]}
    if not {"Home", "Draw", "Away"} <= values.keys():
        return None
    return schema.odds(
        league=league.code,
        utc_date=fixture["utc_date"],
        home=fixture["home"],
        away=fixture["away"],
        bookmaker="bet365",
        home_odds=values["Home"],
        draw_odds=values["Draw"],
        away_odds=values["Away"],
        updated_at=rows[0].get("update"),
    )


def fetch_fixtures(league: League) -> list:
    """Matchs des DAYS_AHEAD prochains jours (réponse brute d'API-Football)."""
    today = datetime.now(timezone.utc).date()
    payload = get_json(
        NAME,
        f"{BASE_URL}/fixtures",
        params={
            "league": league.api_football,
            "season": season_for(today),
            "from": today.isoformat(),
            "to": (today + timedelta(days=config.DAYS_AHEAD)).isoformat(),
        },
        headers=_headers(),
    )
    return _response(payload)


def fetch_odds(league: League) -> list[dict]:
    headers = _headers()
    horizon = datetime.now(timezone.utc) + timedelta(days=config.DAYS_AHEAD)
    out = []
    for f in fetch_fixtures(league):
        kickoff = f["fixture"].get("date")
        if not kickoff or schema.parse_utc(kickoff) > horizon:
            continue
        fixture = {
            "utc_date": schema.iso_utc(schema.parse_utc(kickoff)),
            "home": f["teams"]["home"]["name"],
            "away": f["teams"]["away"]["name"],
        }
        odds_payload = get_json(
            NAME, f"{BASE_URL}/odds", params={"fixture": f["fixture"]["id"], "bookmaker": BET365, "bet": MATCH_WINNER}, headers=headers
        )
        quote = parse_odds(odds_payload, league, fixture)
        if quote:
            out.append(quote)
    return out

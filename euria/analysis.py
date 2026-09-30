"""Analyse d'un match, partagée par le bot et l'API : variables, probabilités, cotes, value, météo."""

from __future__ import annotations

from datetime import datetime

from . import predictor, services, stats
from .fallback import Result
from .schema import parse_utc
from .teams import same_team


def group_events(quotes: list[dict]) -> list[dict]:
    """Regroupe les cotes par match (une ligne par bookmaker -> un événement avec toutes ses cotes)."""
    events: dict[tuple, dict] = {}
    for q in quotes:
        key = (q["home"], q["away"], q["utc_date"])
        events.setdefault(key, {"home": q["home"], "away": q["away"], "utc_date": q["utc_date"], "quotes": []})["quotes"].append(q)
    return sorted(events.values(), key=lambda e: e["utc_date"])


def find_event(quotes: list[dict], home: str, away: str) -> dict | None:
    for event in group_events(quotes):
        if same_team(event["home"], home) and same_team(event["away"], away):
            return event
    return None


def analyze(
    league: str,
    home: str,
    away: str,
    kickoff: datetime,
    matches: Result,
    quotes: list[dict] | None = None,
    with_weather: bool = True,
) -> dict:
    messages: list[str] = []
    result: dict = {
        "league": league, "home": home, "away": away, "kickoff": kickoff.isoformat(),
        "features": None, "probabilities": None, "standings": None,
        "odds": {"selected": None, "all": quotes or []}, "value_bets": [], "weather": None,
        "messages": messages, "sources": {"matches": matches.meta()},
    }

    if not matches.ok:
        messages.append(matches.message or "Matchs indisponibles.")
    else:
        feats = stats.features(matches.data, home, away, before=kickoff)
        if feats is None:
            messages.append("Variables indisponibles : équipe introuvable ou moins de matchs joués que le minimum requis.")
        else:
            result["features"] = dict(zip(stats.FEATURE_NAMES, [round(x, 3) for x in feats["vector"]]))
            result["probabilities"] = predictor.predict(feats["vector"])
            table = {row["team"]: row for row in stats.standings(matches.data, before=kickoff)}
            result["standings"] = {"home": table.get(feats["home_team"]), "away": table.get(feats["away_team"])}

    if quotes:
        quote = predictor.choose_quote(quotes)
        result["odds"]["selected"] = quote
        if quote and result["probabilities"]:
            result["value_bets"] = predictor.value_bets(result["probabilities"], quote)
    else:
        messages.append("Cotes indisponibles pour ce match.")

    if with_weather:
        weather = services.weather(home, kickoff)
        result["weather"] = weather.data
        result["sources"]["weather"] = weather.meta()
        if not weather.ok:
            messages.append(weather.message or "Météo indisponible.")
    return result


def kickoff_of(event: dict) -> datetime:
    return parse_utc(event["utc_date"])

"""Format unique renvoyé par toutes les sources, quelle que soit l'API d'origine.

Les données circulent en dictionnaires simples (sérialisables en JSON pour le cache et l'API) ;
ces fonctions garantissent que les clés sont toujours les mêmes."""

from __future__ import annotations

from datetime import datetime, timezone

STATUSES = {"SCHEDULED", "LIVE", "FINISHED", "POSTPONED", "CANCELLED"}


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def match(
    *, league: str, season: int, utc_date: str, home: str, away: str, status: str,
    home_goals: int | None = None, away_goals: int | None = None, venue: str | None = None,
) -> dict:
    if status not in STATUSES:
        raise ValueError(f"statut inconnu : {status}")
    return {
        "league": league, "season": season, "utc_date": utc_date, "home": home, "away": away,
        "status": status, "home_goals": home_goals, "away_goals": away_goals, "venue": venue,
    }


def odds(
    *, league: str, utc_date: str, home: str, away: str, bookmaker: str,
    home_odds: float, draw_odds: float, away_odds: float, updated_at: str | None = None,
) -> dict:
    return {
        "league": league, "utc_date": utc_date, "home": home, "away": away, "bookmaker": bookmaker,
        "home_odds": round(float(home_odds), 3), "draw_odds": round(float(draw_odds), 3),
        "away_odds": round(float(away_odds), 3), "updated_at": updated_at,
    }


def weather(
    *, latitude: float, longitude: float, time: str, temperature_c: float | None,
    precipitation_mm: float | None, wind_speed_ms: float | None,
) -> dict:
    return {
        "latitude": latitude, "longitude": longitude, "time": time, "temperature_c": temperature_c,
        "precipitation_mm": precipitation_mm, "wind_speed_ms": wind_speed_ms,
    }


def venue(*, team: str, stadium: str | None, latitude: float, longitude: float) -> dict:
    return {"team": team, "stadium": stadium, "latitude": round(float(latitude), 5), "longitude": round(float(longitude), 5)}


def xpoints(*, team: str, played: int, points: int, expected_points: float, form: str | None = None) -> dict:
    """Points réels contre points attendus. luck > 0 : l'équipe a pris plus de points que son jeu ne le laissait prévoir."""
    return {
        "team": team, "played": played, "points": points, "expected_points": round(float(expected_points), 2),
        "luck": round(points - float(expected_points), 2), "form": form,
    }


def opinion(
    *, league: str, utc_date: str, home: str, away: str, outcome: str, probability: float,
    fair_odds: float | None = None, confidence: str | None = None,
) -> dict:
    """Pronostic 1N2 d'un modèle extérieur (outcome : Home, Draw ou Away ; probability entre 0 et 1)."""
    if outcome not in ("Home", "Draw", "Away"):
        raise ValueError(f"issue inconnue : {outcome}")
    return {
        "league": league, "utc_date": utc_date, "home": home, "away": away, "outcome": outcome,
        "probability": round(float(probability), 4), "fair_odds": fair_odds, "confidence": confidence,
    }


def news(*, headline: str, url: str, published_at: str, source: str | None) -> dict:
    return {"headline": headline, "url": url, "published_at": published_at, "source": source}

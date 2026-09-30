"""Open-Meteo : prévision météo au stade — SECOURS de la météo (sans clé).

⚠️ Offre gratuite réservée à un usage NON COMMERCIAL (< 10 000 appels/jour), licence CC BY 4.0.
Si Euria devient commercial : passer à l'offre payante d'Open-Meteo ou retirer ce secours.
Adresse : GET https://api.open-meteo.com/v1/forecast?latitude=..&longitude=..
          &hourly=temperature_2m,precipitation,wind_speed_10m&wind_speed_unit=ms&timezone=UTC
          &start_hour=AAAA-MM-JJTHH:00&end_hour=AAAA-MM-JJTHH:00
Horizon : 16 jours. Champs utilisés : hourly.time / temperature_2m / precipitation / wind_speed_10m."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .. import schema
from ..http import SourceError, get_json, require

NAME = "Open-Meteo"
ATTRIBUTION = "Données météo : Open-Meteo.com (CC BY 4.0)"
URL = "https://api.open-meteo.com/v1/forecast"
HORIZON = timedelta(days=16)


def _kickoff_hour(kickoff: datetime) -> datetime:
    rounded = kickoff.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return rounded + timedelta(hours=1) if kickoff.minute >= 30 else rounded


def parse_forecast(payload: dict, latitude: float, longitude: float) -> dict:
    hourly = payload.get("hourly") if isinstance(payload, dict) else None
    require(NAME, isinstance(hourly, dict) and hourly.get("time"), "hourly.time absent")
    for field in ("temperature_2m", "precipitation", "wind_speed_10m"):
        require(NAME, field in hourly, f"hourly.{field} absent")
    return schema.weather(
        latitude=latitude,
        longitude=longitude,
        time=hourly["time"][0] + ":00Z",
        temperature_c=hourly["temperature_2m"][0],
        precipitation_mm=hourly["precipitation"][0],
        wind_speed_ms=hourly["wind_speed_10m"][0],
    )


def fetch_weather(latitude: float, longitude: float, kickoff: datetime) -> dict:
    if kickoff - datetime.now(timezone.utc) > HORIZON:
        raise SourceError(NAME, "coup d'envoi hors de l'horizon de prévision (16 jours)")
    hour = _kickoff_hour(kickoff).strftime("%Y-%m-%dT%H:00")
    payload = get_json(
        NAME,
        URL,
        params={
            "latitude": round(latitude, 4),
            "longitude": round(longitude, 4),
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
            "start_hour": hour,
            "end_hour": hour,
        },
    )
    return parse_forecast(payload, round(latitude, 4), round(longitude, 4))

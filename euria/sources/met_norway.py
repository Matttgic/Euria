"""MET Norway (api.met.no) : prévision météo au stade — source PRINCIPALE de la météo (sans clé).

Adresse : GET https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=..&lon=..
Exige un User-Agent identifiant l'application (config.USER_AGENT). Licence CC BY 4.0 : usage
commercial autorisé, mention obligatoire. Horizon : environ 10 jours (pas horaire sur ~2,5 jours,
puis pas de 6 h).
Champs utilisés : properties.timeseries[].time, data.instant.details.air_temperature / wind_speed,
data.next_1_hours.details.precipitation_amount (ou next_6_hours, ramené à l'heure)."""

from __future__ import annotations

from datetime import datetime, timedelta

from .. import schema
from ..http import SourceError, get_json, require

NAME = "MET Norway"
ATTRIBUTION = "Données météo : MET Norway (CC BY 4.0)"
URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
MAX_GAP = timedelta(hours=3)  # écart maximal accepté entre le coup d'envoi et le pas de prévision


def parse_forecast(payload: dict, latitude: float, longitude: float, kickoff: datetime) -> dict:
    series = (payload.get("properties") or {}).get("timeseries") if isinstance(payload, dict) else None
    require(NAME, isinstance(series, list) and series, "properties.timeseries absent")
    best = min(series, key=lambda s: abs(schema.parse_utc(s["time"]) - kickoff))
    if abs(schema.parse_utc(best["time"]) - kickoff) > MAX_GAP:
        raise SourceError(NAME, "coup d'envoi hors de l'horizon de prévision")
    data = best["data"]
    details = data["instant"]["details"]
    if "next_1_hours" in data:
        precipitation = data["next_1_hours"]["details"].get("precipitation_amount")
    elif "next_6_hours" in data:
        amount = data["next_6_hours"]["details"].get("precipitation_amount")
        precipitation = round(amount / 6, 2) if amount is not None else None
    else:
        precipitation = None
    return schema.weather(
        latitude=latitude,
        longitude=longitude,
        time=best["time"],
        temperature_c=details.get("air_temperature"),
        precipitation_mm=precipitation,
        wind_speed_ms=details.get("wind_speed"),
    )


def fetch_weather(latitude: float, longitude: float, kickoff: datetime) -> dict:
    # MET Norway demande au plus 4 décimales (sinon le cache côté serveur est contourné).
    lat, lon = round(latitude, 4), round(longitude, 4)
    payload = get_json(NAME, URL, params={"lat": lat, "lon": lon})
    return parse_forecast(payload, lat, lon, kickoff)

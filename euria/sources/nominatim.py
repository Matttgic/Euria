"""Nominatim (OpenStreetMap) : coordonnées d'un stade à partir de son nom — seconde moitié du SECOURS des stades.

Sans clé. Règles d'usage : 1 requête par seconde au maximum, User-Agent identifiant l'application,
mention « © OpenStreetMap contributors » (licence ODbL).
Adresse : GET https://nominatim.openstreetmap.org/search?q={stade, ville}&format=jsonv2&limit=1
Champs : [0].lat / [0].lon (chaînes)."""

from __future__ import annotations

import threading
import time

from ..http import SourceError, get_json, require

NAME = "Nominatim"
ATTRIBUTION = "© OpenStreetMap contributors"
URL = "https://nominatim.openstreetmap.org/search"
_MIN_INTERVAL_S = 1.1
_lock = threading.Lock()
_last_call = 0.0


def parse_place(payload: list) -> tuple[float, float]:
    require(NAME, isinstance(payload, list), "liste attendue")
    if not payload:
        raise SourceError(NAME, "lieu introuvable")
    return float(payload[0]["lat"]), float(payload[0]["lon"])


def geocode(query: str) -> tuple[float, float]:
    global _last_call
    with _lock:  # respecte la limite d'1 requête/seconde, même avec plusieurs appels simultanés
        wait = _MIN_INTERVAL_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
    return parse_place(get_json(NAME, URL, params={"q": query, "format": "jsonv2", "limit": 1}))

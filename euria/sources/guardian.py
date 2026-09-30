"""The Guardian (Open Platform) : articles de foot récents sur les blessures — actus des alertes.

Clé gratuite « Developer » (GUARDIAN_API_KEY) : usage NON COMMERCIAL, 1 appel/seconde, 500/jour.
Remplace Noozra, bloqué sur GitHub Actions par une protection anti-robots (« Just a moment... »).
Adresse (documentation officielle) : GET https://content.guardianapis.com/search
          ?q={équipe} injury&section=football&from-date={AAAA-MM-JJ}&order-by=newest&page-size=5&api-key=…
Champs utilisés : response.status / response.results[].webTitle / webUrl / webPublicationDate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .. import config, schema
from ..http import SourceError, get_json, require

NAME = "The Guardian"
ATTRIBUTION = "Actus : The Guardian"
URL = "https://content.guardianapis.com/search"


def parse_search(payload: dict) -> list[dict]:
    response = payload.get("response") if isinstance(payload, dict) else None
    require(NAME, isinstance(response, dict) and isinstance(response.get("results"), list), "response.results absent")
    if response.get("status") != "ok":
        raise SourceError(NAME, f"statut {response.get('status')} : {response.get('message', '')}")
    return [
        schema.news(headline=r["webTitle"], url=r.get("webUrl", ""), published_at=r["webPublicationDate"], source=NAME)
        for r in response["results"]
        if r.get("webTitle") and r.get("webPublicationDate")
    ]


def search_injuries(team: str, limit: int = 5) -> list[dict]:
    if not config.GUARDIAN_API_KEY:
        raise SourceError(NAME, "clé GUARDIAN_API_KEY absente")
    since = (datetime.now(timezone.utc) - timedelta(days=config.NEWS_MAX_AGE_DAYS)).date()
    payload = get_json(
        NAME,
        URL,
        params={
            "q": f"{team} injury",
            "section": "football",
            "from-date": since.isoformat(),
            "order-by": "newest",
            "page-size": limit,
            "api-key": config.GUARDIAN_API_KEY,
        },
    )
    return parse_search(payload)

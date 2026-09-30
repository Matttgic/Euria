"""Noozra : recherche dans les titres d'actualité sportive (blessures, suspensions) — sans clé.

Sans clé : 100 requêtes/jour par IP (clé gratuite : 5 000/jour). Chaque titre renvoie vers l'article
de l'éditeur d'origine (Sky Sports, Independent…), en anglais pour l'essentiel.
Adresse : GET https://noozra.com/api/search?q={texte}&category=sports&limit={n}
Champs utilisés : articles[].headline / url / published_at / source."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .. import config, schema
from ..http import get_json, require

NAME = "Noozra"
ATTRIBUTION = "Actus : Noozra (liens vers les éditeurs d'origine)"
URL = "https://noozra.com/api/search"


def parse_search(payload: dict, max_age_days: int | None = None) -> list[dict]:
    articles = payload.get("articles") if isinstance(payload, dict) else None
    require(NAME, isinstance(articles, list), "clé 'articles' absente")
    oldest = None if max_age_days is None else datetime.now(timezone.utc) - timedelta(days=max_age_days)
    out = []
    for a in articles:
        if not a.get("headline") or not a.get("published_at"):
            continue
        if oldest and schema.parse_utc(a["published_at"]) < oldest:
            continue
        out.append(schema.news(headline=a["headline"], url=a.get("url", ""), published_at=a["published_at"], source=a.get("source")))
    return out


def search_injuries(team: str, limit: int = 5) -> list[dict]:
    payload = get_json(NAME, URL, params={"q": f"{team} injury", "category": "sports", "limit": limit})
    return parse_search(payload, config.NEWS_MAX_AGE_DAYS)

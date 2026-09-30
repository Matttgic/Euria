"""Parlay API : cotes 1N2 de plusieurs bookmakers (PMU, Unibet, bet365, Pinnacle…) — source PRINCIPALE des cotes.

Clé gratuite (PARLAY_API_KEY) : 1 000 crédits/mois, 1 crédit par appel pour <= 10 bookmakers.
Adresse : GET https://parlay-api.com/v1/sports/{sport_key}/odds
          ?markets=h2h&oddsFormat=decimal&bookmakers=pmu,unibet,bet365,pinnacle   (en-tête X-API-Key)
Champs utilisés (vus dans la réponse de démonstration /v1/try/soccer_epl/odds) : commence_time,
home_team, away_team, bookmakers[].key / last_update / markets[key=h2h].outcomes[].name / price.

Conditions : usage interne uniquement, pas de republication des cotes brutes, pas de conservation
des cotes détaillées au-delà de 90 jours, clé jamais dans le code ni dans un dépôt."""

from __future__ import annotations

from .. import config, schema
from ..config import League
from ..http import SourceError, get_json, require

NAME = "Parlay API"
ATTRIBUTION = "Cotes : Parlay API"
BASE_URL = "https://parlay-api.com/v1"


def to_decimal(price: float) -> float:
    """La démo renvoie des cotes américaines (-262, +665) même avec oddsFormat=decimal.
    Une cote décimale 1N2 est toujours comprise entre 1 et 100 ; au-delà (ou négative), c'est de l'américain."""
    price = float(price)
    if price < 0:
        return 1 + 100 / abs(price)
    if price >= 100:
        return 1 + price / 100
    return price


def parse_odds(payload, league: League) -> list[dict]:
    events = payload.get("events") if isinstance(payload, dict) else payload
    require(NAME, isinstance(events, list), "liste d'événements attendue")
    out = []
    for event in events:
        home, away = event["home_team"], event["away_team"]
        for book in event.get("bookmakers") or []:
            market = next((m for m in book.get("markets") or [] if m.get("key") == "h2h"), None)
            if not market:
                continue
            prices = {o["name"]: to_decimal(o["price"]) for o in market.get("outcomes") or []}
            if not {home, away, "Draw"} <= prices.keys():
                continue  # marché incomplet (ex. sans le nul) : inutilisable pour du 1N2
            out.append(
                schema.odds(
                    league=league.code,
                    utc_date=event["commence_time"],
                    home=home,
                    away=away,
                    bookmaker=book["key"],
                    home_odds=prices[home],
                    draw_odds=prices["Draw"],
                    away_odds=prices[away],
                    updated_at=market.get("last_update") or book.get("last_update"),
                )
            )
    return out


def fetch_odds(league: League) -> list[dict]:
    if not config.PARLAY_API_KEY:
        raise SourceError(NAME, "clé PARLAY_API_KEY absente")
    payload = get_json(
        NAME,
        f"{BASE_URL}/sports/{league.parlay}/odds",
        params={"markets": "h2h", "oddsFormat": "decimal", "bookmakers": ",".join(config.ODDS_BOOKMAKERS)},
        headers={"X-API-Key": config.PARLAY_API_KEY},
    )
    return parse_odds(payload, league)

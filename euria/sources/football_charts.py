"""Football Charts : points réels contre points attendus (« chance ») de chaque équipe — sans clé.

Sans clé : 300 requêtes/jour et 20/minute par IP (clé gratuite : 5 000/jour).
Conditions : usage personnel ou de recherche, mention « Data by football-charts.com », pas de revente.
Adresse : GET https://footballcharts-backend.onrender.com/api/v1/leagues/{premier|spain1|italy1|germany1|france1}/table/
Champs utilisés : table[].team / played / points / expected_points / last_5_form."""

from __future__ import annotations

from .. import schema
from ..config import League
from ..http import get_json, require

NAME = "Football Charts"
ATTRIBUTION = "Data by football-charts.com"
BASE_URL = "https://footballcharts-backend.onrender.com/api/v1/leagues"


def parse_table(payload: dict) -> list[dict]:
    table = payload.get("table") if isinstance(payload, dict) else None
    require(NAME, isinstance(table, list), "clé 'table' absente")
    out = []
    for row in table:
        require(NAME, {"team", "played", "points", "expected_points"} <= row.keys(), "champs du classement absents")
        if row["expected_points"] is None:
            continue
        out.append(
            schema.xpoints(
                team=row["team"], played=row["played"], points=row["points"],
                expected_points=row["expected_points"], form=row.get("last_5_form"),
            )
        )
    return out


def fetch_table(league: League) -> list[dict]:
    return parse_table(get_json(NAME, f"{BASE_URL}/{league.football_charts}/table/"))

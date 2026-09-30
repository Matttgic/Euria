"""football-data.co.uk : CSV des résultats de la saison — SECOURS des matchs (sans clé).

Hors annuaire Public APIs (validé). Mis à jour environ deux fois par semaine : les derniers
matchs peuvent manquer. Ne contient que des matchs joués (pas de calendrier à venir).
Adresse : GET https://football-data.co.uk/mmz4281/{2627}/{E0|F1|D1|I1|SP1}.csv
Colonnes utilisées : Date (jj/mm/aaaa), Time (heure de Londres), HomeTeam, AwayTeam, FTHG, FTAG."""

from __future__ import annotations

import csv
import io
from datetime import datetime
from zoneinfo import ZoneInfo

from .. import schema
from ..config import League
from ..http import get_text, require

NAME = "football-data.co.uk"
ATTRIBUTION = "Résultats : football-data.co.uk"
BASE_URL = "https://football-data.co.uk/mmz4281"
_LONDON = ZoneInfo("Europe/London")
REQUIRED_COLUMNS = {"Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}


def season_folder(season: int) -> str:
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def _kickoff(date_str: str, time_str: str) -> str:
    fmt = "%d/%m/%Y" if len(date_str.split("/")[-1]) == 4 else "%d/%m/%y"
    local = datetime.strptime(f"{date_str} {time_str or '15:00'}", f"{fmt} %H:%M").replace(tzinfo=_LONDON)
    return schema.iso_utc(local)


def parse_csv(text: str, league: League, season: int) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    require(NAME, REQUIRED_COLUMNS <= set(reader.fieldnames or []), f"colonnes attendues {sorted(REQUIRED_COLUMNS)}")
    out = []
    for row in reader:
        if not row.get("HomeTeam") or row.get("FTHG") in (None, ""):
            continue
        out.append(
            schema.match(
                league=league.code,
                season=season,
                utc_date=_kickoff(row["Date"], row.get("Time", "")),
                home=row["HomeTeam"],
                away=row["AwayTeam"],
                status="FINISHED",
                home_goals=int(row["FTHG"]),
                away_goals=int(row["FTAG"]),
            )
        )
    return out


def fetch_matches(league: League, season: int) -> list[dict]:
    text = get_text(NAME, f"{BASE_URL}/{season_folder(season)}/{league.couk}.csv")
    return parse_csv(text, league, season)

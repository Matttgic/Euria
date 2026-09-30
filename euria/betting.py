"""Suivi des paris (data/suivi_paris.csv) : ajout, règlement avec les scores réels, bilan.

Le fichier garde les colonnes historiques (Match, Pari, Cote, Value, Date, Result) et en ajoute
trois (Ligue, Bookmaker, CoupEnvoi). Les anciennes lignes restent lisibles telles quelles."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config, services
from .config import LEAGUES, season_for
from .schema import parse_utc
from .teams import same_team

COLUMNS = ["Match", "Pari", "Cote", "Value", "Date", "Result", "Ligue", "Bookmaker", "CoupEnvoi"]


def load(path: Path | None = None) -> list[dict]:
    path = path or config.BETS_FILE
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [{col: (row.get(col) or "") for col in COLUMNS} for row in csv.DictReader(f)]


def save(rows: list[dict], path: Path | None = None) -> None:
    path = path or config.BETS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def bet_key(match: str, kickoff_or_date: str) -> str:
    return f"{match}|{kickoff_or_date[:10]}"


def existing_keys(rows: list[dict]) -> set[str]:
    return {bet_key(r["Match"], r["CoupEnvoi"] or r["Date"]) for r in rows}


def _parse_bet_date(value: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _winner(m: dict) -> str:
    if m["home_goals"] > m["away_goals"]:
        return "Home"
    return "Away" if m["away_goals"] > m["home_goals"] else "Draw"


def _find_result(row: dict) -> tuple[str, dict] | None:
    if " vs " not in row["Match"]:
        return None
    home, away = row["Match"].split(" vs ", 1)
    if row["CoupEnvoi"]:
        kickoff = parse_utc(row["CoupEnvoi"])
        window = (kickoff - timedelta(days=1), kickoff + timedelta(days=1))
    else:  # anciennes lignes : seule la date du pari est connue, le match suit dans les 3 semaines
        placed = _parse_bet_date(row["Date"])
        if placed is None:
            return None
        kickoff = placed
        window = (placed - timedelta(days=1), placed + timedelta(days=21))
    for code in [row["Ligue"]] if row["Ligue"] in LEAGUES else list(LEAGUES):
        res = services.matches(code, season_for(kickoff))
        if not res.ok:
            continue
        for m in res.data:
            if m["status"] != "FINISHED" or m["home_goals"] is None:
                continue
            if not (window[0] <= parse_utc(m["utc_date"]) <= window[1]):
                continue
            if same_team(home, m["home"]) and same_team(away, m["away"]):
                return code, m
    return None


def settle(rows: list[dict]) -> int:
    """Renseigne Result (Win/Loss) pour les paris dont le match est terminé. Renvoie le nombre réglé."""
    settled = 0
    for row in rows:
        if row["Result"]:
            continue
        found = _find_result(row)
        if not found:
            continue
        code, m = found
        row["Result"] = "Win" if row["Pari"] == _winner(m) else "Loss"
        row["Ligue"] = row["Ligue"] or code
        settled += 1
    return settled


def summary(rows: list[dict]) -> str:
    closed = [r for r in rows if r["Result"] in ("Win", "Loss")]
    if not closed:
        return "En attente des premiers résultats terminés..."
    wins = sum(1 for r in closed if r["Result"] == "Win")
    profit = sum((float(r["Cote"]) - 1) * config.STAKE_EUR if r["Result"] == "Win" else -config.STAKE_EUR for r in closed)
    roi = profit / (config.STAKE_EUR * len(closed)) * 100
    pending = sum(1 for r in rows if not r["Result"])
    return (
        "📊 *BILAN AUTOMATIQUE*\n"
        f"✅ Réussite : {wins / len(closed) * 100:.1f} %\n"
        f"💰 Profit : {profit:.2f} € (mise {config.STAKE_EUR:.0f} €, ROI {roi:.1f} %)\n"
        f"Total : {len(closed)} paris clôturés, {pending} en attente"
    )

"""Suivi des paris (data/suivi_paris.csv) : ajout, règlement avec les scores réels, bilan.

Le fichier garde les colonnes historiques (Match, Pari, Cote, Value, Date, Result) et en ajoute
trois (Ligue, Bookmaker, CoupEnvoi), puis trois pour la CLV (CoteCloture, SourceCloture, CLV).
Les anciennes lignes restent lisibles telles quelles."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config, market, services
from .config import LEAGUES, season_for
from .schema import parse_utc
from .teams import same_team

COLUMNS = [
    "Match", "Pari", "Cote", "Value", "Date", "Result", "Ligue", "Bookmaker", "CoupEnvoi",
    "CoteCloture", "SourceCloture", "CLV",
]
OUTCOME_INDEX = {"Home": 0, "Draw": 1, "Away": 2}


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


def _search_window(row: dict) -> tuple[datetime, datetime] | None:
    """Fenêtre où chercher le match du pari."""
    if row["CoupEnvoi"]:
        kickoff = parse_utc(row["CoupEnvoi"])
        return kickoff - timedelta(days=1), kickoff + timedelta(days=1)
    # anciennes lignes : seule la date du pari est connue, le match suit dans les 3 semaines
    placed = _parse_bet_date(row["Date"])
    if placed is None:
        return None
    return placed - timedelta(days=1), placed + timedelta(days=21)


def _leagues_to_search(row: dict) -> list[str]:
    return [row["Ligue"]] if row["Ligue"] in LEAGUES else list(LEAGUES)


def _find_result(row: dict) -> tuple[str, dict] | None:
    if " vs " not in row["Match"]:
        return None
    home, away = row["Match"].split(" vs ", 1)
    window = _search_window(row)
    if window is None:
        return None
    kickoff = window[0] + timedelta(days=1)
    for code in _leagues_to_search(row):
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


def _find_closing(row: dict) -> dict | None:
    if " vs " not in row["Match"]:
        return None
    home, away = row["Match"].split(" vs ", 1)
    window = _search_window(row)
    if window is None:
        return None
    for code in _leagues_to_search(row):
        res = services.closing_odds(code, season_for(window[0] + timedelta(days=1)))
        if not res.ok:
            continue
        for m in res.data:
            if window[0] <= parse_utc(m["utc_date"]) <= window[1] and same_team(home, m["home"]) and same_team(away, m["away"]):
                return m
    return None


def fill_clv(rows: list[dict]) -> int:
    """Renseigne la cote de clôture juste (marge retirée, méthode de Shin) et la CLV des paris réglés.

    CLV = cote prise × probabilité de clôture − 1. Positive en moyenne = on bat le marché."""
    filled = 0
    for row in rows:
        if row["CLV"] or row["Result"] not in ("Win", "Loss") or row["Pari"] not in OUTCOME_INDEX:
            continue
        found = _find_closing(row)
        if not found:
            continue
        probs = market.implied_shin(found["odds"])
        if not probs:
            continue
        p = probs[OUTCOME_INDEX[row["Pari"]]]
        row["CoteCloture"] = f"{1 / p:.3f}"
        row["SourceCloture"] = found["source"]
        row["CLV"] = f"{market.clv(float(row['Cote']), p):.4f}"
        filled += 1
    return filled


def clv_summary(rows: list[dict]) -> dict | None:
    values = [float(r["CLV"]) for r in rows if r["CLV"]]
    if not values:
        return None
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "beat_close": sum(1 for v in values if v > 0) / len(values),
    }


def summary(rows: list[dict]) -> str:
    closed = [r for r in rows if r["Result"] in ("Win", "Loss")]
    if not closed:
        return "En attente des premiers résultats terminés..."
    wins = sum(1 for r in closed if r["Result"] == "Win")
    profit = sum((float(r["Cote"]) - 1) * config.STAKE_EUR if r["Result"] == "Win" else -config.STAKE_EUR for r in closed)
    roi = profit / (config.STAKE_EUR * len(closed)) * 100
    pending = sum(1 for r in rows if not r["Result"])
    text = (
        "📊 *BILAN AUTOMATIQUE*\n"
        f"✅ Réussite : {wins / len(closed) * 100:.1f} %\n"
        f"💰 Profit : {profit:.2f} € (mise {config.STAKE_EUR:.0f} €, ROI {roi:.1f} %)\n"
        f"Total : {len(closed)} paris clôturés, {pending} en attente"
    )
    clv = clv_summary(rows)
    if clv:
        text += (
            f"\n📈 CLV moyenne : {clv['mean'] * 100:+.1f} % sur {clv['n']} paris "
            f"({clv['beat_close'] * 100:.0f} % au-dessus de la clôture)\n"
            "_Cotes de clôture : football-data.co.uk_"
        )
    return text

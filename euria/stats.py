"""Classement et variables du modèle, calculés uniquement à partir des scores.

Le même calcul s'applique quelle que soit la source des matchs : c'est ce qui permet à la source
de secours de remplacer la principale sans changer les prédictions."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from . import config
from .schema import parse_utc
from .teams import best_match

# Ordre exact des 12 variables attendu par models/model_sklearn.pkl
# (reprise de l'ancien main.py : « 12 features exactes comme dans le Colab »).
FEATURE_NAMES = [
    "home_gf", "home_ga", "home_cs", "home_fw", "home_fd", "home_fl",
    "away_gf", "away_ga", "away_cs", "away_fw", "away_fd", "away_fl",
]


def finished_before(matches: Iterable[dict], before: datetime | None = None) -> list[dict]:
    done = [m for m in matches if m["status"] == "FINISHED" and m["home_goals"] is not None]
    if before is not None:
        done = [m for m in done if parse_utc(m["utc_date"]) < before]
    return sorted(done, key=lambda m: m["utc_date"])


def team_names(matches: Iterable[dict]) -> set[str]:
    return {m["home"] for m in matches} | {m["away"] for m in matches}


def resolve_team(name: str, matches: Iterable[dict]) -> str | None:
    """Nom de l'équipe tel qu'il est écrit dans cette liste de matchs (sources différentes)."""
    return best_match(name, team_names(matches))


def _team_results(finished: list[dict], team: str) -> list[tuple[int, int]]:
    """(buts marqués, buts encaissés) de l'équipe, dans l'ordre chronologique."""
    out = []
    for m in finished:
        if m["home"] == team:
            out.append((m["home_goals"], m["away_goals"]))
        elif m["away"] == team:
            out.append((m["away_goals"], m["home_goals"]))
    return out


def team_profile(finished: list[dict], team: str) -> dict | None:
    results = _team_results(finished, team)
    if len(results) < config.MIN_MATCHES_PLAYED:
        return None
    last5 = results[-5:]
    return {
        "played": len(results),
        "gf": sum(g for g, _ in results) / len(results),  # buts marqués par match (saison)
        "ga": sum(a for _, a in results) / len(results),  # buts encaissés par match (saison)
        "cs": sum(1 for _, a in results if a == 0),  # matchs sans encaisser de but (total)
        "fw": sum(1 for g, a in last5 if g > a),  # forme : victoires sur les 5 derniers
        "fd": sum(1 for g, a in last5 if g == a),
        "fl": sum(1 for g, a in last5 if g < a),
    }


def features(matches: list[dict], home: str, away: str, before: datetime) -> dict | None:
    """Variables du modèle pour home vs away, avec les seuls matchs joués avant le coup d'envoi.
    Renvoie None si une équipe est introuvable ou n'a pas assez de matchs joués."""
    finished = finished_before(matches, before)
    home_name, away_name = resolve_team(home, matches), resolve_team(away, matches)
    if not home_name or not away_name:
        return None
    h, a = team_profile(finished, home_name), team_profile(finished, away_name)
    if not h or not a:
        return None
    vector = [h["gf"], h["ga"], h["cs"], h["fw"], h["fd"], h["fl"], a["gf"], a["ga"], a["cs"], a["fw"], a["fd"], a["fl"]]
    return {"home_team": home_name, "away_team": away_name, "home": h, "away": a, "vector": vector}


def standings(matches: list[dict], before: datetime | None = None) -> list[dict]:
    """Classement simplifié : points, puis différence de buts, puis buts marqués.
    (Les confrontations directes, critère officiel en Liga et en Serie A, ne sont pas prises en compte.)"""
    table: dict[str, dict] = {}
    for m in finished_before(matches, before):
        for team, gf, ga in ((m["home"], m["home_goals"], m["away_goals"]), (m["away"], m["away_goals"], m["home_goals"])):
            row = table.setdefault(team, {"team": team, "played": 0, "won": 0, "drawn": 0, "lost": 0, "gf": 0, "ga": 0, "points": 0})
            row["played"] += 1
            row["gf"] += gf
            row["ga"] += ga
            if gf > ga:
                row["won"] += 1
                row["points"] += 3
            elif gf == ga:
                row["drawn"] += 1
                row["points"] += 1
            else:
                row["lost"] += 1
    rows = sorted(table.values(), key=lambda r: (-r["points"], -(r["gf"] - r["ga"]), -r["gf"], r["team"]))
    for rank, row in enumerate(rows, start=1):
        row["goal_diff"] = row["gf"] - row["ga"]
        row["rank"] = rank
    return rows

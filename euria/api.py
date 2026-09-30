"""API FastAPI d'Euria. Lancement : uvicorn main:app

Chaque réponse a la forme {"data": ..., "meta": {"source", "updated_at", "stale", "message", "attribution"}} :
la source et l'heure de mise à jour accompagnent toujours la donnée. Si toutes les sources sont
en panne, data vaut la dernière valeur connue (stale = true) ou null, avec un message clair —
jamais une erreur 500."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import __version__, analysis, services, stats
from .config import LEAGUES
from .fallback import Result
from .schema import parse_utc
from .teams import same_team

app = FastAPI(title="Euria", version=__version__, description="Value bets sur les 5 grands championnats européens.")


def _league(code: str) -> str:
    code = code.upper()
    if code not in LEAGUES:
        raise HTTPException(404, f"Championnat inconnu : {code}. Choix : {', '.join(LEAGUES)}")
    return code


def _envelope(result: Result, data=None) -> dict:
    return {"data": result.data if data is None and result.ok else data, "meta": result.meta()}


@app.exception_handler(Exception)
async def _unexpected(_, exc: Exception):  # dernier filet : pas de page cassée
    return JSONResponse(status_code=500, content={"data": None, "meta": {"message": f"Erreur interne : {exc.__class__.__name__}"}})


@app.get("/")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/leagues")
def leagues() -> dict:
    return {"data": [{"code": l.code, "name": l.name} for l in LEAGUES.values()]}


@app.get("/matches/{league}")
def get_matches(league: str, status: Literal["all", "upcoming", "finished"] = "all") -> dict:
    result = services.matches(_league(league))
    if not result.ok:
        return _envelope(result)
    data = result.data
    if status == "upcoming":
        data = [m for m in data if m["status"] == "SCHEDULED"]
    elif status == "finished":
        data = [m for m in data if m["status"] == "FINISHED"]
    return _envelope(result, sorted(data, key=lambda m: m["utc_date"]))


@app.get("/standings/{league}")
def get_standings(league: str) -> dict:
    result = services.matches(_league(league))
    if not result.ok:
        return _envelope(result)
    meta_note = "Classement calculé à partir des scores (points, différence de buts, buts marqués)."
    body = _envelope(result, stats.standings(result.data))
    body["meta"]["note"] = meta_note
    return body


@app.get("/odds/{league}")
def get_odds(league: str) -> dict:
    return _envelope(services.odds(_league(league)))


@app.get("/weather")
def get_weather(
    home: str = Query(..., description="Équipe à domicile (le stade est le sien)"),
    kickoff: datetime = Query(..., description="Coup d'envoi, ISO 8601, ex. 2026-10-10T11:30:00Z"),
) -> dict:
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    return _envelope(services.weather(home, kickoff))


class PredictRequest(BaseModel):
    league: str
    home: str
    away: str
    kickoff: datetime | None = None  # facultatif : retrouvé dans les cotes ou le calendrier


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    code = _league(req.league)
    matches = services.matches(code)
    odds = services.odds(code)
    event = analysis.find_event(odds.data or [], req.home, req.away)
    kickoff = req.kickoff
    if kickoff is None and event:
        kickoff = parse_utc(event["utc_date"])
    if kickoff is None and matches.ok:
        upcoming = [m for m in matches.data if m["status"] == "SCHEDULED" and same_team(m["home"], req.home) and same_team(m["away"], req.away)]
        kickoff = parse_utc(upcoming[0]["utc_date"]) if upcoming else None
    if kickoff is None:
        kickoff = datetime.now(timezone.utc) + timedelta(hours=1)
    elif kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    report = analysis.analyze(code, req.home, req.away, kickoff, matches, event["quotes"] if event else None)
    report["sources"]["odds"] = odds.meta()
    attributions = sorted({s["attribution"] for s in report["sources"].values() if s and s.get("attribution")})
    return {"data": report, "meta": {"attribution": attributions, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}}

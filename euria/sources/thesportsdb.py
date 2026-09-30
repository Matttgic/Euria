"""TheSportsDB : nom du stade et ville d'un club — première moitié du SECOURS des stades.

Clé publique de test « 123 » (documentée par TheSportsDB, ce n'est pas un secret) : suffisante
pour la recherche d'équipe, trop limitée pour les calendriers (1 à 5 résultats), d'où son rôle ici.
Adresse : GET https://www.thesportsdb.com/api/v1/json/123/searchteams.php?t={nom}
Champs : teams[].strTeam / strSport / strStadium / strLocation."""

from __future__ import annotations

from ..http import SourceError, get_json, require

NAME = "TheSportsDB"
ATTRIBUTION = "Stades : TheSportsDB"
URL = "https://www.thesportsdb.com/api/v1/json/123/searchteams.php"


def parse_team(payload: dict) -> dict:
    require(NAME, isinstance(payload, dict) and "teams" in payload, "clé 'teams' absente")
    teams = [t for t in payload["teams"] or [] if t.get("strSport") == "Soccer" and t.get("strStadium")]
    if not teams:
        raise SourceError(NAME, "équipe introuvable")
    team = teams[0]
    return {"team": team["strTeam"], "stadium": team["strStadium"], "location": team.get("strLocation") or ""}


def fetch_team(name: str) -> dict:
    return parse_team(get_json(NAME, URL, params={"t": name}))

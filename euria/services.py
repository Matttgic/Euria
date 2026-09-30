"""Données prêtes à l'emploi pour le bot et l'API : chaque fonction applique la chaîne
principale -> secours -> dernière valeur connue, avec le cache adapté, et renvoie un fallback.Result."""

from __future__ import annotations

from datetime import datetime, timezone

from . import cache, config, schema
from .config import TTL, get_league, season_for
from .fallback import Provider, Result, fetch
from .http import SourceError
from .sources import (
    football_data_couk,
    football_data_org,
    met_norway,
    nominatim,
    open_meteo,
    openligadb,
    parlay,
    thesportsdb,
    wikidata,
)
from .teams import best_match, normalize


# ---------------------------------------------------------------------------
# Matchs (calendrier + scores)
# ---------------------------------------------------------------------------
def matches(league_code: str, season: int | None = None) -> Result:
    league = get_league(league_code)
    season = season if season is not None else season_for()
    providers = [
        Provider(football_data_org.NAME, lambda: football_data_org.fetch_matches(league, season), football_data_org.ATTRIBUTION),
        Provider(football_data_couk.NAME, lambda: football_data_couk.fetch_matches(league, season), football_data_couk.ATTRIBUTION),
    ]
    if league.openligadb:
        providers.append(Provider(openligadb.NAME, lambda: openligadb.fetch_matches(league, season), openligadb.ATTRIBUTION))
    # Une saison terminée ne change plus : inutile de la relire toutes les heures.
    ttl = TTL["matches"] if season >= season_for() else TTL["venues"]
    return fetch(f"matches:{league.code}:{season}", ttl, providers)


# ---------------------------------------------------------------------------
# Cotes 1N2 — pas de secours gratuit : API-Football (offre gratuite) n'a plus accès à la saison
# en cours (« Free plans do not have access to this season », constaté le 30/09/2026). En cas de
# panne de Parlay, la dernière valeur connue est utilisée.
# ---------------------------------------------------------------------------
def odds(league_code: str) -> Result:
    league = get_league(league_code)
    cache.prune("odds:", config.ODDS_MAX_RETENTION_S)  # conditions Parlay : 90 jours maximum
    return fetch(
        f"odds:{league.code}",
        TTL["odds"],
        [
            Provider(parlay.NAME, lambda: parlay.fetch_odds(league), parlay.ATTRIBUTION),
        ],
    )


# ---------------------------------------------------------------------------
# Stades
# ---------------------------------------------------------------------------
_SPORTSDB_OSM = "TheSportsDB + Nominatim"


def _wikidata_venues() -> Result:
    return fetch("venues:wikidata:all", TTL["venues"], [Provider(wikidata.NAME, wikidata.fetch_all_venues)])


def _venue_from_wikidata(team: str) -> dict:
    listing = _wikidata_venues()
    if not listing.ok:
        raise SourceError(wikidata.NAME, listing.message or "liste des stades indisponible")
    by_name = {v["team"]: v for v in listing.data}
    found = best_match(team, by_name)
    if not found:
        raise SourceError(wikidata.NAME, f"club « {team} » absent")
    return {**by_name[found], "team": team}


def _venue_from_sportsdb_osm(team: str) -> dict:
    info = thesportsdb.fetch_team(team)
    query = ", ".join(part for part in (info["stadium"], info["location"]) if part)
    lat, lon = nominatim.geocode(query)
    return schema.venue(team=team, stadium=info["stadium"], latitude=lat, longitude=lon)


def venue(team: str) -> Result:
    return fetch(
        f"venue:{normalize(team)}",
        TTL["venues"],
        [
            Provider(wikidata.NAME, lambda: _venue_from_wikidata(team), wikidata.ATTRIBUTION),
            Provider(_SPORTSDB_OSM, lambda: _venue_from_sportsdb_osm(team), f"{thesportsdb.ATTRIBUTION} · {nominatim.ATTRIBUTION}"),
        ],
    )


# ---------------------------------------------------------------------------
# Météo au stade, à l'heure du coup d'envoi
# ---------------------------------------------------------------------------
def weather(home_team: str, kickoff: datetime) -> Result:
    place = venue(home_team)
    if not place.ok:
        return Result(None, None, None, stale=True, message=f"Météo indisponible : stade de {home_team} inconnu.", errors=place.errors)
    lat, lon = place.data["latitude"], place.data["longitude"]
    kickoff = kickoff.astimezone(timezone.utc)
    result = fetch(
        f"weather:{lat:.3f},{lon:.3f}:{kickoff:%Y-%m-%dT%H}",
        TTL["weather"],
        [
            Provider(met_norway.NAME, lambda: met_norway.fetch_weather(lat, lon, kickoff), met_norway.ATTRIBUTION),
            Provider(open_meteo.NAME, lambda: open_meteo.fetch_weather(lat, lon, kickoff), open_meteo.ATTRIBUTION),
        ],
    )
    if result.ok:
        result.data = {**result.data, "stadium": place.data.get("stadium"), "kickoff": schema.iso_utc(kickoff)}
    return result

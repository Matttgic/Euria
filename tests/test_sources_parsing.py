"""Chaque source, appliquée à une vraie réponse capturée le 30/09/2026, renvoie le format commun."""

from datetime import datetime, timedelta, timezone

import pytest

from euria import schema
from euria.config import LEAGUES
from euria.http import SourceError
from euria.sources import betbetter, football_charts, guardian, football_data_couk, football_data_org, met_norway, nominatim, open_meteo, openligadb, parlay, thesportsdb, wikidata
from tests.conftest import load_sample

MATCH_KEYS = set(schema.match(league="PL", season=2026, utc_date="x", home="a", away="b", status="SCHEDULED"))
ODDS_KEYS = set(schema.odds(league="PL", utc_date="x", home="a", away="b", bookmaker="k", home_odds=2, draw_odds=3, away_odds=4))
WEATHER_KEYS = set(schema.weather(latitude=0, longitude=0, time="x", temperature_c=0, precipitation_mm=0, wind_speed_ms=0))


def test_openligadb():
    matches = openligadb.parse_matches(load_sample("openligadb_bl1.json"), LEAGUES["BL1"], 2026)
    assert all(set(m) == MATCH_KEYS for m in matches)
    first = matches[0]
    assert (first["home"], first["away"], first["home_goals"], first["away_goals"]) == ("FC Bayern München", "VfB Stuttgart", 5, 1)
    assert matches[-1]["status"] == "SCHEDULED" and matches[-1]["home_goals"] is None


def test_football_data_couk_converts_london_time_to_utc():
    matches = football_data_couk.parse_csv(load_sample("couk_E0.csv"), LEAGUES["PL"], 2026)
    assert all(set(m) == MATCH_KEYS for m in matches)
    # 21/08/2026 20:00 à Londres (heure d'été) = 19:00 UTC
    assert matches[0]["utc_date"] == "2026-08-21T19:00:00Z"
    assert (matches[0]["home"], matches[0]["away"], matches[0]["home_goals"], matches[0]["away_goals"]) == ("Arsenal", "Coventry", 3, 0)


def test_football_data_couk_closing_odds_prefers_pinnacle():
    closing = football_data_couk.parse_closing_odds(load_sample("couk_E0_2526_closing.csv"))
    assert len(closing) == 5
    first = closing[0]
    assert (first["home"], first["away"], first["utc_date"]) == ("Liverpool", "Bournemouth", "2025-08-15T19:00:00Z")
    assert first["source"] == "Pinnacle" and first["odds"] == [1.29, 6.55, 9.75]


def test_football_data_couk_closing_odds_falls_back_to_market_average():
    text = ("Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA\n"
            "16/08/2025,15:00,Brighton,Fulham,1,1,,,,1.85,3.7,4.3\n"
            "16/08/2025,15:00,Leeds,Everton,1,0,,,,,,\n")
    closing = football_data_couk.parse_closing_odds(text)
    assert len(closing) == 1  # sans aucune cote de clôture, le match est ignoré
    assert closing[0]["source"] == "Moyenne marché" and closing[0]["odds"] == [1.85, 3.7, 4.3]


def test_football_data_couk_rejects_unexpected_columns():
    with pytest.raises(SourceError):
        football_data_couk.parse_csv("a,b\n1,2\n", LEAGUES["PL"], 2026)


def test_football_data_org_documented_format():
    # Exemple tiré de la documentation officielle v4 (ressource Match).
    payload = {"matches": [{
        "utcDate": "2022-02-27T16:05:00Z", "status": "FINISHED", "venue": "Stade de l'Aube",
        "homeTeam": {"name": "ES Troyes AC", "shortName": "Troyes"},
        "awayTeam": {"name": "Olympique de Marseille", "shortName": "Marseille"},
        "score": {"fullTime": {"home": 1, "away": 1}},
    }, {
        "utcDate": "2022-03-05T16:00:00Z", "status": "TIMED", "venue": None,
        "homeTeam": {"name": "Olympique de Marseille", "shortName": "Marseille"},
        "awayTeam": {"name": "ES Troyes AC", "shortName": "Troyes"},
        "score": {"fullTime": {"home": None, "away": None}},
    }]}
    matches = football_data_org.parse_matches(payload, LEAGUES["FL1"], 2021)
    assert matches[0] == {**matches[0], "home": "Troyes", "status": "FINISHED", "home_goals": 1, "venue": "Stade de l'Aube"}
    assert matches[1]["status"] == "SCHEDULED" and matches[1]["home_goals"] is None


def test_parlay_converts_american_odds():
    quotes = parlay.parse_odds(load_sample("parlay_demo_epl.json"), LEAGUES["PL"])
    assert quotes and all(set(q) == ODDS_KEYS for q in quotes)
    arsenal = quotes[0]
    assert (arsenal["home"], arsenal["away"], arsenal["bookmaker"]) == ("Arsenal", "Leeds United", "pinnacle")
    assert arsenal["home_odds"] == pytest.approx(1.4)  # -250 en cote américaine
    assert arsenal["away_odds"] == pytest.approx(7.0)  # +600


def test_parlay_decimal_passthrough():
    assert parlay.to_decimal(2.15) == 2.15
    assert parlay.to_decimal(-200) == pytest.approx(1.5)
    assert parlay.to_decimal(150) == pytest.approx(2.5)


def test_met_norway_hourly_and_six_hourly():
    payload = load_sample("met_norway.json")
    hourly = met_norway.parse_forecast(payload, 48.84, 2.25, datetime(2026, 9, 30, 19, 10, tzinfo=timezone.utc))
    assert set(hourly) == WEATHER_KEYS and hourly["time"] == "2026-09-30T19:00:00Z"
    six = met_norway.parse_forecast(payload, 48.84, 2.25, datetime(2026, 10, 3, 13, 0, tzinfo=timezone.utc))
    assert six["time"] == "2026-10-03T12:00:00Z" and six["precipitation_mm"] is not None


def test_met_norway_beyond_horizon():
    with pytest.raises(SourceError):
        met_norway.parse_forecast(load_sample("met_norway.json"), 0, 0, datetime(2026, 11, 1, tzinfo=timezone.utc))


def test_open_meteo():
    weather = open_meteo.parse_forecast(load_sample("open_meteo.json"), 48.84, 2.25)
    assert weather == {**weather, "time": "2026-10-12T19:00:00Z", "temperature_c": 13.0, "wind_speed_ms": 0.78}
    assert set(weather) == WEATHER_KEYS


def test_wikidata_venues():
    venues = wikidata.parse_venues(load_sample("wikidata_venues.json"))
    psg = next(v for v in venues if v["team"] == "Paris Saint-Germain FC")
    assert psg["stadium"] == "Parc des Princes"
    assert psg["latitude"] == pytest.approx(48.84, abs=0.01) and psg["longitude"] == pytest.approx(2.25, abs=0.01)


def test_thesportsdb_and_nominatim():
    assert thesportsdb.parse_team(load_sample("thesportsdb_barcelona.json"))["stadium"] == "Spotify Camp Nou"
    lat, lon = nominatim.parse_place(load_sample("nominatim_campnou.json"))
    assert lat == pytest.approx(41.38, abs=0.01) and lon == pytest.approx(2.12, abs=0.01)
    with pytest.raises(SourceError):
        nominatim.parse_place([])


def test_football_charts_luck():
    rows = football_charts.parse_table(load_sample("football_charts_premier_table.json"))
    city = rows[0]
    assert (city["team"], city["points"], city["expected_points"], city["luck"]) == ("Manchester City", 15, 10.51, 4.49)


def test_betbetter_away_at_home_convention():
    opinions = betbetter.parse_picks(load_sample("betbetter_bundesliga_picks.json"), LEAGUES["BL1"])
    assert len(opinions) == 3  # le pronostic « Total Goals » est ignoré
    gladbach = opinions[0]
    # « TSG Hoffenheim @ Borussia Monchengladbach » : Gladbach reçoit (vérifié sur le calendrier OpenLigaDB)
    assert (gladbach["home"], gladbach["away"], gladbach["outcome"]) == ("Borussia Monchengladbach", "TSG Hoffenheim", "Home")
    assert gladbach["probability"] == pytest.approx(0.304) and gladbach["utc_date"] == "2026-10-18T15:30:00Z"
    assert opinions[2]["outcome"] == "Draw"


def test_betbetter_error_field():
    with pytest.raises(SourceError):
        betbetter.parse_picks({"picks": [], "error": "model offline"}, LEAGUES["PL"])


def test_guardian_documented_format():
    news = guardian.parse_search(load_sample("guardian_search_doc_example.json"))
    assert news == [{
        "headline": "Russia-Ukraine war latest: what we know on day 240 of the invasion",
        "url": "https://www.theguardian.com/world/2022/oct/21/russia-ukraine-war-latest-what-we-know-on-day-240-of-the-invasion",
        "published_at": "2022-10-21T14:06:14Z",
        "source": "The Guardian",
    }]
    with pytest.raises(SourceError):
        guardian.parse_search({"response": {"status": "error", "message": "Invalid authentication credentials", "results": []}})


def test_guardian_needs_key_and_filters_last_days(monkeypatch):
    from euria import config as cfg

    seen = {}

    def fake_get_json(source, url, params=None, headers=None):
        seen.update(params)
        return {"response": {"status": "ok", "results": []}}

    monkeypatch.setattr(guardian, "get_json", fake_get_json)
    monkeypatch.setattr(cfg, "GUARDIAN_API_KEY", None)
    with pytest.raises(SourceError):
        guardian.search_injuries("Arsenal")
    monkeypatch.setattr(cfg, "GUARDIAN_API_KEY", "g_test")
    assert guardian.search_injuries("Arsenal") == []
    assert seen["api-key"] == "g_test" and seen["section"] == "football" and seen["q"] == "Arsenal injury"
    assert seen["from-date"] == (datetime.now(timezone.utc) - timedelta(days=cfg.NEWS_MAX_AGE_DAYS)).date().isoformat()

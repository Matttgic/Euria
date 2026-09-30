"""Bot et API de bout en bout, avec des sources simulées (aucun appel réseau)."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from euria import betting, bot, config, schema, services, telegram
from euria.fallback import Result

NOW = datetime.now(timezone.utc)


def _finished(days_ago, home, away, hg, ag):
    return schema.match(league="PL", season=config.season_for(), utc_date=schema.iso_utc(NOW - timedelta(days=days_ago)),
                        home=home, away=away, status="FINISHED", home_goals=hg, away_goals=ag)


SEASON = [
    _finished(40 - i * 7, h, a, hg, ag)
    for i, (h, a, hg, ag) in enumerate([
        ("Arsenal", "Leeds", 3, 0), ("Leeds", "Hull", 0, 0), ("Hull", "Arsenal", 0, 2), ("Arsenal", "Hull", 4, 0),
        ("Hull", "Leeds", 1, 1), ("Leeds", "Arsenal", 0, 1),
    ])
] + [_finished(1, "Arsenal", "Leeds", 2, 1), _finished(2, "Leeds", "Hull", 0, 0), _finished(3, "Hull", "Arsenal", 1, 1),
     _finished(4, "Hull", "Leeds", 2, 2)]
KICKOFF = schema.iso_utc(NOW + timedelta(days=2))
QUOTES = [
    schema.odds(league="PL", utc_date=KICKOFF, home="Arsenal", away="Leeds United", bookmaker="pmu", home_odds=6.0, draw_odds=6.0, away_odds=6.0),
    schema.odds(league="PL", utc_date=KICKOFF, home="Arsenal", away="Leeds United", bookmaker="pinnacle", home_odds=1.4, draw_odds=4.5, away_odds=8.0),
]
XPOINTS = [
    schema.xpoints(team="Arsenal", played=5, points=12, expected_points=10.49, form="WWWWL"),
    schema.xpoints(team="Leeds United", played=5, points=7, expected_points=8.2, form="DWLDW"),
]
OPINIONS = [schema.opinion(league="PL", utc_date=KICKOFF, home="Arsenal", away="Leeds United", outcome="Home", probability=0.7, fair_odds=1.43)]
NEWS = {"Arsenal": [schema.news(headline="Havertz_back *soon*", url="https://example.invalid/a", published_at=schema.iso_utc(NOW), source="The Guardian")]}
WEATHER = {**schema.weather(latitude=51.55, longitude=-0.1, time=KICKOFF, temperature_c=12.0, precipitation_mm=0.4, wind_speed_ms=5.0), "stadium": "Emirates Stadium"}


@pytest.fixture
def fake_sources(monkeypatch):
    def matches(code, season=None):
        data = SEASON if code == "PL" else []
        return Result(data, "football-data.co.uk", "2026-09-30T08:00:00+00:00", attribution="Résultats : football-data.co.uk")

    def odds(code):
        if code == "PL":
            return Result(QUOTES, "Parlay API", "2026-09-30T08:00:00+00:00", attribution="Cotes : Parlay API")
        return Result(None, None, None, stale=True, message="Donnée indisponible : toutes les sources sont en panne.")

    monkeypatch.setattr(services, "matches", matches)
    monkeypatch.setattr(services, "odds", odds)
    monkeypatch.setattr(services, "weather", lambda home, kickoff: Result(WEATHER, "MET Norway", "2026-09-30T08:00:00+00:00", attribution="Données météo : MET Norway (CC BY 4.0)"))
    monkeypatch.setattr(services, "xpoints", lambda code: Result(XPOINTS, "Football Charts", "2026-09-30T08:00:00+00:00", attribution="Data by football-charts.com"))
    monkeypatch.setattr(services, "second_opinion", lambda code: Result(OPINIONS, "Bet Better", "2026-09-30T08:00:00+00:00", attribution="Bet Better — https://betbetter.world"))
    monkeypatch.setattr(services, "news", lambda team: Result(NEWS.get(team, []), "The Guardian", "2026-09-30T08:00:00+00:00", attribution="Actus : The Guardian"))
    sent: list[str] = []
    monkeypatch.setattr(telegram, "send", lambda msg: sent.append(msg) or True)
    monkeypatch.setattr(config, "MIN_MATCHES_PLAYED", 3)
    monkeypatch.setattr(config, "ODDS_BOOKMAKERS", ["pmu", "pinnacle"])
    return sent


def test_bot_settles_bets_and_sends_value_bet(fake_sources):
    betting.save([
        {"Match": "Arsenal vs Leeds", "Pari": "Home", "Cote": "1.5", "Value": "1.2",
         "Date": (NOW - timedelta(days=2)).strftime("%Y-%m-%d %H:%M"), "Result": "", "Ligue": "", "Bookmaker": "", "CoupEnvoi": ""},
    ])
    summary = bot.run()
    rows = betting.load()
    assert rows[0]["Result"] == "Win" and rows[0]["Ligue"] == "PL"  # 2-1 hier
    assert summary["settled"] == 1 and summary["alerts"] == 1
    new = rows[1]
    assert (new["Match"], new["Bookmaker"], new["Ligue"], new["CoupEnvoi"]) == ("Arsenal vs Leeds United", "pmu", "PL", KICKOFF)
    assert any("NOUVELLES OPPORTUNITÉS" in m and "Cotes : Parlay API" in m for m in fake_sources)
    assert any("Sources en difficulté" in m for m in fake_sources)  # les 4 autres championnats sans cotes
    alert = next(m for m in fake_sources if "NOUVELLES OPPORTUNITÉS" in m)
    assert "Points réels − attendus : Arsenal +1.5 · Leeds United −1.2" in alert
    assert "Bet Better : Domicile 70 %" in alert
    assert "Havertz\\_back \\*soon\\*" in alert  # titre extérieur échappé pour le Markdown de Telegram
    assert "Data by football-charts.com" in alert and "Bet Better — https://betbetter.world" in alert
    assert config.PREDICTIONS_FILE.exists()
    log = config.PREDICTIONS_FILE.read_text(encoding="utf-8")
    assert "home_gf" in log and "6.0" not in log.split("\n")[1]  # pas de cotes brutes dans le journal
    assert "home_luck" in log and ",1.51," in log.split("\n")[1]
    # Deuxième exécution : pas de doublon
    assert bot.run()["alerts"] == 0


def test_api_endpoints(fake_sources):
    from euria.api import app

    client = TestClient(app)
    assert client.get("/").json()["status"] == "ok"
    body = client.get("/matches/PL", params={"status": "finished"}).json()
    assert body["meta"]["source"] == "football-data.co.uk" and body["meta"]["updated_at"]
    standings = client.get("/standings/pl").json()
    assert standings["data"][0]["team"] == "Arsenal"
    assert client.get("/odds/SA").json()["data"] is None  # panne : message clair, pas d'erreur 500
    assert client.get("/matches/XX").status_code == 404

    pred = client.post("/predict", json={"league": "PL", "home": "Arsenal", "away": "Leeds United"}).json()
    report = pred["data"]
    assert report["kickoff"].startswith(KICKOFF[:13])
    assert set(report["probabilities"]) == {"Home", "Draw", "Away"}
    assert report["odds"]["selected"]["bookmaker"] == "pmu"
    assert report["weather"]["stadium"] == "Emirates Stadium"
    assert "Données météo : MET Norway (CC BY 4.0)" in pred["meta"]["attribution"]
    assert report["sources"]["matches"]["updated_at"]
    assert report["xpoints"]["home"]["luck"] == 1.51 and report["second_opinion"]["outcome"] == "Home"
    assert report["news"]["home"][0]["source"] == "The Guardian" and report["news"]["away"] == []
    assert "Data by football-charts.com" in pred["meta"]["attribution"]


def test_enrichment_endpoints(fake_sources):
    from euria.api import app

    client = TestClient(app)
    assert client.get("/xpoints/PL").json()["data"][0]["team"] == "Arsenal"
    assert client.get("/second-opinion/pl").json()["meta"]["source"] == "Bet Better"
    assert client.get("/news", params={"team": "Arsenal"}).json()["data"][0]["headline"].startswith("Havertz")


def test_weather_endpoint(fake_sources):
    from euria.api import app

    body = TestClient(app).get("/weather", params={"home": "Arsenal", "kickoff": KICKOFF}).json()
    assert body["data"]["temperature_c"] == 12.0 and body["meta"]["attribution"].startswith("Données météo")


def test_prediction_log_rotates_when_columns_change(fake_sources):
    config.PREDICTIONS_FILE.write_text("ancienne,entete\n1,2\n", encoding="utf-8")
    bot.run()
    archived = list(config.PREDICTIONS_FILE.parent.glob("predictions_*.csv"))
    assert len(archived) == 1 and archived[0].read_text(encoding="utf-8").startswith("ancienne")
    assert config.PREDICTIONS_FILE.read_text(encoding="utf-8").startswith("run_at,")


def test_weather_beyond_forecast_horizon_is_explained():
    result = services.weather("Arsenal", NOW + timedelta(days=20))  # pas de fixture : vrai service, sans réseau
    assert result.data is None and "plus de 16 jours" in result.message and result.errors == []

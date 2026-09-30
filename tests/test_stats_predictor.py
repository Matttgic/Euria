from datetime import datetime, timedelta, timezone

import pytest

from euria import config, predictor, schema, stats

START = datetime(2026, 8, 15, 15, tzinfo=timezone.utc)


def _m(day, home, away, hg, ag):
    return schema.match(league="PL", season=2026, utc_date=schema.iso_utc(START + timedelta(days=day)),
                        home=home, away=away, status="FINISHED", home_goals=hg, away_goals=ag)


# Strong gagne tout 3-0, Weak perd tout 0-2 ; Mid fait nul.
MATCHES = [
    _m(0, "Strong", "Mid", 3, 0), _m(1, "Weak", "Mid", 0, 2), _m(7, "Mid", "Strong", 0, 3), _m(8, "Mid", "Weak", 2, 0),
    _m(14, "Strong", "Weak", 3, 0), _m(21, "Weak", "Strong", 0, 3), _m(28, "Strong", "Mid", 3, 0), _m(29, "Weak", "Mid", 0, 2),
    _m(35, "Mid", "Strong", 0, 3), _m(36, "Mid", "Weak", 2, 0), _m(42, "Mid", "Mid B", 1, 1),
    schema.match(league="PL", season=2026, utc_date="2026-11-01T15:00:00Z", home="Strong", away="Weak", status="SCHEDULED"),
]


def test_team_profile_matches_model_definition():
    profile = stats.team_profile(stats.finished_before(MATCHES), "Strong")
    assert profile == {"played": 6, "gf": 3.0, "ga": 0.0, "cs": 6, "fw": 5, "fd": 0, "fl": 0}


def test_features_ignore_matches_after_kickoff_and_resolve_names(monkeypatch):
    monkeypatch.setattr(config, "MIN_MATCHES_PLAYED", 4)
    kickoff = START + timedelta(days=22)  # seuls les matchs des jours 0 à 21 comptent
    feats = stats.features(MATCHES, "Strong FC", "Weak", before=kickoff)
    assert feats["home_team"] == "Strong" and feats["home"]["played"] == 4
    assert len(feats["vector"]) == len(stats.FEATURE_NAMES) == 12


def test_features_none_when_too_few_matches(monkeypatch):
    monkeypatch.setattr(config, "MIN_MATCHES_PLAYED", 50)
    assert stats.features(MATCHES, "Strong", "Weak", before=START + timedelta(days=60)) is None
    assert stats.features(MATCHES, "Inconnu", "Weak", before=START + timedelta(days=60)) is None


def test_standings():
    table = stats.standings(MATCHES)
    assert [r["team"] for r in table[:2]] == ["Strong", "Mid"]
    assert table[0] == {**table[0], "points": 18, "won": 6, "goal_diff": 18, "rank": 1}


def test_model_class_order_home_draw_away():
    strong = [3.0, 0.3, 6, 5, 0, 0]
    weak = [0.3, 3.0, 0, 0, 0, 5]
    home_fav = predictor.predict(strong + weak)
    away_fav = predictor.predict(weak + strong)
    assert set(home_fav) == {"Home", "Draw", "Away"} and sum(home_fav.values()) == pytest.approx(1, abs=1e-3)
    assert max(home_fav, key=home_fav.get) == "Home"
    assert max(away_fav, key=away_fav.get) == "Away"


def test_choose_quote_follows_preference(monkeypatch):
    monkeypatch.setattr(config, "ODDS_BOOKMAKERS", ["pmu", "pinnacle"])
    quotes = [{"bookmaker": "pinnacle"}, {"bookmaker": "pmu"}]
    assert predictor.choose_quote(quotes)["bookmaker"] == "pmu"
    assert predictor.choose_quote([{"bookmaker": "bet365"}])["bookmaker"] == "bet365"
    assert predictor.choose_quote([]) is None


def test_value_bets():
    quote = {"home_odds": 2.0, "draw_odds": 3.0, "away_odds": 5.0}
    bets = predictor.value_bets({"Home": 0.6, "Draw": 0.25, "Away": 0.15}, quote, threshold=1.1)
    assert bets == [{"outcome": "Home", "probability": 0.6, "odds": 2.0, "value": 1.2}]

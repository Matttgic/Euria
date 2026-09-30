import pytest

from euria import config
from euria.config import LEAGUES
from euria.http import get_json
from euria.sources import parlay
from tests.live.conftest import require_key

pytestmark = pytest.mark.live


def test_demo_fields_still_exist():
    """Démo sans clé (Premier League seulement, 5 matchs, 60 appels/heure)."""
    payload = get_json("Parlay API", f"{parlay.BASE_URL}/try/soccer_epl/odds")
    events = payload["events"]
    if not events:
        pytest.skip("aucun match de Premier League à venir dans la démo")
    event = events[0]
    for field in ("home_team", "away_team", "commence_time", "bookmakers"):
        assert field in event, field
    market = event["bookmakers"][0]["markets"][0]
    assert market["key"] == "h2h" and {"name", "price"} <= set(market["outcomes"][0])
    assert parlay.parse_odds(payload, LEAGUES["PL"])


def test_all_leagues_with_key():
    require_key(config.PARLAY_API_KEY, "PARLAY_API_KEY")
    for league in LEAGUES.values():  # 5 crédits
        quotes = parlay.fetch_odds(league)
        assert isinstance(quotes, list)

import pytest

from euria import config
from euria.config import LEAGUES
from euria.http import get_json
from euria.sources import api_football
from tests.live.conftest import require_key

pytestmark = pytest.mark.live


def test_fixture_and_odds_fields_still_exist():
    """Appelle les mêmes adresses que la production (environ 2 appels sur les 100 quotidiens)."""
    require_key(config.API_FOOTBALL_KEY, "API_FOOTBALL_KEY")
    fixtures = api_football.fetch_fixtures(LEAGUES["PL"])  # lève SourceError si l'API refuse la requête
    if not fixtures:
        pytest.skip("aucun match de Premier League dans les prochains jours")
    f = fixtures[0]
    assert f["fixture"]["id"] and f["fixture"]["date"] and f["teams"]["home"]["name"] and f["teams"]["away"]["name"]
    odds = get_json("API-Football", f"{api_football.BASE_URL}/odds",
                    params={"fixture": f["fixture"]["id"], "bookmaker": api_football.BET365, "bet": api_football.MATCH_WINNER},
                    headers={"x-apisports-key": config.API_FOOTBALL_KEY})
    rows = api_football._response(odds)
    if rows:
        values = rows[0]["bookmakers"][0]["bets"][0]["values"]
        assert {v["value"] for v in values} >= {"Home", "Draw", "Away"}

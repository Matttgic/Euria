import pytest

from euria import config
from euria.http import get_json
from euria.sources import api_football
from tests.live.conftest import require_key

pytestmark = pytest.mark.live


def test_fixture_and_odds_fields_still_exist():
    """Vérifie les champs repris de l'ancien code (environ 2 appels sur les 100 quotidiens)."""
    require_key(config.API_FOOTBALL_KEY, "API_FOOTBALL_KEY")
    headers = {"x-apisports-key": config.API_FOOTBALL_KEY}
    fixtures = get_json("API-Football", f"{api_football.BASE_URL}/fixtures", params={"league": 39, "next": 10}, headers=headers)
    assert not fixtures.get("errors"), fixtures.get("errors")
    f = fixtures["response"][0]
    assert f["fixture"]["id"] and f["fixture"]["date"] and f["teams"]["home"]["name"]
    odds = get_json("API-Football", f"{api_football.BASE_URL}/odds",
                    params={"fixture": f["fixture"]["id"], "bookmaker": api_football.BET365, "bet": api_football.MATCH_WINNER}, headers=headers)
    if odds["response"]:
        values = odds["response"][0]["bookmakers"][0]["bets"][0]["values"]
        assert {v["value"] for v in values} >= {"Home", "Draw", "Away"}

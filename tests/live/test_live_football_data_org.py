import pytest

from euria import config
from euria.config import LEAGUES
from euria.http import get_json
from euria.sources import football_data_org
from tests.live.conftest import current_season, require_key

pytestmark = pytest.mark.live


def test_matches_fields_still_exist():
    require_key(config.FOOTBALL_DATA_TOKEN, "FOOTBALL_DATA_TOKEN")
    payload = get_json("football-data.org", f"{football_data_org.BASE_URL}/competitions/PL/matches",
                       params={"season": current_season()}, headers={"X-Auth-Token": config.FOOTBALL_DATA_TOKEN})
    match = payload["matches"][0]
    for field in ("utcDate", "status", "homeTeam", "awayTeam", "score"):
        assert field in match, field
    assert "shortName" in match["homeTeam"] or "name" in match["homeTeam"]
    assert "home" in match["score"]["fullTime"]
    assert football_data_org.fetch_matches(LEAGUES["PL"], current_season())

import pytest

from euria.config import LEAGUES
from euria.http import get_json
from euria.sources import openligadb
from tests.live.conftest import current_season

pytestmark = pytest.mark.live


def test_bundesliga_fields_still_exist():
    payload = get_json("OpenLigaDB", f"{openligadb.BASE_URL}/getmatchdata/bl1/{current_season()}")
    m = payload[0]
    for field in ("matchDateTimeUTC", "team1", "team2", "matchIsFinished", "matchResults"):
        assert field in m, field
    assert "teamName" in m["team1"]
    finished = [x for x in payload if x["matchIsFinished"]]
    if finished:
        assert any(r["resultTypeID"] == openligadb.FINAL_RESULT_TYPE for r in finished[0]["matchResults"])
    assert openligadb.fetch_matches(LEAGUES["BL1"], current_season())

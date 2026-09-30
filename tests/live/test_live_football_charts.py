import pytest

from euria.config import LEAGUES
from euria.http import get_json
from euria.sources import football_charts

pytestmark = pytest.mark.live


def test_table_fields_still_exist():
    payload = get_json("Football Charts", f"{football_charts.BASE_URL}/premier/table/")
    row = payload["table"][0]
    for field in ("team", "played", "points", "expected_points", "last_5_form"):
        assert field in row, field


@pytest.mark.parametrize("code", list(LEAGUES))
def test_every_league_has_expected_points(code):
    rows = football_charts.fetch_table(LEAGUES[code])
    assert len(rows) >= 16 and all(isinstance(r["expected_points"], float) for r in rows)

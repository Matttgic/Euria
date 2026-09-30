import pytest

from euria.config import LEAGUES
from euria.sources import football_data_couk
from tests.live.conftest import current_season

pytestmark = pytest.mark.live


@pytest.mark.parametrize("code", list(LEAGUES))
def test_csv_columns_still_exist(code):
    matches = football_data_couk.fetch_matches(LEAGUES[code], current_season())  # vérifie REQUIRED_COLUMNS
    assert matches and matches[0]["home_goals"] is not None

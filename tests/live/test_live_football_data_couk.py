import pytest

from euria.config import LEAGUES
from euria.sources import football_data_couk
from tests.live.conftest import current_season

pytestmark = pytest.mark.live


@pytest.mark.parametrize("code", list(LEAGUES))
def test_csv_columns_still_exist(code):
    matches = football_data_couk.fetch_matches(LEAGUES[code], current_season())  # vérifie REQUIRED_COLUMNS
    assert matches and matches[0]["home_goals"] is not None


@pytest.mark.parametrize("code", list(LEAGUES))
def test_closing_odds_still_published(code):
    # Saison précédente complète : au moins Pinnacle ou la moyenne de clôture sur chaque match joué.
    closing = football_data_couk.fetch_closing_odds(LEAGUES[code], current_season() - 1)
    assert len(closing) >= 250 and all(len(m["odds"]) == 3 for m in closing)

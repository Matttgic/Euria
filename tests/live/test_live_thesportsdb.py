import pytest

from euria.sources import thesportsdb

pytestmark = pytest.mark.live


def test_team_search_fields_still_exist():
    team = thesportsdb.fetch_team("Chelsea")
    assert team["stadium"] and "location" in team

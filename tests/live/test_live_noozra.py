import pytest

from euria.http import get_json
from euria.sources import noozra

pytestmark = pytest.mark.live


def test_search_fields_still_exist():
    """1 appel sur les 100 quotidiens sans clé."""
    payload = get_json("Noozra", noozra.URL, params={"q": "injury", "category": "sports", "limit": 3})
    assert payload["articles"], "aucun titre sportif contenant « injury »"
    for field in ("headline", "url", "published_at", "source"):
        assert field in payload["articles"][0], field

import pytest

from euria import config
from euria.sources import guardian
from tests.live.conftest import require_key

pytestmark = pytest.mark.live


def test_search_fields_still_exist():
    """1 appel sur les 500 quotidiens. Vérifie aussi que l'API répond depuis GitHub Actions."""
    require_key(config.GUARDIAN_API_KEY, "GUARDIAN_API_KEY")
    news = guardian.search_injuries("Arsenal")  # lève SourceError si le format ou l'accès change
    assert isinstance(news, list)
    for item in news:
        assert item["headline"] and item["url"].startswith("https://") and item["published_at"]

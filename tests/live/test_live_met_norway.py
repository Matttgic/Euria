from datetime import datetime, timedelta, timezone

import pytest

from euria.sources import met_norway

pytestmark = pytest.mark.live


def test_forecast_fields_still_exist():
    kickoff = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(hours=3)
    weather = met_norway.fetch_weather(48.8414, 2.2530, kickoff)  # Parc des Princes
    assert weather["temperature_c"] is not None and weather["wind_speed_ms"] is not None
    assert weather["precipitation_mm"] is not None

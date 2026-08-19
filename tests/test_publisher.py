"""Publisher round-trips a payload through fakeredis."""

import fakeredis

from microgrid_forecaster.config import RedisCfg
from microgrid_forecaster.service.publisher import RedisPublisher
from microgrid_forecaster.service.schemas import ForecastPayload


def test_publish_sets_latest(monkeypatch):
    fake = fakeredis.FakeStrictRedis()
    monkeypatch.setattr(
        "microgrid_forecaster.service.publisher.redis.from_url", lambda url: fake
    )
    pub = RedisPublisher(RedisCfg(enabled=True))
    payload = ForecastPayload(
        issued_at="2026-06-30T13:00:00",
        horizon_h=24,
        model_version="v1",
        forecast={"demand": [1.0] * 24},
    )
    pub.publish(payload)
    stored = fake.get("forecaster:load:latest")
    assert stored is not None
    assert b"demand" in stored

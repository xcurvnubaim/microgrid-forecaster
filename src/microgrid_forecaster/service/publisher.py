"""Publish forecast payloads to Redis (the diagram's `forecaster:load` block)."""

from __future__ import annotations

import redis

from ..config import RedisCfg
from .schemas import ForecastPayload


class RedisPublisher:
    def __init__(self, cfg: RedisCfg):
        self.cfg = cfg
        self.client = redis.from_url(cfg.url)

    def publish(self, payload: ForecastPayload) -> None:
        body = payload.model_dump_json()
        # publish (pub/sub) + set last value (so late subscribers can GET it)
        self.client.publish(self.cfg.channel, body)
        self.client.set(f"{self.cfg.channel}:latest", body)

    def ping(self) -> bool:
        return bool(self.client.ping())

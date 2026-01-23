from __future__ import annotations

from redis import Redis

from .settings import ServiceSettings


def get_redis(settings: ServiceSettings) -> Redis:
    # decode_responses=False keeps bytes, which is fine for rq internals.
    return Redis.from_url(settings.redis_url)

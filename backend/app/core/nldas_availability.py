from __future__ import annotations

import copy
import os
import time
from threading import Lock
from typing import Any


_CACHE_TTL_SECONDS = int(os.environ.get("NLDAS_LATEST_DAY_CACHE_SECONDS", "900"))
_cache: dict[tuple[str | None, int], tuple[float, dict[str, Any]]] = {}
_lock = Lock()


def get_latest_nldas_availability(
    *,
    short_name: str | None = None,
    max_lookback_days: int = 21,
    force_refresh: bool = False,
) -> dict[str, Any]:
    key = (short_name, max_lookback_days)
    now = time.time()

    with _lock:
        cached = _cache.get(key)
        if (
            not force_refresh
            and cached is not None
            and now - cached[0] <= _CACHE_TTL_SECONDS
        ):
            return copy.deepcopy(cached[1])

    from app.adapt.data_earthaccess import find_latest_complete_nldas_day

    availability = find_latest_complete_nldas_day(
        short_name=short_name,
        max_lookback_days=max_lookback_days,
    )

    with _lock:
        _cache[key] = (time.time(), copy.deepcopy(availability))

    return availability

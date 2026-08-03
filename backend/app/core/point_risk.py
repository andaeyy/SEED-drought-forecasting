from __future__ import annotations

from typing import Any

import numpy as np

from app.core.map_features import CATEGORY_LABELS, RISK_LABELS, normalize_longitude
from app.core.schemas import PointRiskResponse


def drought_risk_at_coordinate(
    *,
    lat: float,
    lon: float,
    lat2d: Any,
    lon2d: Any,
    pdry: Any,
    category: Any,
    et_pred: Any | None = None,
    sm_pred: Any | None = None,
) -> PointRiskResponse:
    latg = np.asarray(lat2d, dtype=np.float32)
    long = np.asarray(lon2d, dtype=np.float32)
    dry = np.asarray(pdry, dtype=np.float32)
    cat = np.asarray(category, dtype=np.int8)

    if latg.shape != long.shape or latg.shape != dry.shape or latg.shape != cat.shape:
        raise ValueError("lat2d, lon2d, pdry, and category must have matching shapes")

    lon_user = float(normalize_longitude(float(lon)))
    lon_grid = normalize_longitude(long)

    dlat = latg - float(lat)
    dlon = np.abs(lon_grid - lon_user)
    dlon = np.minimum(dlon, 360.0 - dlon)
    d2 = dlat * dlat + dlon * dlon

    if not np.any(np.isfinite(d2)):
        raise ValueError("forecast grid does not contain finite coordinates")

    row, col = np.unravel_index(np.nanargmin(d2), d2.shape)
    category_value = int(cat[row, col])
    category_key = category_value if category_value in CATEGORY_LABELS else 0
    pdry_value = float(dry[row, col])
    et_value = None if et_pred is None else float(np.asarray(et_pred, dtype=np.float32)[row, col])
    sm_value = None if sm_pred is None else float(np.asarray(sm_pred, dtype=np.float32)[row, col])

    return PointRiskResponse(
        requested_lat=float(lat),
        requested_lon=lon_user,
        grid_lat=float(latg[row, col]),
        grid_lon=float(lon_grid[row, col]),
        grid_row=int(row),
        grid_col=int(col),
        pdry=pdry_value,
        pdry_pct=100.0 * pdry_value,
        category=category_value,
        category_label=CATEGORY_LABELS[category_key],
        risk_label=RISK_LABELS[category_key],
        et_mm_per_day=et_value,
        sm_m3_per_m3=sm_value,
    )

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


CATEGORY_LABELS: dict[int, str] = {
    0: "None",
    1: "D0",
    2: "D1",
    3: "D2",
    4: "D3",
    5: "D4",
}

RISK_LABELS: dict[int, str] = {
    0: "Normal",
    1: "Abnormally Dry",
    2: "Moderate Drought",
    3: "Severe Drought",
    4: "Extreme Drought",
    5: "Exceptional Drought",
}


def normalize_longitude(lon: Any) -> np.ndarray:
    lon_arr = np.asarray(lon, dtype=np.float64)
    normalized = ((lon_arr + 180.0) % 360.0) - 180.0
    return np.where((normalized == -180.0) & (lon_arr > 0.0), 180.0, normalized)


def forecast_result_to_geojson(
    result: Any,
    *,
    prefer_polygons: bool = True,
) -> dict[str, Any]:
    collection = forecast_arrays_to_geojson(
        lat2d=_result_value(result, "lat2d"),
        lon2d=_result_value(result, "lon2d"),
        pdry=_result_value(result, "pdry"),
        category=_result_value(result, "category"),
        prefer_polygons=prefer_polygons,
    )

    target_day = _result_value(result, "target_day", default=None)
    horizon_days = _result_value(result, "horizon_days", default=None)
    if target_day is not None:
        collection["properties"]["target_day"] = str(target_day)
    if horizon_days is not None:
        collection["properties"]["horizon_days"] = int(horizon_days)

    return collection


def forecast_result_scalar_to_geojson(
    result: Any,
    *,
    layer: str,
    prefer_polygons: bool = True,
) -> dict[str, Any]:
    layer_key = layer.strip().lower()
    layer_specs = {
        "et": ("et_pred", "Evapotranspiration", "mm/day"),
        "sm": ("sm_pred", "Soil moisture", "m3/m3"),
    }
    if layer_key not in layer_specs:
        raise ValueError(f"Unsupported scalar forecast layer: {layer!r}")

    value_key, label, units = layer_specs[layer_key]
    collection = forecast_scalar_arrays_to_geojson(
        lat2d=_result_value(result, "lat2d"),
        lon2d=_result_value(result, "lon2d"),
        values=_result_value(result, value_key),
        layer=layer_key,
        label=label,
        units=units,
        prefer_polygons=prefer_polygons,
    )

    target_day = _result_value(result, "target_day", default=None)
    horizon_days = _result_value(result, "horizon_days", default=None)
    if target_day is not None:
        collection["properties"]["target_day"] = str(target_day)
    if horizon_days is not None:
        collection["properties"]["horizon_days"] = int(horizon_days)

    return collection


def _result_value(result: Any, key: str, default: Any = ...) -> Any:
    if isinstance(result, Mapping):
        if default is ...:
            return result[key]
        return result.get(key, default)
    if default is ...:
        return getattr(result, key)
    return getattr(result, key, default)


def forecast_arrays_to_geojson(
    *,
    lat2d: Any,
    lon2d: Any,
    pdry: Any,
    category: Any,
    prefer_polygons: bool = True,
) -> dict[str, Any]:
    """emits regular grids as polygons and falls back to points"""
    lat, lon, dry, cat = _coerce_grid_arrays(lat2d, lon2d, pdry, category)
    lon = normalize_longitude(lon)

    valid = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & np.isfinite(dry)
        & np.isfinite(cat)
        & (cat >= 0)
        & (cat <= 5)
    )
    polygon_edges = _safe_regular_grid_edges(lat, lon) if prefer_polygons else None
    geometry_mode = "Polygon" if polygon_edges is not None else "Point"

    features: list[dict[str, Any]] = []
    coordinate_bounds: list[tuple[float, float]] = []

    if polygon_edges is not None:
        lat_edges, lon_edges = polygon_edges
        for i, j in np.argwhere(valid):
            polygon = _cell_polygon(lat_edges, lon_edges, int(i), int(j))
            features.append(
                _feature(
                    geometry={"type": "Polygon", "coordinates": [polygon]},
                    row=int(i),
                    col=int(j),
                    pdry_value=float(dry[i, j]),
                    category_value=int(cat[i, j]),
                )
            )
            coordinate_bounds.extend((float(x), float(y)) for x, y in polygon[:-1])
    else:
        for i, j in np.argwhere(valid):
            point = [float(lon[i, j]), float(lat[i, j])]
            features.append(
                _feature(
                    geometry={"type": "Point", "coordinates": point},
                    row=int(i),
                    col=int(j),
                    pdry_value=float(dry[i, j]),
                    category_value=int(cat[i, j]),
                )
            )
            coordinate_bounds.append((point[0], point[1]))

    bounds = _bounds_from_coordinates(coordinate_bounds)

    return {
        "type": "FeatureCollection",
        "bbox": _bbox_from_bounds(bounds),
        "bounds": bounds,
        "features": features,
        "properties": {
            "geometry_mode": geometry_mode,
            "feature_count": len(features),
            "category_labels": CATEGORY_LABELS,
            "risk_labels": RISK_LABELS,
        },
    }


def forecast_scalar_arrays_to_geojson(
    *,
    lat2d: Any,
    lon2d: Any,
    values: Any,
    layer: str,
    label: str,
    units: str,
    prefer_polygons: bool = True,
) -> dict[str, Any]:
    lat = np.asarray(lat2d, dtype=np.float64)
    lon = normalize_longitude(lon2d)
    value_arr = np.asarray(values, dtype=np.float64)

    if lat.ndim != 2:
        raise ValueError(f"lat2d must be a 2D array, got shape {lat.shape}")
    if lon.shape != lat.shape or value_arr.shape != lat.shape:
        raise ValueError(
            f"Scalar layer shapes must match lat2d {lat.shape}; "
            f"got lon2d {lon.shape} and values {value_arr.shape}"
        )

    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(value_arr)
    polygon_edges = _safe_regular_grid_edges(lat, lon) if prefer_polygons else None
    geometry_mode = "Polygon" if polygon_edges is not None else "Point"

    features: list[dict[str, Any]] = []
    coordinate_bounds: list[tuple[float, float]] = []
    if polygon_edges is not None:
        lat_edges, lon_edges = polygon_edges
        for i, j in np.argwhere(valid):
            polygon = _cell_polygon(lat_edges, lon_edges, int(i), int(j))
            features.append(
                _scalar_feature(
                    geometry={"type": "Polygon", "coordinates": [polygon]},
                    row=int(i),
                    col=int(j),
                    value=float(value_arr[i, j]),
                    layer=layer,
                    label=label,
                    units=units,
                )
            )
            coordinate_bounds.extend((float(x), float(y)) for x, y in polygon[:-1])
    else:
        for i, j in np.argwhere(valid):
            point = [float(lon[i, j]), float(lat[i, j])]
            features.append(
                _scalar_feature(
                    geometry={"type": "Point", "coordinates": point},
                    row=int(i),
                    col=int(j),
                    value=float(value_arr[i, j]),
                    layer=layer,
                    label=label,
                    units=units,
                )
            )
            coordinate_bounds.append((point[0], point[1]))

    bounds = _bounds_from_coordinates(coordinate_bounds)
    return {
        "type": "FeatureCollection",
        "bbox": _bbox_from_bounds(bounds),
        "bounds": bounds,
        "features": features,
        "properties": {
            "geometry_mode": geometry_mode,
            "feature_count": len(features),
            "layer": layer,
            "label": label,
            "units": units,
        },
    }


def forecast_to_geojson(result: Any, *, prefer_polygons: bool = True) -> dict[str, Any]:
    return forecast_result_to_geojson(result, prefer_polygons=prefer_polygons)


def build_forecast_geojson(result: Any, *, prefer_polygons: bool = True) -> dict[str, Any]:
    return forecast_result_to_geojson(result, prefer_polygons=prefer_polygons)


def _coerce_grid_arrays(
    lat2d: Any,
    lon2d: Any,
    pdry: Any,
    category: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat = np.asarray(lat2d, dtype=np.float64)
    lon = np.asarray(lon2d, dtype=np.float64)
    dry = np.asarray(pdry, dtype=np.float64)
    cat = np.asarray(category, dtype=np.float64)

    if lat.ndim != 2:
        raise ValueError(f"lat2d must be a 2D array, got shape {lat.shape}")

    shape = lat.shape
    for name, arr in (("lon2d", lon), ("pdry", dry), ("category", cat)):
        if arr.shape != shape:
            raise ValueError(f"{name} shape {arr.shape} does not match lat2d shape {shape}")

    return lat, lon, dry, cat


def _safe_regular_grid_edges(
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if min(lat.shape) < 2:
        return None
    if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
        return None

    row_lats = np.mean(lat, axis=1)
    col_lons = np.mean(lon, axis=0)
    lat_scale = max(1.0, float(np.nanmax(np.abs(row_lats))))
    lon_scale = max(1.0, float(np.nanmax(np.abs(col_lons))))
    if not np.allclose(lat, row_lats[:, None], rtol=0.0, atol=1e-6 * lat_scale):
        return None
    if not np.allclose(lon, col_lons[None, :], rtol=0.0, atol=1e-6 * lon_scale):
        return None
    if np.any(np.abs(np.diff(col_lons)) >= 180.0):
        return None

    lat_edges = _regular_axis_edges(row_lats)
    lon_edges = _regular_axis_edges(col_lons)
    if lat_edges is None or lon_edges is None:
        return None
    if float(np.min(lon_edges)) < -180.0 or float(np.max(lon_edges)) > 180.0:
        return None

    return lat_edges, lon_edges


def _regular_axis_edges(axis: np.ndarray) -> np.ndarray | None:
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)):
        return None

    diffs = np.diff(axis)
    if np.any(np.isclose(diffs, 0.0, rtol=0.0, atol=1e-12)):
        return None
    if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
        return None

    step = float(np.median(diffs))
    if not np.allclose(diffs, step, rtol=0.01, atol=max(abs(step) * 0.01, 1e-6)):
        return None

    edges = np.empty(axis.size + 1, dtype=np.float64)
    edges[1:-1] = (axis[:-1] + axis[1:]) / 2.0
    edges[0] = axis[0] - step / 2.0
    edges[-1] = axis[-1] + step / 2.0
    return edges


def _cell_polygon(
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
    row: int,
    col: int,
) -> list[list[float]]:
    south = float(min(lat_edges[row], lat_edges[row + 1]))
    north = float(max(lat_edges[row], lat_edges[row + 1]))
    west = float(min(lon_edges[col], lon_edges[col + 1]))
    east = float(max(lon_edges[col], lon_edges[col + 1]))
    return [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
    ]


def _feature(
    *,
    geometry: Mapping[str, Any],
    row: int,
    col: int,
    pdry_value: float,
    category_value: int,
) -> dict[str, Any]:
    category_key = category_value if category_value in CATEGORY_LABELS else 0
    return {
        "type": "Feature",
        "geometry": dict(geometry),
        "properties": {
            "row": row,
            "col": col,
            "pdry": pdry_value,
            "pdry_pct": pdry_value * 100.0,
            "category": category_value,
            "category_label": CATEGORY_LABELS[category_key],
            "risk_label": RISK_LABELS[category_key],
        },
    }


def _scalar_feature(
    *,
    geometry: Mapping[str, Any],
    row: int,
    col: int,
    value: float,
    layer: str,
    label: str,
    units: str,
) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": dict(geometry),
        "properties": {
            "row": row,
            "col": col,
            "layer": layer,
            "label": label,
            "value": value,
            "units": units,
        },
    }


def _bounds_from_coordinates(coordinates: list[tuple[float, float]]) -> dict[str, float] | None:
    if not coordinates:
        return None

    xs = np.asarray([coord[0] for coord in coordinates], dtype=np.float64)
    ys = np.asarray([coord[1] for coord in coordinates], dtype=np.float64)
    return {
        "west": float(np.min(xs)),
        "south": float(np.min(ys)),
        "east": float(np.max(xs)),
        "north": float(np.max(ys)),
    }


def _bbox_from_bounds(bounds: dict[str, float] | None) -> list[float] | None:
    if bounds is None:
        return None
    return [bounds["west"], bounds["south"], bounds["east"], bounds["north"]]


__all__ = [
    "CATEGORY_LABELS",
    "RISK_LABELS",
    "build_forecast_geojson",
    "forecast_arrays_to_geojson",
    "forecast_result_scalar_to_geojson",
    "forecast_result_to_geojson",
    "forecast_scalar_arrays_to_geojson",
    "forecast_to_geojson",
    "normalize_longitude",
]

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import xarray as xr


INPUT_VARS = ["PRECTmms", "TBOT", "WIND", "QBOT", "PSRF", "FSDS", "FLDS"]
DEFAULT_MODEL_GRID_PATH = os.path.join(os.path.dirname(__file__), "model_artifacts", "grid", "model_grid.npz")


@dataclass(frozen=True)
class LocalDailyData:
    day_keys: np.ndarray
    X: np.ndarray
    lat2d: np.ndarray
    lon2d: np.ndarray


def _to_day_key(time_values) -> np.ndarray:
    if np.issubdtype(time_values.dtype, np.datetime64):
        return time_values.astype("datetime64[D]")
    return np.array(
        [np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}") for t in time_values],
        dtype="datetime64[D]",
    )


def _norm_lon_180(lon) -> np.ndarray:
    lon = np.asarray(lon)
    return np.where(lon > 180, lon - 360, lon)


def load_precomputed_model_grid(path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(path)
    try:
        lat2d = arr["lat2d"].astype(np.float32)
        lon2d = arr["lon2d"].astype(np.float32)
    finally:
        arr.close()

    if lat2d.shape != lon2d.shape or lat2d.ndim != 2:
        raise ValueError(f"Invalid model grid in {path}: lat={lat2d.shape}, lon={lon2d.shape}")
    return lat2d, lon2d


def _xr_open_dataset_safe(path: str, **kwargs) -> xr.Dataset:
    last_err = None
    for eng in ("h5netcdf", "netcdf4", "scipy"):
        try:
            return xr.open_dataset(path, engine=eng, **kwargs)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to open {path} with xarray. Last error: {last_err}")


def _find_sample_forcing_file(forcing_dir: str, years: list[int]) -> str:
    for y in years:
        p = os.path.join(forcing_dir, f"clmforc.nldas.{int(y)}.nc")
        if os.path.exists(p):
            return p

    for y in range(1979, 2100):
        p = os.path.join(forcing_dir, f"clmforc.nldas.{y}.nc")
        if os.path.exists(p):
            return p

    raise FileNotFoundError(f"No clmforc.nldas.YYYY.nc found in {forcing_dir}")


def _elm_bounds(et_path: str, sm_path: str) -> Tuple[float, float, float, float]:
    errors: list[str] = []

    for label, path in (("ET", et_path), ("SM", sm_path)):
        ds = None
        try:
            ds = _xr_open_dataset_safe(path, decode_cf=True)
            lat = ds["lat"].values
            lon = ds["lon"].values
            if lat.ndim != 1 or lon.ndim != 1:
                errors.append(
                    f"{label} file has non-1D coordinates for bounds inference: "
                    f"lat.ndim={lat.ndim}, lon.ndim={lon.ndim}"
                )
                continue

            latmin, latmax = float(np.min(lat)), float(np.max(lat))
            lon = _norm_lon_180(lon)
            lonmin, lonmax = float(np.min(lon)), float(np.max(lon))
            return latmin, latmax, lonmin, lonmax
        except Exception as e:
            errors.append(f"{label} file {path}: {type(e).__name__}: {e}")
        finally:
            if ds is not None:
                ds.close()

    raise RuntimeError("Failed to infer ELM bounds. " + " | ".join(errors))


def _compute_nldas_crop_slices(sample_forcing_file: str, latmin: float, latmax: float, lonmin: float, lonmax: float, pad_cells: int = 1):
    ds = _xr_open_dataset_safe(sample_forcing_file, decode_cf=True)
    try:
        lat2d = ds["LATIXY"].values
        lon2d = ds["LONGXY"].values
    finally:
        ds.close()

    if lat2d.ndim == 3:
        lat2d = lat2d[0]
    if lon2d.ndim == 3:
        lon2d = lon2d[0]

    lon2d = _norm_lon_180(lon2d)

    mask = (lat2d >= latmin) & (lat2d <= latmax) & (lon2d >= lonmin) & (lon2d <= lonmax)
    ii, jj = np.where(mask)
    if ii.size == 0:
        raise RuntimeError("ELM bounds do not overlap NLDAS grid (check lon convention).")

    i0 = max(int(ii.min()) - pad_cells, 0)
    i1 = min(int(ii.max()) + 1 + pad_cells, lat2d.shape[0])
    j0 = max(int(jj.min()) - pad_cells, 0)
    j1 = min(int(jj.max()) + 1 + pad_cells, lat2d.shape[1])

    return slice(i0, i1), slice(j0, j1)


def get_model_grid_from_local_forcing(
    forcing_dir: str,
    et_path: str,
    sm_path: str,
    *,
    sample_year: Optional[int] = None,
    pad_cells: int = 1,
    model_grid_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    grid_path = model_grid_path or os.environ.get("MODEL_GRID_PATH") or DEFAULT_MODEL_GRID_PATH
    if grid_path and os.path.exists(grid_path):
        return load_precomputed_model_grid(grid_path)

    forcing_dir = os.path.abspath(forcing_dir)
    if sample_year is None:
        for y in range(1979, 2100):
            f = os.path.join(forcing_dir, f"clmforc.nldas.{y}.nc")
            if os.path.exists(f):
                sample_year = y
                break
    if sample_year is None:
        raise FileNotFoundError(f"No clmforc.nldas.YYYY.nc found in {forcing_dir}")

    sample_file = os.path.join(forcing_dir, f"clmforc.nldas.{sample_year}.nc")
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"Sample forcing file not found: {sample_file}")

    latmin, latmax, lonmin, lonmax = _elm_bounds(et_path, sm_path)
    lat_slice, lon_slice = _compute_nldas_crop_slices(sample_file, latmin, latmax, lonmin, lonmax, pad_cells=pad_cells)

    ds = _xr_open_dataset_safe(sample_file, decode_cf=True)
    try:
        lat2d = ds["LATIXY"].values
        lon2d = ds["LONGXY"].values
    finally:
        ds.close()

    if lat2d.ndim == 3:
        lat2d = lat2d[0]
    if lon2d.ndim == 3:
        lon2d = lon2d[0]

    lat2d = lat2d[lat_slice, lon_slice].astype(np.float32)
    lon2d = _norm_lon_180(lon2d)[lat_slice, lon_slice].astype(np.float32)
    return lat2d, lon2d


def load_local_daily_forcing(
    forcing_dir: str,
    years: list[int],
    et_path: str,
    sm_path: str,
) -> LocalDailyData:
    forcing_dir = os.path.abspath(forcing_dir)

    sample_file = _find_sample_forcing_file(forcing_dir, years)

    latmin, latmax, lonmin, lonmax = _elm_bounds(et_path, sm_path)
    lat_slice, lon_slice = _compute_nldas_crop_slices(sample_file, latmin, latmax, lonmin, lonmax, pad_cells=1)

    files = [os.path.join(forcing_dir, f"clmforc.nldas.{y}.nc") for y in years]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise FileNotFoundError(
            f"No forcing files found for requested years: {years}. "
            f"Directory checked: {forcing_dir}"
        )

    dsets = []
    ds = None
    try:
        for f in files:
            dsets.append(_xr_open_dataset_safe(f, decode_cf=True))

        ds = xr.concat(dsets, dim="time", data_vars="minimal", coords="minimal", compat="override").sortby("time")
        ds = ds.isel(lat=lat_slice, lon=lon_slice)

        lat2d = ds["LATIXY"].values
        lon2d = ds["LONGXY"].values
        if lat2d.ndim == 3:
            lat2d = lat2d[0]
        if lon2d.ndim == 3:
            lon2d = lon2d[0]
        lat2d = lat2d.astype(np.float32)
        lon2d = _norm_lon_180(lon2d).astype(np.float32)

        day_keys = _to_day_key(ds.time.values)
        _, keep = np.unique(day_keys, return_index=True)
        keep = np.sort(keep)
        ds = ds.isel(time=keep)
        day_keys = _to_day_key(ds.time.values)

        X = xr.concat([ds[v] for v in INPUT_VARS], dim="channel").transpose("time", "lat", "lon", "channel")
        X = X.astype("float32").compute().values

        return LocalDailyData(day_keys=day_keys, X=X, lat2d=lat2d, lon2d=lon2d)
    finally:
        if ds is not None:
            ds.close()
        for one in dsets:
            try:
                one.close()
            except Exception:
                pass

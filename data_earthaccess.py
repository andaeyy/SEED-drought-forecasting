from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr


# use FORA for FLDS; FORB falls back to zero
DEFAULT_NLDAS_SHORTNAME = os.environ.get("NLDAS_SHORTNAME", "NLDAS_FORA0125_H")

NEEDED_NLDAS_VARS = ["SWdown", "Rainf", "Tair", "Qair", "PSurf", "Wind_E", "Wind_N"]
DAILY_FORCING_CHANNELS = ["PRECT", "TBOT", "WIND", "QBOT", "PSRF", "FSDS", "FLDS"]

PROCESSED_DAILY_CACHE_VERSION = "daily_forcing_v1"
_RAW_HOURLY_FILENAME_RE = re.compile(r"A(\d{8})\.(\d{4})")
_RAW_HOURLY_DATA_EXTENSIONS = (".nc", ".nc4", ".h5", ".hdf", ".grb", ".grib", ".grib2")


def _default_nldas_cache_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "NLDAS_Cache"))


def _resolve_nldas_cache_dir(cache_dir: Optional[str]) -> str:
    return os.path.abspath(cache_dir or os.environ.get("NLDAS_CACHE_DIR", _default_nldas_cache_dir()))


def _parse_iso_to_utc_dt(s: str) -> datetime:
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _xr_open_dataset_safe(path: str, **kwargs) -> xr.Dataset:
    engines = [None, "netcdf4", "h5netcdf", "scipy"]
    last_err = None
    for eng in engines:
        try:
            if eng is None:
                return xr.open_dataset(path, **kwargs)
            return xr.open_dataset(path, engine=eng, **kwargs)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to open {path} with xarray. Last error: {last_err}")


def _open_many_hourly(paths: List[str]) -> xr.Dataset:
    dsets = []
    for p in paths:
        ds = _xr_open_dataset_safe(p, decode_cf=True)
        dsets.append(ds)

    ds_all = xr.concat(dsets, dim="time", data_vars="minimal", coords="minimal", compat="override")

    for ds in dsets:
        try:
            ds.close()
        except Exception:
            pass

    return ds_all


def _subset_bbox(ds: xr.Dataset, lat2d: np.ndarray, lon2d: np.ndarray, pad_deg: float = 0.5) -> xr.Dataset:
    lat_min = float(np.nanmin(lat2d)) - pad_deg
    lat_max = float(np.nanmax(lat2d)) + pad_deg
    lon_min = float(np.nanmin(lon2d)) - pad_deg
    lon_max = float(np.nanmax(lon2d)) + pad_deg

    # matches the source longitude convention
    ds_lon = ds["lon"].values
    if np.nanmax(ds_lon) <= 180.0 and lon_min > 180.0:
        lon_min -= 360.0
        lon_max -= 360.0
    elif np.nanmax(ds_lon) > 180.0 and lon_min < 0.0:
        lon_min += 360.0
        lon_max += 360.0

    return ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))


def _precompute_bilinear_weights(
    lat_src_1d: np.ndarray,
    lon_src_1d: np.ndarray,
    lat_tgt_2d: np.ndarray,
    lon_tgt_2d: np.ndarray,
) -> Dict[str, np.ndarray]:
    lat_src = np.asarray(lat_src_1d).astype(np.float64)
    lon_src = np.asarray(lon_src_1d).astype(np.float64)

    lat_flip = False
    lon_flip = False
    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        lat_flip = True
    if lon_src[0] > lon_src[-1]:
        lon_src = lon_src[::-1]
        lon_flip = True

    lat_t = np.asarray(lat_tgt_2d).astype(np.float64)
    lon_t = np.asarray(lon_tgt_2d).astype(np.float64)

    i1 = np.searchsorted(lat_src, lat_t, side="right")
    j1 = np.searchsorted(lon_src, lon_t, side="right")

    i1 = np.clip(i1, 1, lat_src.size - 1)
    j1 = np.clip(j1, 1, lon_src.size - 1)

    i0 = i1 - 1
    j0 = j1 - 1

    lat0 = lat_src[i0]
    lat1 = lat_src[i1]
    lon0 = lon_src[j0]
    lon1 = lon_src[j1]

    denom_lat = np.where((lat1 - lat0) == 0, 1.0, (lat1 - lat0))
    denom_lon = np.where((lon1 - lon0) == 0, 1.0, (lon1 - lon0))

    wy = (lat_t - lat0) / denom_lat
    wx = (lon_t - lon0) / denom_lon

    w00 = (1 - wy) * (1 - wx)
    w01 = (1 - wy) * wx
    w10 = wy * (1 - wx)
    w11 = wy * wx

    return {
        "i0": i0.astype(np.int64),
        "i1": i1.astype(np.int64),
        "j0": j0.astype(np.int64),
        "j1": j1.astype(np.int64),
        "w00": w00.astype(np.float32),
        "w01": w01.astype(np.float32),
        "w10": w10.astype(np.float32),
        "w11": w11.astype(np.float32),
        "lat_flip": np.array(lat_flip),
        "lon_flip": np.array(lon_flip),
    }


def _bilinear_regrid_3d(data_tyx: np.ndarray, w: Dict[str, np.ndarray]) -> np.ndarray:
    x = data_tyx
    if bool(w["lat_flip"]):
        x = x[:, ::-1, :]
    if bool(w["lon_flip"]):
        x = x[:, :, ::-1]

    i0, i1, j0, j1 = w["i0"], w["i1"], w["j0"], w["j1"]
    w00, w01, w10, w11 = w["w00"], w["w01"], w["w10"], w["w11"]

    v00 = x[:, i0, j0]
    v01 = x[:, i0, j1]
    v10 = x[:, i1, j0]
    v11 = x[:, i1, j1]

    return (v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11).astype(np.float32)


def _normalize_day(day) -> np.datetime64:
    if isinstance(day, np.datetime64):
        return day.astype("datetime64[D]")
    if isinstance(day, datetime):
        if day.tzinfo is not None:
            day = day.astimezone(timezone.utc)
        return np.datetime64(day.date().isoformat(), "D")
    if hasattr(day, "isoformat"):
        return np.datetime64(day.isoformat()[:10], "D")
    return np.datetime64(day, "D")


def _day_key_str(day: np.datetime64) -> str:
    return str(day.astype("datetime64[D]"))


def _daily_day_range(start_day, end_day) -> List[np.datetime64]:
    start = _normalize_day(start_day)
    end = _normalize_day(end_day)
    if end < start:
        raise ValueError(f"end_day {end} is before start_day {start}")
    n_days = int((end - start) / np.timedelta64(1, "D")) + 1
    return [start + np.timedelta64(i, "D") for i in range(n_days)]


def _contiguous_day_spans(days: List[np.datetime64]) -> List[Tuple[np.datetime64, np.datetime64]]:
    if not days:
        return []
    sorted_days = sorted(days)
    spans = []
    span_start = sorted_days[0]
    prev = sorted_days[0]
    for day in sorted_days[1:]:
        if day == prev + np.timedelta64(1, "D"):
            prev = day
            continue
        spans.append((span_start, prev))
        span_start = day
        prev = day
    spans.append((span_start, prev))
    return spans


def _day_start_iso(day: np.datetime64) -> str:
    return f"{_day_key_str(day)}T00:00:00Z"


def _json_fingerprint(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _grid_fingerprint(lat2d: np.ndarray, lon2d: np.ndarray) -> str:
    h = hashlib.sha256()
    for name, arr in (("lat", lat2d), ("lon", lon2d)):
        arr64 = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        h.update(name.encode("utf-8"))
        h.update(json.dumps({"shape": arr64.shape, "dtype": "float64"}).encode("utf-8"))
        h.update(arr64.tobytes())
    return h.hexdigest()[:20]


def _safe_path_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return safe or "unknown"


def _processed_daily_cache_identity(short_name: str, lat2d: np.ndarray, lon2d: np.ndarray) -> Dict:
    return {
        "cache_code_version": PROCESSED_DAILY_CACHE_VERSION,
        "short_name": short_name,
        "needed_nldas_vars": list(NEEDED_NLDAS_VARS),
        "daily_forcing_channels": list(DAILY_FORCING_CHANNELS),
        "needed_nldas_vars_hash": _json_fingerprint(list(NEEDED_NLDAS_VARS)),
        "daily_forcing_channels_hash": _json_fingerprint(list(DAILY_FORCING_CHANNELS)),
        "grid_fingerprint": _grid_fingerprint(lat2d, lon2d),
        "grid_shape": list(np.asarray(lat2d).shape),
    }


def _processed_daily_cache_base(
    cache_dir: str,
    processed_cache_dir: Optional[str],
    short_name: str,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
) -> Tuple[str, Dict]:
    root = os.path.abspath(processed_cache_dir or os.path.join(cache_dir, "processed_daily"))
    identity = _processed_daily_cache_identity(short_name, lat2d, lon2d)
    base = os.path.join(
        root,
        _safe_path_component(identity["cache_code_version"]),
        _safe_path_component(short_name),
        f"vars_{identity['needed_nldas_vars_hash']}",
        f"channels_{identity['daily_forcing_channels_hash']}",
        f"grid_{identity['grid_fingerprint']}",
    )
    identity["processed_daily_cache_dir"] = root
    identity["processed_daily_cache_base"] = base
    return base, identity


def _processed_daily_cache_path(base: str, day: np.datetime64) -> str:
    day_str = _day_key_str(day)
    return os.path.join(base, day_str[:4], f"{day_str}.npz")


def _load_processed_daily_cache_day(path: str, day: np.datetime64, expected_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            x = z["X"]
            day_key = str(z["day_key"].item())
            channels = [str(c) for c in z["daily_forcing_channels"].tolist()]
            if day_key != _day_key_str(day):
                return None
            if channels != list(DAILY_FORCING_CHANNELS):
                return None
            if tuple(x.shape) != expected_shape:
                return None
            return x.astype(np.float32, copy=False)
    except Exception:
        return None


def _write_processed_daily_cache_day(path: str, day: np.datetime64, x_day: np.ndarray, identity: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    day_str = _day_key_str(day)
    metadata = dict(identity)
    metadata["day_key"] = day_str
    tmp_path = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
    np.savez_compressed(
        tmp_path,
        X=np.asarray(x_day, dtype=np.float32),
        day_key=np.array(day_str),
        daily_forcing_channels=np.array(DAILY_FORCING_CHANNELS, dtype="U16"),
        needed_nldas_vars=np.array(NEEDED_NLDAS_VARS, dtype="U16"),
        cache_metadata_json=np.array(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(tmp_path, path)


def _hourly_key(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M")


def _extract_nldas_hour_from_name(name: str) -> Optional[datetime]:
    match = _RAW_HOURLY_FILENAME_RE.search(name)
    if not match:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _expected_hourly_datetimes(t0: datetime, t1: datetime) -> List[datetime]:
    if t1 <= t0:
        raise ValueError(f"end_iso {t1.isoformat()} must be after start_iso {t0.isoformat()}")
    cur = t0.replace(minute=0, second=0, microsecond=0)
    if cur < t0:
        cur += timedelta(hours=1)
    expected = []
    while cur < t1:
        expected.append(cur)
        cur += timedelta(hours=1)
    return expected


def _find_cached_hourly_paths(
    cache_dir: str,
    short_name: str,
    expected_hours: List[datetime],
) -> Dict[str, str]:
    expected_keys = {_hourly_key(dt) for dt in expected_hours}
    hits: Dict[str, str] = {}
    short_name_lc = (short_name or "").lower()

    for root, dirs, files in os.walk(cache_dir):
        dirs[:] = [d for d in dirs if d != "processed_daily" and not d.startswith(".")]
        for name in files:
            name_lc = name.lower()
            if short_name_lc and short_name_lc not in name_lc:
                continue
            if name_lc.endswith((".tmp", ".part", ".lock")):
                continue
            if not name_lc.endswith(_RAW_HOURLY_DATA_EXTENSIONS):
                continue
            dt = _extract_nldas_hour_from_name(name)
            if dt is None:
                continue
            key = _hourly_key(dt)
            if key not in expected_keys:
                continue
            path = os.path.abspath(os.path.join(root, name))
            if key not in hits or path < hits[key]:
                hits[key] = path
    return hits


def _probe_raw_hourly_cache(
    cache_dir: str,
    short_name: str,
    t0: datetime,
    t1: datetime,
) -> Tuple[List[str], List[datetime], List[datetime], float]:
    expected_hours = _expected_hourly_datetimes(t0, t1)
    t_lookup = time.perf_counter()
    hits = _find_cached_hourly_paths(cache_dir, short_name, expected_hours)
    lookup_seconds = time.perf_counter() - t_lookup

    paths = []
    missing = []
    for dt in expected_hours:
        path = hits.get(_hourly_key(dt))
        if path:
            paths.append(path)
        else:
            missing.append(dt)
    return paths, expected_hours, missing, lookup_seconds


def _complete_day_key_strings_from_hourly(ds_hourly: xr.Dataset) -> set:
    if "time" not in ds_hourly.coords:
        return set()
    hours = np.asarray(ds_hourly["time"].values).astype("datetime64[h]")
    if hours.size == 0:
        return set()
    days = hours.astype("datetime64[D]")
    complete_days = set()
    for day in np.unique(days):
        if np.unique(hours[days == day]).size >= 24:
            complete_days.add(str(day))
    return complete_days


def fetch_nldas_forb_hourly_to_cache(
    start_iso: str,
    end_iso: str,
    cache_dir: Optional[str] = None,
    short_name: Optional[str] = None,
    quiet: bool = True,
) -> Tuple[List[str], Dict]:
    cache_dir = _resolve_nldas_cache_dir(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    short_name = short_name or DEFAULT_NLDAS_SHORTNAME

    t0 = _parse_iso_to_utc_dt(start_iso)
    t1 = _parse_iso_to_utc_dt(end_iso)

    cached_paths, expected_hours, missing_hours, lookup_seconds = _probe_raw_hourly_cache(
        cache_dir,
        short_name,
        t0,
        t1,
    )
    raw_cache_hit = len(expected_hours) > 0 and len(missing_hours) == 0
    used = {
        "short_name": short_name,
        "cache_dir": os.path.abspath(cache_dir),
        "start_iso": start_iso,
        "end_iso": end_iso,
        "raw_hourly_cache_hit": raw_cache_hit,
        "n_expected_hours": len(expected_hours),
        "n_cached_hourly_files_before": len(cached_paths),
        "n_missing_hourly_files_before": len(missing_hours),
        "missing_hourly_keys_before": [_hourly_key(dt) for dt in missing_hours[:24]],
        "raw_hourly_cache_lookup_time_seconds": lookup_seconds,
    }

    if raw_cache_hit:
        used.update(
            {
                "n_granules": len(cached_paths),
                "n_downloaded_paths": 0,
                "n_paths": len(cached_paths),
                "earthaccess_search_performed": False,
                "earthaccess_download_performed": False,
                "search_performed": False,
                "download_performed": False,
                "earthaccess_search_time_seconds": 0.0,
                "earthaccess_download_time_seconds": 0.0,
                "n_cached_hourly_files_after": len(cached_paths),
                "n_missing_hourly_files_after": 0,
                "missing_hourly_keys_after": [],
            }
        )
        return cached_paths, used

    import earthaccess

    try:
        earthaccess.login(persist=True)
    except Exception:
        pass

    t_search = time.perf_counter()
    results = earthaccess.search_data(
        short_name=short_name,
        temporal=(t0.isoformat(), t1.isoformat()),
        count=-1,
    )
    search_seconds = time.perf_counter() - t_search
    if not results:
        raise RuntimeError(f"No NLDAS granules found for {short_name} in {t0} .. {t1}")

    t_download = time.perf_counter()
    downloaded = earthaccess.download(results, local_path=cache_dir)
    download_seconds = time.perf_counter() - t_download

    downloaded_paths = []
    for p in downloaded:
        if not p:
            continue
        downloaded_paths.append(os.path.abspath(p))
    paths = downloaded_paths

    cached_paths_after, _, missing_after, _ = _probe_raw_hourly_cache(cache_dir, short_name, t0, t1)
    if len(missing_after) == 0:
        paths = cached_paths_after

    used.update(
        {
            "raw_hourly_cache_hit": False,
            "n_granules": len(results),
            "n_downloaded_paths": len(downloaded_paths),
            "n_paths": len(paths),
            "earthaccess_search_performed": True,
            "earthaccess_download_performed": True,
            "search_performed": True,
            "download_performed": True,
            "earthaccess_search_time_seconds": search_seconds,
            "earthaccess_download_time_seconds": download_seconds,
            "n_cached_hourly_files_after": len(cached_paths_after),
            "n_missing_hourly_files_after": len(missing_after),
            "missing_hourly_keys_after": [_hourly_key(dt) for dt in missing_after[:24]],
        }
    )
    return paths, used


def open_forb_hourly_as_xarray(paths: List[str]) -> xr.Dataset:
    if not paths:
        raise ValueError("open_forb_hourly_as_xarray retrieved empty paths list")
    return _open_many_hourly(sorted(paths))


def to_daily_forcing_on_target_grid(
    ds_hourly: xr.Dataset,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    pad_deg: float = 0.5,
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    for v in NEEDED_NLDAS_VARS:
        if v not in ds_hourly:
            raise KeyError(f"Missing {v} in NLDAS dataset. Available: {list(ds_hourly.data_vars)}")

    has_lwdown = "LWdown" in ds_hourly

    ds = _subset_bbox(ds_hourly, lat2d, lon2d, pad_deg=pad_deg)

    wind_mag = np.sqrt(ds["Wind_E"] ** 2 + ds["Wind_N"] ** 2)

    rain_units = (ds["Rainf"].attrs.get("units") or "").lower()
    rain_is_rate = ("s-1" in rain_units) or ("/s" in rain_units)

    ds_mean = xr.Dataset(
        {
            "SWdown": ds["SWdown"],
            "Tair": ds["Tair"],
            "Qair": ds["Qair"],
            "PSurf": ds["PSurf"],
            "WIND": wind_mag,
        }
    )
    if has_lwdown:
        ds_mean["LWdown"] = ds["LWdown"]

    ds_mean = ds_mean.resample(time="1D").mean(keep_attrs=True)

    if rain_is_rate:
        ds_rain = xr.Dataset({"Rainf": ds["Rainf"]}).resample(time="1D").mean(keep_attrs=True)
        prect_mms = ds_rain["Rainf"].astype("float32")
    else:
        ds_rain = xr.Dataset({"Rainf": ds["Rainf"]}).resample(time="1D").sum(keep_attrs=True)
        prect_mms = (ds_rain["Rainf"] / 86400.0).astype("float32")

    ds_daily = xr.merge([ds_mean, ds_rain])

    lat_src = ds_daily["lat"].values
    lon_src = ds_daily["lon"].values
    w = _precompute_bilinear_weights(lat_src, lon_src, lat2d, lon2d)

    def rg(var_da: xr.DataArray) -> np.ndarray:
        arr = var_da.transpose("time", "lat", "lon").astype("float32").values
        return _bilinear_regrid_3d(arr, w)

    FSDS = rg(ds_daily["SWdown"])
    TBOT = rg(ds_daily["Tair"])
    QBOT = rg(ds_daily["Qair"])
    PSRF = rg(ds_daily["PSurf"])
    WIND = rg(ds_daily["WIND"])

    if has_lwdown:
        FLDS = rg(ds_daily["LWdown"])
    else:
        FLDS = np.zeros_like(FSDS, dtype=np.float32)

    PRECT = _bilinear_regrid_3d(
        prect_mms.transpose("time", "lat", "lon").values.astype("float32"),
        w,
    )

    X = np.stack([PRECT, TBOT, WIND, QBOT, PSRF, FSDS, FLDS], axis=-1).astype(np.float32)
    day_keys = ds_daily["time"].values.astype("datetime64[D]")

    return X, day_keys


def load_or_build_daily_forcing_on_target_grid(
    start_day,
    end_day,
    cache_dir: Optional[str],
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    processed_cache_dir: Optional[str] = None,
    short_name: str = DEFAULT_NLDAS_SHORTNAME,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    cache_dir = _resolve_nldas_cache_dir(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    short_name = short_name or DEFAULT_NLDAS_SHORTNAME

    requested_days = _daily_day_range(start_day, end_day)
    expected_day_shape = tuple(np.asarray(lat2d).shape) + (len(DAILY_FORCING_CHANNELS),)
    cache_base, cache_identity = _processed_daily_cache_base(
        cache_dir,
        processed_cache_dir,
        short_name,
        lat2d,
        lon2d,
    )

    day_to_x: Dict[str, np.ndarray] = {}
    cached_days: List[str] = []
    missing_days: List[np.datetime64] = []

    for day in requested_days:
        day_str = _day_key_str(day)
        path = _processed_daily_cache_path(cache_base, day)
        x_day = _load_processed_daily_cache_day(path, day, expected_day_shape)
        if x_day is None:
            missing_days.append(day)
            continue
        day_to_x[day_str] = x_day
        cached_days.append(day_str)

    built_days: List[str] = []
    raw_hourly_fetches: List[Dict] = []
    xarray_open_time_seconds = 0.0
    daily_resample_regrid_time_seconds = 0.0

    for span_start, span_end in _contiguous_day_spans(missing_days):
        start_iso = _day_start_iso(span_start)
        end_iso = _day_start_iso(span_end + np.timedelta64(1, "D"))

        paths, fetch_meta = fetch_nldas_forb_hourly_to_cache(
            start_iso=start_iso,
            end_iso=end_iso,
            cache_dir=cache_dir,
            short_name=short_name,
        )
        raw_hourly_fetches.append(fetch_meta)

        t_open = time.perf_counter()
        ds_hourly = open_forb_hourly_as_xarray(paths)
        xarray_open_time_seconds += time.perf_counter() - t_open

        try:
            complete_day_keys = _complete_day_key_strings_from_hourly(ds_hourly)

            t_daily = time.perf_counter()
            X_span, span_day_keys = to_daily_forcing_on_target_grid(ds_hourly, lat2d, lon2d)
            daily_resample_regrid_time_seconds += time.perf_counter() - t_daily
        finally:
            ds_hourly.close()

        span_day_to_idx = {_day_key_str(day): i for i, day in enumerate(span_day_keys.astype("datetime64[D]"))}
        for day in _daily_day_range(span_start, span_end):
            day_str = _day_key_str(day)
            if day_str not in complete_day_keys or day_str not in span_day_to_idx:
                continue
            x_day = np.asarray(X_span[span_day_to_idx[day_str]], dtype=np.float32)
            path = _processed_daily_cache_path(cache_base, day)
            _write_processed_daily_cache_day(path, day, x_day, cache_identity)
            day_to_x[day_str] = x_day
            built_days.append(day_str)

    unresolved_days = [_day_key_str(day) for day in requested_days if _day_key_str(day) not in day_to_x]
    if unresolved_days:
        raise RuntimeError(f"Unable to build complete NLDAS daily forcing for days: {unresolved_days}")

    ordered_day_keys = np.array([_normalize_day(day) for day in requested_days], dtype="datetime64[D]")
    X_phys = np.stack([day_to_x[_day_key_str(day)] for day in requested_days], axis=0).astype(np.float32)

    raw_cache_hits = [bool(m.get("raw_hourly_cache_hit")) for m in raw_hourly_fetches]
    meta = {
        "processed_daily_cache_hit": len(missing_days) == 0,
        "processed_daily_cache_partial_hit": bool(cached_days) and bool(built_days),
        "processed_daily_requested_day_count": len(requested_days),
        "processed_daily_cached_day_count": len(cached_days),
        "processed_daily_built_day_count": len(built_days),
        "requested_days": [_day_key_str(day) for day in requested_days],
        "cached_days": cached_days,
        "built_days": built_days,
        "processed_daily_cache": cache_identity,
        "raw_hourly_cache_hit": all(raw_cache_hits) if raw_cache_hits else None,
        "raw_hourly_fetches": raw_hourly_fetches,
        "xarray_open_time_seconds": xarray_open_time_seconds,
        "daily_resample_regrid_time_seconds": daily_resample_regrid_time_seconds,
    }

    return X_phys, ordered_day_keys, meta

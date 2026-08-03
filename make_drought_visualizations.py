from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from urllib.parse import urljoin

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.lines import Line2D
from shapely import contains_xy
from shapely.geometry import shape

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import DEFAULT_COPULA_TAU, ELM_ET_PATH, ELM_SM_PATH  # noqa: E402
from drought_index import clayton_lower_tail_joint, drought_from_zscores, to_usdm_category, z_to_u  # noqa: E402


USDM_GEOJSON_BASE = "https://droughtmonitor.unl.edu/data/json/"
USDM_STATS_URL = (
    "https://usdmdataservices.unl.edu/api/USStatistics/"
    "GetDroughtSeverityStatisticsByAreaPercent"
)

USDM_COLORS = {
    0: "#ffffff",
    1: "#ffff00",
    2: "#fcd37f",
    3: "#ffaa00",
    4: "#e60000",
    5: "#730000",
}
USDM_LABELS = ["None", "D0", "D1", "D2", "D3", "D4"]
USDM_CMAP = ListedColormap([USDM_COLORS[i] for i in range(6)])
USDM_NORM = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), USDM_CMAP.N)
HORIZONS = {"Weekly": 7, "Monthly": 30, "Seasonal": 90}
DEFAULT_CASE_DATES = {
    "Drought A": "2018-07-31",
    "Drought B": "2019-09-24",
    "Non-drought A": "2015-05-26",
    "Non-drought B": "2017-05-30",
}


def _open_elm(path: str | Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="h5netcdf", mask_and_scale=True)


def _var_name(ds: xr.Dataset, preferred: str | None = None) -> str:
    if preferred and preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) != 1:
        raise ValueError(f"Expected one data variable in {ds.encoding.get('source')}, got {list(ds.data_vars)}")
    return next(iter(ds.data_vars))


def _circular_doy_distance(day_of_year: np.ndarray, target_doy: int) -> np.ndarray:
    diff = np.abs(day_of_year.astype(np.int16) - int(target_doy))
    return np.minimum(diff, 366 - diff)


def _load_target_and_climatology(
    *,
    et_path: Path,
    sm_path: Path,
    target_date: np.datetime64,
    climatology_window_days: int,
) -> dict[str, np.ndarray]:
    with _open_elm(et_path) as et_ds, _open_elm(sm_path) as sm_ds:
        et_name = _var_name(et_ds, "EVAPOTRANSPIRATION")
        sm_name = _var_name(sm_ds, "H2OSOI")

        target = np.datetime64(target_date, "D")
        if target not in et_ds.time.values.astype("datetime64[D]"):
            raise ValueError(f"{target} is not available in {et_path}")
        if target not in sm_ds.time.values.astype("datetime64[D]"):
            raise ValueError(f"{target} is not available in {sm_path}")

        et_target = et_ds[et_name].sel(time=target).astype("float32").values
        sm_target = sm_ds[sm_name].sel(time=target).astype("float32").values

        time_days = et_ds.time.values.astype("datetime64[D]")
        day_of_year = (
            time_days - time_days.astype("datetime64[Y]")
        ).astype("timedelta64[D]").astype(np.int16) + 1
        target_doy = int((target - target.astype("datetime64[Y]")).astype("timedelta64[D]").astype(int) + 1)
        clim_mask = _circular_doy_distance(day_of_year, target_doy) <= int(climatology_window_days)

        et_clim = et_ds[et_name].isel(time=clim_mask).astype("float32")
        sm_clim = sm_ds[sm_name].isel(time=clim_mask).astype("float32")
        et_mu = et_clim.mean("time", skipna=True).values
        sm_mu = sm_clim.mean("time", skipna=True).values
        et_sd = et_clim.std("time", skipna=True).values
        sm_sd = sm_clim.std("time", skipna=True).values

        lat = et_ds["lat"].values.astype("float32")
        lon = et_ds["lon"].values.astype("float32")

        pdf_et = et_clim.values
        pdf_sm = sm_clim.values

    z_et = (et_target - et_mu) / np.where(et_sd > 0, et_sd, np.nan)
    z_sm = (sm_target - sm_mu) / np.where(sm_sd > 0, sm_sd, np.nan)
    pdry, category = drought_from_zscores(
        z_et=z_et,
        z_sm=z_sm,
        tau=DEFAULT_COPULA_TAU,
        sensitivity=1.0,
    )

    return {
        "lat": lat,
        "lon": np.where(lon > 180.0, lon - 360.0, lon).astype("float32"),
        "et_target": et_target,
        "sm_target": sm_target,
        "z_et": z_et.astype("float32"),
        "z_sm": z_sm.astype("float32"),
        "pdry": pdry.astype("float32"),
        "category": category.astype("int8"),
        "pdf_et": pdf_et.astype("float32"),
        "pdf_sm": pdf_sm.astype("float32"),
    }


def _target_index_from_arrays(
    *,
    et: np.ndarray,
    sm: np.ndarray,
    time_days: np.ndarray,
    target_date: np.datetime64,
    lat: np.ndarray,
    lon: np.ndarray,
    climatology_window_days: int,
) -> dict[str, np.ndarray]:
    target = np.datetime64(target_date, "D")
    target_idx = np.where(time_days == target)[0]
    if target_idx.size == 0:
        raise ValueError(f"{target} is not available in the ELM arrays")
    target_idx = int(target_idx[0])

    day_of_year = (
        time_days - time_days.astype("datetime64[Y]")
    ).astype("timedelta64[D]").astype(np.int16) + 1
    target_doy = int((target - target.astype("datetime64[Y]")).astype("timedelta64[D]").astype(int) + 1)
    clim_mask = _circular_doy_distance(day_of_year, target_doy) <= int(climatology_window_days)

    et_clim = et[clim_mask]
    sm_clim = sm[clim_mask]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        et_mu = np.nanmean(et_clim, axis=0)
        sm_mu = np.nanmean(sm_clim, axis=0)
        et_sd = np.nanstd(et_clim, axis=0)
        sm_sd = np.nanstd(sm_clim, axis=0)
    z_et = (et[target_idx] - et_mu) / np.where(et_sd > 0, et_sd, np.nan)
    z_sm = (sm[target_idx] - sm_mu) / np.where(sm_sd > 0, sm_sd, np.nan)
    pdry, category = drought_from_zscores(
        z_et=z_et,
        z_sm=z_sm,
        tau=DEFAULT_COPULA_TAU,
        sensitivity=1.0,
    )

    return {
        "lat": lat,
        "lon": lon,
        "z_et": z_et.astype("float32"),
        "z_sm": z_sm.astype("float32"),
        "pdry": pdry.astype("float32"),
        "category": category.astype("int8"),
    }


def _load_horizon_case_data(
    *,
    et_path: Path,
    sm_path: Path,
    target_dates: list[np.datetime64],
    windows: dict[str, int],
    climatology_window_days: int,
) -> dict[tuple[str, np.datetime64], dict[str, np.ndarray]]:
    with _open_elm(et_path) as et_ds, _open_elm(sm_path) as sm_ds:
        et_name = _var_name(et_ds, "EVAPOTRANSPIRATION")
        sm_name = _var_name(sm_ds, "H2OSOI")
        lat = et_ds["lat"].values.astype("float32")
        lon = np.where(et_ds["lon"].values.astype("float32") > 180.0, et_ds["lon"].values.astype("float32") - 360.0, et_ds["lon"].values.astype("float32"))
        time_days = et_ds.time.values.astype("datetime64[D]")

        cases: dict[tuple[str, np.datetime64], dict[str, np.ndarray]] = {}
        for horizon_name, window_days in windows.items():
            et_roll = (
                et_ds[et_name]
                .rolling(time=window_days, min_periods=window_days)
                .mean()
                .astype("float32")
                .values
            )
            sm_roll = (
                sm_ds[sm_name]
                .rolling(time=window_days, min_periods=window_days)
                .mean()
                .astype("float32")
                .values
            )
            for target_date in target_dates:
                cases[(horizon_name, target_date)] = _target_index_from_arrays(
                    et=et_roll,
                    sm=sm_roll,
                    time_days=time_days,
                    target_date=target_date,
                    lat=lat,
                    lon=lon.astype("float32"),
                    climatology_window_days=climatology_window_days,
                )
    return cases


def _load_period_standardized_anomalies(
    *,
    et_path: Path,
    sm_path: Path,
    start_date: np.datetime64,
    end_date: np.datetime64,
    sample_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    with _open_elm(et_path) as et_ds, _open_elm(sm_path) as sm_ds:
        et_name = _var_name(et_ds, "EVAPOTRANSPIRATION")
        sm_name = _var_name(sm_ds, "H2OSOI")
        et = et_ds[et_name].astype("float32").values
        sm = sm_ds[sm_name].astype("float32").values
        time_days = et_ds.time.values.astype("datetime64[D]")

    day_of_year = (
        time_days - time_days.astype("datetime64[Y]")
    ).astype("timedelta64[D]").astype(np.int16) + 1
    period_mask = (time_days >= np.datetime64(start_date, "D")) & (time_days <= np.datetime64(end_date, "D"))
    z_et = np.full(et[period_mask].shape, np.nan, dtype=np.float32)
    z_sm = np.full(sm[period_mask].shape, np.nan, dtype=np.float32)
    period_doy = day_of_year[period_mask]
    period_indices = np.where(period_mask)[0]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean for empty slice")
        warnings.filterwarnings("ignore", message="Negative degrees of freedom for slice")
        for doy in np.unique(period_doy):
            clim_mask = day_of_year == doy
            target_positions = np.where(period_doy == doy)[0]
            target_indices = period_indices[target_positions]
            et_mu = np.nanmean(et[clim_mask], axis=0)
            et_sd = np.nanstd(et[clim_mask], axis=0)
            sm_mu = np.nanmean(sm[clim_mask], axis=0)
            sm_sd = np.nanstd(sm[clim_mask], axis=0)
            z_et[target_positions] = (et[target_indices] - et_mu) / np.where(et_sd > 0, et_sd, np.nan)
            z_sm[target_positions] = (sm[target_indices] - sm_mu) / np.where(sm_sd > 0, sm_sd, np.nan)

    m = np.isfinite(z_et) & np.isfinite(z_sm)
    z_et = z_et[m].astype("float32")
    z_sm = z_sm[m].astype("float32")
    if z_et.size > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(z_et.size, size=sample_size, replace=False)
        z_et = z_et[idx]
        z_sm = z_sm[idx]
    return z_et, z_sm


def _download_usdm_geojson(date_yyyymmdd: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "usdm_current.json" if date_yyyymmdd.lower() == "current" else f"usdm_{date_yyyymmdd}.json"
    url = urljoin(USDM_GEOJSON_BASE, filename)
    out = output_dir / filename
    if out.exists() and out.stat().st_size > 0:
        return out

    response = requests.get(url, timeout=90)
    response.raise_for_status()
    out.write_bytes(response.content)
    return out


def _download_usdm_conus_stats(date_yyyymmdd: str, output_dir: Path) -> Path:
    if date_yyyymmdd.lower() == "current":
        return output_dir / "usdm_current_stats_not_requested.csv"
    date = f"{int(date_yyyymmdd[4:6])}/{int(date_yyyymmdd[6:8])}/{date_yyyymmdd[:4]}"
    params = {
        "aoi": "conus",
        "startdate": date,
        "enddate": date,
        "statisticsType": "1",
    }
    out = output_dir / f"usdm_conus_stats_{date_yyyymmdd}.csv"
    if out.exists() and out.stat().st_size > 0:
        return out
    response = requests.get(USDM_STATS_URL, params=params, headers={"Accept": "text/csv"}, timeout=60)
    response.raise_for_status()
    out.write_text(response.text, encoding="utf-8")
    return out


def _iter_polygons(geom):
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        yield from geom.geoms
    elif geom.geom_type == "GeometryCollection":
        for item in geom.geoms:
            yield from _iter_polygons(item)


def _rasterize_usdm_to_grid(geojson_path: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    with geojson_path.open("r", encoding="utf-8") as f:
        feature_collection = json.load(f)

    lon2d, lat2d = np.meshgrid(lon, lat)
    out = np.zeros(lat2d.shape, dtype=np.int8)
    bbox = (float(np.nanmin(lon)), float(np.nanmin(lat)), float(np.nanmax(lon)), float(np.nanmax(lat)))

    features = []
    for feature in feature_collection["features"]:
        dm_value = int(feature["properties"]["DM"]) + 1
        for geom in _iter_polygons(shape(feature["geometry"])):
            gx0, gy0, gx1, gy1 = geom.bounds
            if gx1 < bbox[0] or gx0 > bbox[2] or gy1 < bbox[1] or gy0 > bbox[3]:
                continue
            features.append((dm_value, geom, geom.bounds))

    for dm_value, geom, bounds in features:
        bbox_mask = (
            (lon2d >= bounds[0])
            & (lon2d <= bounds[2])
            & (lat2d >= bounds[1])
            & (lat2d <= bounds[3])
            & (out == 0)
        )
        if not np.any(bbox_mask):
            continue
        inside = contains_xy(geom, lon2d[bbox_mask], lat2d[bbox_mask])
        rows, cols = np.where(bbox_mask)
        out[rows[inside], cols[inside]] = dm_value
    return out


def _category_fractions(cat: np.ndarray) -> dict[str, float]:
    values = np.asarray(cat).ravel()
    values = values[np.isfinite(values)]
    total = values.size
    if total == 0:
        return {label: np.nan for label in USDM_LABELS}
    return {USDM_LABELS[i]: 100.0 * float(np.sum(values == i)) / float(total) for i in range(6)}


def _percent_agreement(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    valid = np.isfinite(a_arr) & np.isfinite(b_arr)
    if not np.any(valid):
        return float("nan")
    return 100.0 * float(np.mean(a_arr[valid] == b_arr[valid]))


def _copula_category_from_zscores(
    z_et: np.ndarray,
    z_sm: np.ndarray,
    *,
    tau: float = DEFAULT_COPULA_TAU,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(z_et) & np.isfinite(z_sm)
    pdry = np.full(np.asarray(z_et).shape, np.nan, dtype=np.float32)
    if np.any(valid):
        pdry[valid] = clayton_lower_tail_joint(
            z_to_u(np.asarray(z_et)[valid]),
            z_to_u(np.asarray(z_sm)[valid]),
            tau=tau,
        )
    return pdry, to_usdm_category(pdry, sensitivity=1.0)


def _plot_joint_pdf_from_z(
    z_et: np.ndarray,
    z_sm: np.ndarray,
    label: str,
    output_dir: Path,
    output_name: str,
) -> Path:
    z_et = np.asarray(z_et).ravel()
    z_sm = np.asarray(z_sm).ravel()
    m = np.isfinite(z_et) & np.isfinite(z_sm)
    z_et = z_et[m]
    z_sm = z_sm[m]

    hist, xedges, yedges = np.histogram2d(
        z_et,
        z_sm,
        bins=120,
        range=[[-3.5, 3.5], [-3.5, 3.5]],
        density=True,
    )
    hist = np.ma.masked_less_equal(hist.T, 0.0)

    xgrid = np.linspace(-3.5, 3.5, 220)
    ygrid = np.linspace(-3.5, 3.5, 220)
    xx, yy = np.meshgrid(xgrid, ygrid)
    pdry_grid, _ = drought_from_zscores(xx, yy, tau=DEFAULT_COPULA_TAU, sensitivity=1.0)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=300)
    mesh = ax.pcolormesh(
        xedges,
        yedges,
        hist,
        cmap="viridis",
        norm=LogNorm(vmin=max(float(hist.min()), 1e-4), vmax=float(hist.max())),
        shading="auto",
    )
    contour_levels = [0.70, 0.80, 0.90, 0.95, 0.98]
    contour_colors = ["#666666", "#8a6f2a", "#b45f06", "#cc0000", "#660000"]
    contour_widths = [0.8, 0.9, 1.0, 1.1, 1.2]
    ax.contour(
        xx,
        yy,
        pdry_grid,
        levels=contour_levels,
        colors=contour_colors,
        linewidths=contour_widths,
    )
    handles = [
        Line2D([0], [0], color=color, lw=width, label=label)
        for color, width, label in zip(contour_colors, contour_widths, ["D0", "D1", "D2", "D3", "D4"])
    ]
    ax.legend(
        handles=handles,
        title="Drought-index contours",
        loc="lower left",
        frameon=True,
        framealpha=0.92,
        borderpad=0.5,
        fontsize=8,
        title_fontsize=8,
    )
    ax.axvline(0.0, color="0.35", linewidth=0.8, linestyle=":")
    ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Evapotranspiration standardized anomaly")
    ax.set_ylabel("Soil moisture standardized anomaly")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.90", linewidth=0.5)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.025, shrink=0.88)
    cbar.set_label("Joint PDF")
    fig.tight_layout()

    out = output_dir / output_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_joint_pdf(data: dict[str, np.ndarray], target_date: str, output_dir: Path) -> Path:
    return _plot_joint_pdf_from_z(
        data["z_et"],
        data["z_sm"],
        target_date,
        output_dir,
        f"joint_pdf_et_sm_{target_date}.png",
    )


def _plot_comparison(
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    product_cat: np.ndarray,
    product_label: str,
    usdm_cat: np.ndarray,
    target_date: str,
    output_dir: Path,
    output_name: str,
) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    lon2d, lat2d = np.meshgrid(lon, lat)
    extent = [float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))]
    product_frac = _category_fractions(product_cat)
    usdm_frac = _category_fractions(usdm_cat)

    fig = plt.figure(figsize=(9.53, 7.30), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.24], hspace=0.18, wspace=0.06)
    axes = [
        fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()),
    ]
    for ax, field in zip(axes, [product_cat, usdm_cat]):
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.pcolormesh(lon2d, lat2d, field, cmap=USDM_CMAP, norm=USDM_NORM, shading="auto", transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.35, edgecolor="0.25")
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.4, edgecolor="0.2")
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="0.3")
        gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="0.65", alpha=0.7, linestyle=":")
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = False
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}

    cax = fig.add_axes([0.91, 0.38, 0.018, 0.42])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(cmap=USDM_CMAP, norm=USDM_NORM),
        cax=cax,
        orientation="vertical",
        ticks=list(range(6)),
    )
    cb.ax.set_yticklabels(USDM_LABELS, fontsize=9)
    cb.ax.tick_params(labelsize=9)
    cb.set_label("Drought category", fontsize=11, fontweight="normal", labelpad=10)

    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.axis("off")
    labels = USDM_LABELS
    rows = [
        [product_label] + [f"{product_frac[label]:.1f}" for label in labels],
        ["USDM"] + [f"{usdm_frac[label]:.1f}" for label in labels],
    ]
    table = ax_tbl.table(
        cellText=rows,
        colLabels=["Product"] + [f"{label} (%)" for label in labels],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold")

    out = output_dir / output_name
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_horizon_case_comparison(
    *,
    cases: dict[tuple[str, np.datetime64], dict[str, np.ndarray]],
    usdm_by_date: dict[np.datetime64, np.ndarray],
    case_labels: dict[str, str],
    output_dir: Path,
    product_label: str,
    output_name: str,
    category_method: str,
) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    first = next(iter(cases.values()))
    lat = first["lat"]
    lon = first["lon"]
    lon2d, lat2d = np.meshgrid(lon, lat)
    extent = [float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))]

    dates = {label: np.datetime64(date, "D") for label, date in case_labels.items()}
    fig = plt.figure(figsize=(15.0, 10.8), dpi=300)
    gs = fig.add_gridspec(
        len(dates),
        len(HORIZONS) * 2,
        left=0.055,
        right=0.93,
        bottom=0.08,
        top=0.88,
        wspace=0.035,
        hspace=0.16,
    )

    axes = []
    pair_annotations = []
    for row_idx, (case_label, target_date) in enumerate(dates.items()):
        row_axes = []
        for horizon_idx, horizon_name in enumerate(HORIZONS):
            case = cases[(horizon_name, target_date)]
            if category_method == "copula":
                _, app_cat = _copula_category_from_zscores(case["z_et"], case["z_sm"])
            elif category_method == "msdi_kendall":
                app_cat = case["category"]
            else:
                raise ValueError(f"Unknown category_method: {category_method}")
            usdm_cat = usdm_by_date[target_date]
            agreement = _percent_agreement(app_cat, usdm_cat)
            pair_axes = []
            for product_idx, (panel_label, field) in enumerate(
                [(product_label, app_cat), ("USDM", usdm_cat)]
            ):
                col_idx = horizon_idx * 2 + product_idx
                ax = fig.add_subplot(gs[row_idx, col_idx], projection=ccrs.PlateCarree())
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.pcolormesh(
                    lon2d,
                    lat2d,
                    field,
                    cmap=USDM_CMAP,
                    norm=USDM_NORM,
                    shading="auto",
                    transform=ccrs.PlateCarree(),
                )
                ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.25, edgecolor="0.25")
                ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.3, edgecolor="0.25")
                ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.22, edgecolor="0.35")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.6)
                    spine.set_edgecolor("0.25")
                row_axes.append(ax)
                pair_axes.append(ax)
            pair_annotations.append((pair_axes[0], pair_axes[1], agreement))
        axes.append(row_axes)

        y0 = axes[row_idx][0].get_position().y0
        y1 = axes[row_idx][0].get_position().y1
        fig.text(
            0.025,
            (y0 + y1) / 2.0,
            f"{case_label}\n{target_date}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=9,
            fontweight="bold",
        )

    for horizon_idx, horizon_name in enumerate(HORIZONS):
        for product_idx, label in enumerate([product_label, "USDM"]):
            ax = axes[0][horizon_idx * 2 + product_idx]
            pos = ax.get_position()
            fig.text(
                (pos.x0 + pos.x1) / 2.0,
                0.902,
                f"{horizon_name}\n{label}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                linespacing=1.15,
            )

    for left_ax, right_ax, agreement in pair_annotations:
        left_pos = left_ax.get_position()
        right_pos = right_ax.get_position()
        x = (left_pos.x0 + right_pos.x1) / 2.0
        y = left_pos.y0 - 0.010
        label = "Agreement: n/a" if not np.isfinite(agreement) else f"Agreement: {agreement:.1f}%"
        fig.text(
            x,
            y,
            label,
            ha="center",
            va="top",
            fontsize=7.5,
            color="0.12",
        )

    cax = fig.add_axes([0.945, 0.24, 0.018, 0.50])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(cmap=USDM_CMAP, norm=USDM_NORM),
        cax=cax,
        orientation="vertical",
        ticks=list(range(6)),
    )
    cb.ax.set_yticklabels(USDM_LABELS, fontsize=9)
    cb.ax.tick_params(labelsize=9)
    cb.set_label("Drought category", fontsize=11, fontweight="normal", labelpad=10)

    out = output_dir / output_name
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ET-SM drought-index PDF and USDM comparison figures.")
    parser.add_argument("--date", default="2019-12-31", help="ELM/USDM comparison date, YYYY-MM-DD.")
    parser.add_argument("--output-dir", default=str(HERE.parent / "outputs"), help="Directory for figures and downloads.")
    parser.add_argument("--climatology-window-days", type=int, default=15, help="Day-of-year half-window for z-score climatology.")
    parser.add_argument("--pdf-start", default="2015-01-01", help="Start date for the all-time joint PDF, YYYY-MM-DD.")
    parser.add_argument("--pdf-end", default="2019-12-31", help="End date for the all-time joint PDF, YYYY-MM-DD.")
    parser.add_argument("--pdf-sample-size", type=int, default=750_000, help="Maximum ET/SM anomaly pairs for the all-time PDF.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_date = np.datetime64(args.date, "D")
    date_yyyymmdd = str(target_date).replace("-", "")

    data = _load_target_and_climatology(
        et_path=Path(ELM_ET_PATH),
        sm_path=Path(ELM_SM_PATH),
        target_date=target_date,
        climatology_window_days=args.climatology_window_days,
    )
    geojson_path = _download_usdm_geojson(date_yyyymmdd, output_dir)
    stats_path = _download_usdm_conus_stats(date_yyyymmdd, output_dir)
    usdm_cat = _rasterize_usdm_to_grid(geojson_path, data["lat"], data["lon"])

    pdf_path = _plot_joint_pdf(data, str(target_date), output_dir)
    _, copula_cat = _copula_category_from_zscores(data["z_et"], data["z_sm"])
    copula_comparison_path = _plot_comparison(
        lat=data["lat"],
        lon=data["lon"],
        product_cat=copula_cat,
        product_label="Clayton copula",
        usdm_cat=usdm_cat,
        target_date=str(target_date),
        output_dir=output_dir,
        output_name=f"copula_usdm_comparison_{target_date}.png",
    )
    kendall_comparison_path = _plot_comparison(
        lat=data["lat"],
        lon=data["lon"],
        product_cat=data["category"],
        product_label="MSDI-Kendall ET-SM",
        usdm_cat=usdm_cat,
        target_date=str(target_date),
        output_dir=output_dir,
        output_name=f"msdi_kendall_usdm_comparison_{target_date}.png",
    )

    period_z_et, period_z_sm = _load_period_standardized_anomalies(
        et_path=Path(ELM_ET_PATH),
        sm_path=Path(ELM_SM_PATH),
        start_date=np.datetime64(args.pdf_start, "D"),
        end_date=np.datetime64(args.pdf_end, "D"),
        sample_size=args.pdf_sample_size,
    )
    period_pdf_path = _plot_joint_pdf_from_z(
        period_z_et,
        period_z_sm,
        f"{args.pdf_start} to {args.pdf_end}",
        output_dir,
        f"joint_pdf_et_sm_{args.pdf_start}_to_{args.pdf_end}.png",
    )

    case_dates = [np.datetime64(date, "D") for date in DEFAULT_CASE_DATES.values()]
    horizon_cases = _load_horizon_case_data(
        et_path=Path(ELM_ET_PATH),
        sm_path=Path(ELM_SM_PATH),
        target_dates=case_dates,
        windows=HORIZONS,
        climatology_window_days=args.climatology_window_days,
    )
    usdm_by_date = {}
    for target in case_dates:
        date_key = str(target).replace("-", "")
        geojson = _download_usdm_geojson(date_key, output_dir)
        sample = horizon_cases[("Weekly", target)]
        usdm_by_date[target] = _rasterize_usdm_to_grid(geojson, sample["lat"], sample["lon"])
    copula_horizons_path = _plot_horizon_case_comparison(
        cases=horizon_cases,
        usdm_by_date=usdm_by_date,
        case_labels=DEFAULT_CASE_DATES,
        output_dir=output_dir,
        product_label="Copula-based index",
        output_name="fig7_usdm_vs_copula_drought_index_all_horizons_conditions_percent_agreement.png",
        category_method="copula",
    )
    kendall_horizons_path = _plot_horizon_case_comparison(
        cases=horizon_cases,
        usdm_by_date=usdm_by_date,
        case_labels=DEFAULT_CASE_DATES,
        output_dir=output_dir,
        product_label="MSDI-Kendall index",
        output_name="fig8_usdm_vs_msdi_kendall_drought_index_all_horizons_conditions_percent_agreement.png",
        category_method="msdi_kendall",
    )

    print(f"Joint PDF: {pdf_path}")
    print(f"Joint PDF 2015-2019: {period_pdf_path}")
    print(f"Copula comparison: {copula_comparison_path}")
    print(f"MSDI-Kendall comparison: {kendall_comparison_path}")
    print(f"Fig. 7 Copula horizon/condition comparison: {copula_horizons_path}")
    print(f"Fig. 8 MSDI-Kendall horizon/condition comparison: {kendall_horizons_path}")
    print(f"USDM GeoJSON: {geojson_path}")
    print(f"USDM CONUS statistics CSV: {stats_path}")


if __name__ == "__main__":
    main()

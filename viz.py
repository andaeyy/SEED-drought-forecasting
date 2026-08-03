from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


def _norm_lon_180(lon2d: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon2d, dtype=np.float32)
    return np.where(lon > 180.0, lon - 360.0, lon)


def _extent_from_latlon(lat2d: np.ndarray, lon2d: np.ndarray, pad_deg: float = 2.0):
    lonp = _norm_lon_180(lon2d)
    lon_min = float(np.nanmin(lonp)) - pad_deg
    lon_max = float(np.nanmax(lonp)) + pad_deg
    lat_min = float(np.nanmin(lat2d)) - pad_deg
    lat_max = float(np.nanmax(lat2d)) + pad_deg
    return [lon_min, lon_max, lat_min, lat_max]


def _try_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except Exception:
        return None, None


def plot_field(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    field: np.ndarray,
    title: str,
    *,
    pad_deg: float = 2.0,
):
    field = np.asarray(field, dtype=np.float32)
    lat2d = np.asarray(lat2d, dtype=np.float32)
    lonp = _norm_lon_180(lon2d)

    ccrs, cfeature = _try_cartopy()

    if ccrs is None:
        fig, ax = plt.subplots(figsize=(12, 7))
        m = ax.pcolormesh(lonp, lat2d, field, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        fig.colorbar(m, ax=ax, shrink=0.85)
        fig.tight_layout()
        return fig

    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(_extent_from_latlon(lat2d, lonp, pad_deg=pad_deg), crs=ccrs.PlateCarree())

    m = ax.pcolormesh(lonp, lat2d, field, shading="auto", transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.STATES, linewidth=0.7)
    ax.coastlines(linewidth=0.8)

    ax.set_title(title)
    plt.colorbar(m, ax=ax, shrink=0.80, pad=0.03)
    plt.tight_layout()
    return fig


def plot_category(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    cat: np.ndarray,
    title: str = "Drought map using ELM based ML-Emulator",
    *,
    pad_deg: float = 2.0,
):
    cat = np.asarray(cat, dtype=np.float32)
    lat2d = np.asarray(lat2d, dtype=np.float32)
    lonp = _norm_lon_180(lon2d)

    ccrs, cfeature = _try_cartopy()

    cmap = ListedColormap(["#FFFFFF", "#FFFF00", "#FCD37F", "#FFAA00", "#E60000", "#730000"])
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), cmap.N)

    if ccrs is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        m = ax.pcolormesh(lonp, lat2d, cat, shading="auto", cmap=cmap, norm=norm)
        ax.set_title(title)
        cb = fig.colorbar(m, ax=ax, shrink=0.85, ticks=[0, 1, 2, 3, 4, 5])
        cb.ax.set_yticklabels(["None", "D0", "D1", "D2", "D3", "D4"])
        fig.tight_layout()
        return fig

    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(_extent_from_latlon(lat2d, lonp, pad_deg=pad_deg), crs=ccrs.PlateCarree())

    m = ax.pcolormesh(lonp, lat2d, cat, shading="auto", cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.STATES, linewidth=0.7)
    ax.coastlines(linewidth=0.8)

    ax.set_title(title)
    cb = plt.colorbar(m, ax=ax, shrink=0.80, pad=0.03, ticks=[0, 1, 2, 3, 4, 5])
    cb.ax.set_yticklabels(["None", "D0", "D1", "D2", "D3", "D4"])
    plt.tight_layout()
    return fig

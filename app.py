from __future__ import annotations

import numpy as np
import streamlit as st

from config import (
    BASE_DIR,
    FORCING_DIR,
    ELM_ET_PATH,
    ELM_SM_PATH,
    MODEL_GRID_PATH,
    NLDAS_CACHE_DIR,
    PROCESSED_DAILY_CACHE_DIR,
    DEFAULT_COPULA_TAU,
    GPU_CONFIG_SOURCE,
    GPU_DEVICE_ID,
    TIMESCALES,
)
from models import resolve_timescale_model_dir
from inference import run_forecast
from viz import plot_category


st.set_page_config(page_title="ADAPT Drought Forecast", layout="wide")


st.title("ADAPT Drought Forecasting")
st.caption("Configure parameters to run model inference and inspect drought risk at certain coordinates.")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 1.5rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(120, 120, 120, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _norm_lon_180_scalar(lon: float) -> float:
    return float(lon - 360.0) if lon > 180.0 else float(lon)


def drought_risk_at_coordinate(
    lat: float,
    lon: float,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    pdry: np.ndarray,
    category: np.ndarray,
) -> dict:
    """
    Drought risk inspection for a user-specified coordinate.
    """
    latg = np.asarray(lat2d, dtype=np.float32)
    long = np.asarray(lon2d, dtype=np.float32)
    p = np.asarray(pdry, dtype=np.float32)
    cat = np.asarray(category, dtype=np.int8)

    lon_user = _norm_lon_180_scalar(float(lon))
    lon_grid = np.where(long > 180.0, long - 360.0, long)

    dlat = latg - float(lat)
    dlon = np.abs(lon_grid - lon_user)
    dlon = np.minimum(dlon, 360.0 - dlon)
    d2 = dlat * dlat + dlon * dlon

    i, j = np.unravel_index(np.nanargmin(d2), d2.shape)
    p_here = float(p[i, j])
    cat_here = int(cat[i, j])

    cat_label = {0: "None", 1: "D0", 2: "D1", 3: "D2", 4: "D3", 5: "D4"}.get(cat_here, "None")
    risk_label = ["Normal", "Abnormally Dry", "Moderate Drought", "Severe Drought", "Extreme Drought", "Exceptional Drought"][np.clip(cat_here, 0, 5)]

    return {
        "grid_lat": float(latg[i, j]),
        "grid_lon": float(lon_grid[i, j]),
        "pdry": p_here,
        "pdry_pct": 100.0 * p_here,
        "category": cat_here,
        "category_label": cat_label,
        "risk_label": risk_label,
    }


def _render_drought_legend_table() -> None:
    st.markdown(
        """
        <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
          <thead>
            <tr>
              <th style="text-align:left; padding:4px 6px;">Category</th>
              <th style="text-align:left; padding:4px 6px;">Label</th>
              <th style="text-align:left; padding:4px 6px;">Typical Map Color</th>
              <th style="text-align:left; padding:4px 6px;">Hex Color</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="padding:4px 6px;">None / Normal</td>
              <td style="padding:4px 6px;">&ndash;</td>
              <td style="padding:4px 6px;">White</td>
              <td style="padding:4px 6px;">
                <code>#FFFFFF</code>
              </td>
            </tr>
            <tr>
              <td style="padding:4px 6px;">D0</td>
              <td style="padding:4px 6px;">Abnormally Dry</td>
              <td style="padding:4px 6px;">Yellow</td>
              <td style="padding:4px 6px;">
                <code>#FFFF00</code>
              </td>
            </tr>
            <tr>
              <td style="padding:4px 6px;">D1</td>
              <td style="padding:4px 6px;">Moderate Drought</td>
              <td style="padding:4px 6px;">Light Tan/Orange</td>
              <td style="padding:4px 6px;">
                <code>#FCD37F</code>
              </td>
            </tr>
            <tr>
              <td style="padding:4px 6px;">D2</td>
              <td style="padding:4px 6px;">Severe Drought</td>
              <td style="padding:4px 6px;">Orange</td>
              <td style="padding:4px 6px;">
                <code>#FFAA00</code>
              </td>
            </tr>
            <tr>
              <td style="padding:4px 6px;">D3</td>
              <td style="padding:4px 6px;">Extreme Drought</td>
              <td style="padding:4px 6px;">Red</td>
              <td style="padding:4px 6px;">
                <code>#E60000</code>
              </td>
            </tr>
            <tr>
              <td style="padding:4px 6px;">D4</td>
              <td style="padding:4px 6px;">Exceptional Drought</td>
              <td style="padding:4px 6px;">Dark Red / Maroon</td>
              <td style="padding:4px 6px;">
                <code>#730000</code>
              </td>
            </tr>
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.subheader("Forecast Settings")
    st.caption(f"GPU: `{GPU_DEVICE_ID}` ({GPU_CONFIG_SOURCE})")

    timescale = st.selectbox("Timescale", list(TIMESCALES.keys()), index=0)
    drought_sensitivity = 1.0

    fallback_day = (np.datetime64("today", "D") - np.timedelta64(3, "D")).astype("datetime64[D]")
    default_as_of = fallback_day
    default_as_of_date = np.asarray(default_as_of).astype("datetime64[D]").item()
    as_of_day = st.date_input("As-of day (last forcing day)", value=default_as_of_date)

    run_btn = st.button("Run Forecast", use_container_width=True)

if run_btn:
    spec = TIMESCALES[timescale]

    model_dir = resolve_timescale_model_dir(BASE_DIR, spec.parent_dirs, spec.best_arch_folder)

    bundle = {
        "model_dir": model_dir,
        "forcing_dir": FORCING_DIR,
        "et_path": ELM_ET_PATH,
        "sm_path": ELM_SM_PATH,
        "model_grid_path": MODEL_GRID_PATH,
        "cache_dir": NLDAS_CACHE_DIR,
        "processed_cache_dir": PROCESSED_DAILY_CACHE_DIR,
        "horizon_days": spec.horizon_days,
        "default_history_days": spec.default_history_days,
    }

    with st.spinner("Fetching NLDAS forcings, running model inference, and calculating drought indices"):
        res = run_forecast(
            bundle=bundle,
            timescale=timescale,
            as_of_day=as_of_day,
            tau=DEFAULT_COPULA_TAU,
            drought_sensitivity=drought_sensitivity,
        )

    st.session_state["forecast_result"] = res

res = st.session_state.get("forecast_result")
if res is not None:
    st.success(f"Done. Target day = {str(res.target_day)} (horizon = {res.horizon_days} days).")

    # approximates ER95 from standardized forecast anomalies
    er95_sm = float(np.mean(np.abs(np.asarray(res.z_sm, dtype=np.float32)) > 1.96))
    er95_et = float(np.mean(np.abs(np.asarray(res.z_et, dtype=np.float32)) > 1.96))
    er95_overall = max(er95_sm, er95_et)
    reliability_pct = 100.0 * (1.0 - er95_overall)

    st.subheader("Forecast Output")
    fig = plot_category(
        res.lat2d,
        res.lon2d,
        res.category,
        title="Drought Risk Category Map",
    )
    fig.set_size_inches(8.2, 4.8)
    map_col, side_col = st.columns([2.2, 1.0], gap="large")
    with map_col:
        st.pyplot(fig, clear_figure=True, use_container_width=True)

    lat_min = float(np.nanmin(res.lat2d))
    lat_max = float(np.nanmax(res.lat2d))
    lon_180 = np.where(np.asarray(res.lon2d) > 180.0, np.asarray(res.lon2d) - 360.0, np.asarray(res.lon2d))
    lon_min = float(np.nanmin(lon_180))
    lon_max = float(np.nanmax(lon_180))

    with side_col:
        st.metric("Forecast Reliability", f"{reliability_pct:.1f}%")
        st.caption("Defined as 1 - ER95 exceedance rate.")

        st.markdown("**Coordinate Check**")
        with st.form("point_risk_form"):
            q_lat = st.number_input(
                "Latitude",
                min_value=lat_min,
                max_value=lat_max,
                value=(lat_min + lat_max) / 2.0,
            )
            q_lon = st.number_input(
                "Longitude",
                min_value=lon_min,
                max_value=lon_max,
                value=(lon_min + lon_max) / 2.0,
            )
            check_btn = st.form_submit_button("Check Coordinate", use_container_width=True)

        if check_btn:
            risk = drought_risk_at_coordinate(
                lat=float(q_lat),
                lon=float(q_lon),
                lat2d=res.lat2d,
                lon2d=res.lon2d,
                pdry=res.pdry,
                category=res.category,
            )
            st.session_state["point_risk"] = risk

        risk = st.session_state.get("point_risk")
        if risk is not None:
            st.metric("Drought Risk", risk["risk_label"])
            st.write(
                f"Grid: ({risk['grid_lat']:.3f}, {risk['grid_lon']:.3f}) | "
                f"{risk['category_label']} | Dryness: {risk['pdry_pct']:.1f}%"
            )

        st.markdown("**Drought Categories**")
        _render_drought_legend_table()
else:
    st.info("Select forecast settings in the left panel and press Run Forecast.")

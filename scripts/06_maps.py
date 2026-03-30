#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path
import geopandas as gpd
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# Allow importing config.py from repository root
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    CATALOG_CLEAN,
    DATA_INTERMEDIATE,
    FIG_PROJECTED,
    TULE_POLYGON,
    FAULT_ENDPOINTS,
    MAP_LAT_RANGE,
    MAP_LON_RANGE,
    MAP_DEPTH_RANGE,
    MAP_HEIGHT,
    MAP_ZOOM,
    MAP_STYLE,
    MAP_ANIMATION_HTML,
    MAP_ANIMATION_DAILY_CSV,
)

def build_rolling_animation_table(df: pd.DataFrame, window_days: int = 5) -> pd.DataFrame:
    """
    Build animation frames with a rolling temporal window.
    Each frame contains only events from the previous `window_days`.
    """
    df = df.sort_values("datetime").copy()
    df["date"] = df["datetime"].dt.floor("D")

    unique_dates = sorted(df["date"].dropna().unique())
    frames = []

    for current_date in unique_dates:
        start_date = current_date - pd.Timedelta(days=window_days - 1)
        subset = df[(df["date"] >= start_date) & (df["date"] <= current_date)].copy()
        subset["frame_date"] = current_date.strftime("%Y-%m-%d")
        frames.append(subset)

    if not frames:
        raise ValueError("No valid dates were found to build animation frames.")

    return pd.concat(frames, ignore_index=True)

def build_cumulative_animation_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the catalog so each animation frame contains
    all events up to that day (cumulative animation).
    """
    df = df.sort_values("datetime").copy()
    df["date"] = df["datetime"].dt.date

    unique_dates = sorted(df["date"].dropna().unique())
    frames = []

    for current_date in unique_dates:
        subset = df[df["date"] <= current_date].copy()
        subset["frame_date"] = str(current_date)
        frames.append(subset)

    if not frames:
        raise ValueError("No valid dates were found to build animation frames.")

    return pd.concat(frames, ignore_index=True)


def make_animated_map(df_anim: pd.DataFrame):
    """
    Create animated cumulative seismicity map using the reloc catalog as-is.
    """
    center_lat = (MAP_LAT_RANGE[0] + MAP_LAT_RANGE[1]) / 2
    center_lon = (MAP_LON_RANGE[0] + MAP_LON_RANGE[1]) / 2

    fig = px.scatter_mapbox(
        df_anim,
        lat="latitude",
        lon="longitude",
        color="depth_km",
        size="magnitude",
        animation_frame="frame_date",
        hover_data={
            "datetime": True,
            "event_id": True,
            "depth_km": ":.2f",
            "magnitude": ":.2f",
            "err_h_m": ":.2f",
            "err_x_m": ":.2f",
            "err_y_m": ":.2f",
            "err_z_m": ":.2f",
            "latitude": ":.5f",
            "longitude": ":.5f",
        },
        range_color=list(MAP_DEPTH_RANGE),
        color_continuous_scale="bluered",
        size_max=8,
        zoom=MAP_ZOOM,
        height=MAP_HEIGHT,
        mapbox_style=MAP_STYLE,
        title="Los Cabos seismic sequence (5-day rolling window)"
    )

    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        mapbox=dict(
            center={"lat": center_lat, "lon": center_lon},
            zoom=MAP_ZOOM,
            bounds={
                "west": MAP_LON_RANGE[0],
                "east": MAP_LON_RANGE[1],
                "south": MAP_LAT_RANGE[0],
                "north": MAP_LAT_RANGE[1],
            },
        ),
        coloraxis_colorbar=dict(title="Depth (km)"),
    )

    return fig


def main() -> None:
    import geopandas as gpd
    import plotly.graph_objects as go

    DATA_INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    FIG_PROJECTED.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Read clean hypoDD catalog
    # -----------------------------
    df = pd.read_csv(CATALOG_CLEAN)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Keep only valid rows for mapping
    df = df.dropna(subset=["datetime", "latitude", "longitude"]).copy()

    # Temporal filter
    df = df[
        (df["datetime"] >= "2024-08-01") &
        (df["datetime"] <= "2025-02-11 23:59:59")
    ].copy()

    # Magnitude fallback only if needed
    df["magnitude"] = df["magnitude"].fillna(1.0)

    # -----------------------------
    # Build rolling 5-day animation
    # -----------------------------
    df_anim = build_cumulative_animation_table(df)
  #  df_anim = build_rolling_animation_table(df, window_days=5)

    # Save expanded table used for animation
    df_anim.to_csv(
        MAP_ANIMATION_DAILY_CSV,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S.%f",
    )

    # -----------------------------
    # Create animated map
    # -----------------------------
    print("Construyendo figura...")
    fig = make_animated_map(df_anim)

    # =========================================================
    # Add permanent polygon(s) from shallowpoli.shp
    # =========================================================
    print("Añadiendo polígonos permanentes...")
    gdf_poly = gpd.read_file(TULE_POLYGON)

    # Plotly map needs lon/lat in EPSG:4326
    if gdf_poly.crs is not None and str(gdf_poly.crs) != "EPSG:4326":
        gdf_poly = gdf_poly.to_crs("EPSG:4326")

    polygon_legend_shown = False

    for _, row in gdf_poly.iterrows():
        geom = row.geometry

        if geom is None:
            continue

        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            continue

        for poly in polygons:
            x, y = poly.exterior.xy

            fig.add_trace(
                go.Scattermapbox(
                    lon=list(x),
                    lat=list(y),
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(0,150,0,0.15)",
                    line=dict(color="green", width=2),
                    name="Shallow polygon",
                    hoverinfo="skip",
                    showlegend=not polygon_legend_shown,
                )
            )
            polygon_legend_shown = True

    # =========================================================
    # Add permanent line from extremosnodes.csv
    # =========================================================
    print("Añadiendo línea permanente...")
    df_line = pd.read_csv(FAULT_ENDPOINTS)

    # Try to infer lon/lat column names robustly
    cols_lower = {c.lower(): c for c in df_line.columns}

    lon_col = None
    lat_col = None

    lon_candidates = ["longitude", "lon", "x", "long"]
    lat_candidates = ["latitude", "lat", "y"]

    for c in lon_candidates:
        if c in cols_lower:
            lon_col = cols_lower[c]
            break

    for c in lat_candidates:
        if c in cols_lower:
            lat_col = cols_lower[c]
            break

    if lon_col is None or lat_col is None:
        raise ValueError(
            f"No se pudieron identificar columnas lon/lat en {FAULT_ENDPOINTS}. "
            f"Columnas encontradas: {list(df_line.columns)}"
        )

    df_line = df_line.dropna(subset=[lon_col, lat_col]).copy()

    fig.add_trace(
        go.Scattermapbox(
            lon=df_line[lon_col].tolist(),
            lat=df_line[lat_col].tolist(),
            mode="lines+markers",
            line=dict(color="black", width=3),
            marker=dict(size=6, color="black"),
            name="Fault / line",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    # -----------------------------
    # Save HTML
    # -----------------------------
    print("Guardando HTML...")
    fig.write_html(MAP_ANIMATION_HTML)

    # -----------------------------
    # Summary
    # -----------------------------
    print("\nMap summary")
    print("-" * 50)
    print(f"Input catalog:                 {CATALOG_CLEAN}")
    print(f"Mapped events:                {len(df):,}")
    print(f"First event:                  {df['datetime'].min()}")
    print(f"Last event:                   {df['datetime'].max()}")
    print(f"Animation frames:             {df_anim['frame_date'].nunique():,}")
    print(f"Expanded animation CSV:       {MAP_ANIMATION_DAILY_CSV}")
    print(f"Interactive HTML map:         {MAP_ANIMATION_HTML}")


if __name__ == "__main__":
    main()

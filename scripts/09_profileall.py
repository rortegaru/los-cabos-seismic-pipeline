#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

# Allow importing config.py from repository root
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    CATALOG_PAPER,        # full paper catalog
    FAULT_ENDPOINTS,      # CSV with 2 points: x,y (projected CRS)
    CRS_EVENTS,           # e.g. "EPSG:4326"
    CRS_WORK,             # projected CRS, e.g. UTM
    RAIN_DATE,
)

# ------------------------------------------------------------
# USER OUTPUTS
# ------------------------------------------------------------
OUT_ALL = ROOT_DIR / "intermediate" / "10_catalog_sanjose_projection.csv"
OUT_ONSEG = ROOT_DIR / "intermediate" / "10_catalog_sanjose_onsegment.csv"
OUT_NEAR = ROOT_DIR / "intermediate" / "10_catalog_sanjose_nearfault.csv"

# distance threshold for "near-fault" subset
NEAR_FAULT_THRESHOLD_M = 1000.0  # change if needed


def read_fault_endpoints(csv_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the two fault endpoints from CSV.
    Expected columns: x, y
    Optional column: id
    """
    df = pd.read_csv(csv_file)

    if len(df) != 2:
        raise ValueError("FAULT_ENDPOINTS must contain exactly 2 points.")

    if "id" in df.columns:
        df = df.sort_values("id")

    p1 = np.array([df.iloc[0]["x"], df.iloc[0]["y"]], dtype=float)
    p2 = np.array([df.iloc[1]["x"], df.iloc[1]["y"]], dtype=float)

    return p1, p2


def unit_vector(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector is not allowed.")
    return v / n


def main() -> None:
    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1) Read full catalog
    # ------------------------------------------------------------
    df = pd.read_csv(CATALOG_PAPER, parse_dates=["datetime"])

    required = {"longitude", "latitude", "datetime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input catalog: {missing}")

    # ------------------------------------------------------------
    # 2) Convert to projected GeoDataFrame
    # ------------------------------------------------------------
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=CRS_EVENTS,
    ).to_crs(CRS_WORK)

    gdf["utm_x_m"] = gdf.geometry.x
    gdf["utm_y_m"] = gdf.geometry.y

    # ------------------------------------------------------------
    # 3) Read fault geometry
    # ------------------------------------------------------------
    p1, p2 = read_fault_endpoints(FAULT_ENDPOINTS)

    v_fault = p2 - p1
    L_fault_m = np.linalg.norm(v_fault)
    u_fault = unit_vector(v_fault)

    # Right-hand perpendicular
    u_perp = np.array([u_fault[1], -u_fault[0]])

    # ------------------------------------------------------------
    # 4) Projection
    #    Reference point = P1 (more intuitive than midpoint)
    # ------------------------------------------------------------
    pts = gdf[["utm_x_m", "utm_y_m"]].to_numpy()
    w = pts - p1

    # Distance along fault measured from P1
    along_fault_from_p1_m = w @ u_fault

    # Signed perpendicular distance
    r_proj_m = w @ u_perp

    # Absolute perpendicular distance
    dist_to_fault_m = np.abs(r_proj_m)

    # Does orthogonal projection fall inside the finite fault segment?
    on_segment_projection = (
        (along_fault_from_p1_m >= 0.0) &
        (along_fault_from_p1_m <= L_fault_m)
    )

    # Side of fault according to sign of r_proj_m
    side_of_fault = np.where(
        r_proj_m > 0, "right",
        np.where(r_proj_m < 0, "left", "on_axis")
    )

    # Projected point coordinates on the fault axis
    proj_xy = p1 + np.outer(along_fault_from_p1_m, u_fault)
    gdf["proj_x_m"] = proj_xy[:, 0]
    gdf["proj_y_m"] = proj_xy[:, 1]

    # Distance to nearest point of the finite segment
    along_clipped = np.clip(along_fault_from_p1_m, 0.0, L_fault_m)
    closest_xy = p1 + np.outer(along_clipped, u_fault)
    d_to_segment_m = np.sqrt(
        (pts[:, 0] - closest_xy[:, 0]) ** 2 +
        (pts[:, 1] - closest_xy[:, 1]) ** 2
    )

    # ------------------------------------------------------------
    # 5) Time since rainfall
    # ------------------------------------------------------------
    gdf["t_days_since_rain"] = (
        gdf["datetime"] - RAIN_DATE
    ).dt.total_seconds() / 86400.0

    # ------------------------------------------------------------
    # 6) Store columns
    # ------------------------------------------------------------
    gdf["along_fault_from_p1_m"] = along_fault_from_p1_m
    gdf["along_fault_from_p1_km"] = along_fault_from_p1_m / 1000.0

    gdf["r_proj_m"] = r_proj_m
    gdf["r_proj_km"] = r_proj_m / 1000.0

    gdf["dist_to_fault_m"] = dist_to_fault_m
    gdf["dist_to_fault_km"] = dist_to_fault_m / 1000.0

    gdf["distance_to_segment_m"] = d_to_segment_m
    gdf["distance_to_segment_km"] = d_to_segment_m / 1000.0

    gdf["on_segment_projection"] = on_segment_projection
    gdf["side_of_fault"] = side_of_fault

    gdf["fault_length_m"] = L_fault_m
    gdf["fault_length_km"] = L_fault_m / 1000.0

    # ------------------------------------------------------------
    # 7) Save outputs
    # ------------------------------------------------------------
    out_df = pd.DataFrame(gdf.drop(columns="geometry"))
    out_df = out_df.sort_values(["datetime", "event_id"]).reset_index(drop=True)
    out_df.to_csv(OUT_ALL, index=False)

    out_onseg = out_df[out_df["on_segment_projection"]].copy()
    out_onseg.to_csv(OUT_ONSEG, index=False)

    out_near = out_df[
        (out_df["on_segment_projection"]) &
        (out_df["dist_to_fault_m"] <= NEAR_FAULT_THRESHOLD_M)
    ].copy()
    out_near.to_csv(OUT_NEAR, index=False)

    # ------------------------------------------------------------
    # 8) Summary
    # ------------------------------------------------------------
    print("\nAll-event San José fault projection summary")
    print("-" * 60)
    print(f"Input catalog:                    {CATALOG_PAPER}")
    print(f"Fault endpoints file:             {FAULT_ENDPOINTS}")
    print(f"Projected events:                 {len(out_df):,}")
    print(f"Events on segment projection:     {len(out_onseg):,}")
    print(f"Events near fault ({NEAR_FAULT_THRESHOLD_M:.0f} m): {len(out_near):,}")
    print(f"First event:                      {out_df['datetime'].min()}")
    print(f"Last event:                       {out_df['datetime'].max()}")
    print(f"Fault length:                     {L_fault_m/1000:.3f} km")
    print(f"Min along_fault_from_p1_km:       {out_df['along_fault_from_p1_km'].min():.3f}")
    print(f"Max along_fault_from_p1_km:       {out_df['along_fault_from_p1_km'].max():.3f}")
    print(f"Min r_proj_km:                    {out_df['r_proj_km'].min():.3f}")
    print(f"Max r_proj_km:                    {out_df['r_proj_km'].max():.3f}")
    print(f"Max dist_to_fault_km:             {out_df['dist_to_fault_km'].max():.3f}")

    print("\nSaved files")
    print("-" * 60)
    print(f"All projected catalog:            {OUT_ALL}")
    print(f"On-segment subset:                {OUT_ONSEG}")
    print(f"Near-fault subset:                {OUT_NEAR}")

    print("\nGeometry used")
    print("-" * 60)
    print(f"P1 = {p1}")
    print(f"P2 = {p2}")
    print(f"u_fault = {u_fault}")
    print(f"u_perp  = {u_perp}")
    print(f"Dot(u_fault, u_perp) = {np.dot(u_fault, u_perp):.6f}")


if __name__ == "__main__":
    main()

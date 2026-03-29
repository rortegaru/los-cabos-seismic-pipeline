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
    CATALOG_TULE,
    FAULT_ENDPOINTS,
    CATALOG_PROJ,
    CRS_EVENTS,
    CRS_WORK,
    RAIN_DATE,
)


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


def unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Zero-length vector is not allowed.")
    return vector / norm


def main() -> None:
    # Ensure output directory exists
    CATALOG_PROJ.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1) Read El Tule subset
    # -----------------------------
    df = pd.read_csv(CATALOG_TULE, parse_dates=["datetime"])

    required = {"longitude", "latitude", "datetime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input catalog: {missing}")

    # -----------------------------
    # 2) Convert catalog to GeoDataFrame and project to working CRS
    # -----------------------------
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=CRS_EVENTS,
    ).to_crs(CRS_WORK)

    gdf["utm_x_m"] = gdf.geometry.x
    gdf["utm_y_m"] = gdf.geometry.y

    # -----------------------------
    # 3) Read fault endpoints
    # -----------------------------
    p1, p2 = read_fault_endpoints(FAULT_ENDPOINTS)

    # Midpoint used as reference point
    p0 = 0.5 * (p1 + p2)

    # Fault-direction unit vector
    v_fault = p2 - p1
    u_fault = unit_vector(v_fault)

    # Perpendicular unit vector
    u_perp = np.array([u_fault[1], -u_fault[0]])

    # -----------------------------
    # 4) Project each event
    # -----------------------------
    pts = gdf[["utm_x_m", "utm_y_m"]].to_numpy()
    w = pts - p0

    # Coordinate along the fault
    along_fault_m = w @ u_fault

    # Signed perpendicular projection
    r_proj_m = w @ u_perp

    # Absolute perpendicular distance to fault axis
    dist_to_fault_m = np.abs(r_proj_m)

    # Euclidean distance from reference midpoint
    d_euclid_ref_m = np.sqrt((pts[:, 0] - p0[0]) ** 2 + (pts[:, 1] - p0[1]) ** 2)

    gdf["along_fault_m"] = along_fault_m
    gdf["r_proj_m"] = r_proj_m
    gdf["dist_to_fault_m"] = dist_to_fault_m
    gdf["d_euclid_ref_m"] = d_euclid_ref_m

    # -----------------------------
    # 5) Time since rainfall
    # -----------------------------
    gdf["t_days_since_rain"] = (
        gdf["datetime"] - RAIN_DATE
    ).dt.total_seconds() / 86400.0

    # -----------------------------
    # 6) Save output
    # -----------------------------
    out_df = pd.DataFrame(gdf.drop(columns="geometry"))
    out_df = out_df.sort_values(["datetime", "event_id"]).reset_index(drop=True)
    out_df.to_csv(CATALOG_PROJ, index=False)

    # -----------------------------
    # 7) Summary
    # -----------------------------
    print("\nFault-projection summary")
    print("-" * 50)
    print(f"Input El Tule subset:           {CATALOG_TULE}")
    print(f"Fault endpoints file:           {FAULT_ENDPOINTS}")
    print(f"Events projected:               {len(out_df):,}")
    print(f"First event:                    {out_df['datetime'].min()}")
    print(f"Last event:                     {out_df['datetime'].max()}")
    print(f"Minimum r_proj_m:               {out_df['r_proj_m'].min():.2f}")
    print(f"Maximum r_proj_m:               {out_df['r_proj_m'].max():.2f}")
    print(f"Maximum dist_to_fault_m:        {out_df['dist_to_fault_m'].max():.2f}")
    print(f"Maximum d_euclid_ref_m:         {out_df['d_euclid_ref_m'].max():.2f}")

    print(f"\nSaved projected catalog to:     {CATALOG_PROJ}")

    print("\nGeometry used")
    print("-" * 50)
    print(f"P1 = {p1}")
    print(f"P2 = {p2}")
    print(f"P0 = {p0}")
    print(f"u_fault = {u_fault}")
    print(f"u_perp  = {u_perp}")
    print(f"Dot product (should be ~0):     {np.dot(u_fault, u_perp):.6f}")


if __name__ == "__main__":
    main()

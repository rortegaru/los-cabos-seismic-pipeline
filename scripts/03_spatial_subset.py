#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

# Allow importing config.py from repository root
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CATALOG_PAPER, TULE_POLYGON, CATALOG_TULE_FLAG, CATALOG_TULE, CRS_WORK


def main() -> None:
    # Ensure output directory exists
    CATALOG_TULE_FLAG.parent.mkdir(parents=True, exist_ok=True)

    # Read paper catalog
    df = pd.read_csv(CATALOG_PAPER, parse_dates=["datetime"])

    # Events in geographic coordinates (WGS84)
    gdf_events = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    # Reproject events to working CRS
    gdf_events = gdf_events.to_crs(CRS_WORK)

    # Read polygon shapefile
    gdf_polygon = gpd.read_file(TULE_POLYGON)

    if gdf_polygon.crs is None:
        raise ValueError("The polygon shapefile has no defined CRS.")

    if gdf_polygon.crs.to_string() != CRS_WORK:
        gdf_polygon = gdf_polygon.to_crs(CRS_WORK)

    # Merge all polygon parts into a single geometry
    polygon_union = gdf_polygon.union_all()

    # Spatial flag: inside or touching polygon boundary
    gdf_events["in_tule"] = (
        gdf_events.geometry.within(polygon_union)
        | gdf_events.geometry.intersects(polygon_union)
    )

    # Store projected coordinates
    gdf_events["utm_x_m"] = gdf_events.geometry.x
    gdf_events["utm_y_m"] = gdf_events.geometry.y

    # Save full flagged catalog
    df_flagged = pd.DataFrame(gdf_events.drop(columns="geometry"))
    df_flagged = df_flagged.sort_values(["datetime", "event_id"]).reset_index(drop=True)
    df_flagged.to_csv(CATALOG_TULE_FLAG, index=False)

    # Save El Tule subset
    df_tule = df_flagged[df_flagged["in_tule"]].copy()
    df_tule = df_tule.sort_values(["datetime", "event_id"]).reset_index(drop=True)
    df_tule.to_csv(CATALOG_TULE, index=False)

    # Summary
    print("\nSpatial subset summary: El Tule polygon")
    print("-" * 50)
    print(f"Input paper catalog:            {CATALOG_PAPER}")
    print(f"Polygon shapefile:              {TULE_POLYGON}")
    print(f"Total events evaluated:         {len(df_flagged):,}")
    print(f"Events inside El Tule:          {len(df_tule):,}")
    print(f"Retained percentage:            {100 * len(df_tule) / len(df_flagged):.2f}%")

    if len(df_tule) > 0:
        print(f"First El Tule event:            {df_tule['datetime'].min()}")
        print(f"Last El Tule event:             {df_tule['datetime'].max()}")

    print(f"\nSaved flagged catalog to:       {CATALOG_TULE_FLAG}")
    print(f"Saved El Tule subset to:        {CATALOG_TULE}")


if __name__ == "__main__":
    main()

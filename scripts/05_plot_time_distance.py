#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Allow importing config.py from repository root
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CATALOG_PROJ, FIG_PROJECTED, FIG_EUCLIDEAN


def main() -> None:
    # Ensure output directory exists
    FIG_PROJECTED.parent.mkdir(parents=True, exist_ok=True)

    # Read projected catalog
    df = pd.read_csv(CATALOG_PROJ, parse_dates=["datetime"])

    # Convert distances from meters to kilometers
    df["r_proj_km"] = df["r_proj_m"] / 1000.0
    df["d_euclid_ref_km"] = df["d_euclid_ref_m"] / 1000.0

    # -------- Plot 1: projected distance --------
    plt.figure(figsize=(8, 5))
    plt.scatter(df["t_days_since_rain"], df["r_proj_km"], s=12)

    plt.xlim(0, 300)
    plt.ylim(0, 3.4)

    plt.xlabel("Days since rainfall event")
    plt.ylabel("Projected distance normal to fault (km)")
    plt.title("El Tule subset: projected time–distance plot")
    plt.tight_layout()
    plt.savefig(FIG_PROJECTED, dpi=300)
    plt.close()

    # -------- Plot 2: Euclidean distance --------
    plt.figure(figsize=(8, 5))
    plt.scatter(df["t_days_since_rain"], df["d_euclid_ref_km"], s=12)

    plt.xlim(0, 300)
    plt.ylim(0, 3.4)

    plt.xlabel("Days since rainfall event")
    plt.ylabel("Euclidean distance from reference point (km)")
    plt.title("El Tule subset: Euclidean time–distance plot")
    plt.tight_layout()
    plt.savefig(FIG_EUCLIDEAN, dpi=300)
    plt.close()

    # Summary
    print("\nPlot generation summary")
    print("-" * 50)
    print(f"Input projected catalog:        {CATALOG_PROJ}")
    print(f"Number of plotted events:       {len(df):,}")
    print(f"Saved projected plot to:        {FIG_PROJECTED}")
    print(f"Saved Euclidean plot to:        {FIG_EUCLIDEAN}")


if __name__ == "__main__":
    main()

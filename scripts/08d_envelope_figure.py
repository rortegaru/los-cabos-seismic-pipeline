#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Import config
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    CATALOG_PROJ,
    DATA_INTERMEDIATE,
    FIGURES_DIR,
    PLOT_X_MIN,
    PLOT_X_MAX,
    PLOT_Y_MIN,
    PLOT_Y_MAX,
)

# ---------------------------------------------------------
# Files
# ---------------------------------------------------------
CSV_ALL = CATALOG_PROJ
CSV_SEL = DATA_INTERMEDIATE / "selected_envelope_points.csv"

OUT_FIG = FIGURES_DIR / "darcy_envelope_final.png"
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
def load_data():
    df_all = pd.read_csv(CSV_ALL)
    df_sel = pd.read_csv(CSV_SEL)

    df_all["datetime"] = pd.to_datetime(df_all["datetime"], errors="coerce")
    df_sel["datetime"] = pd.to_datetime(df_sel["datetime"], errors="coerce")

    df_all["t_days"] = df_all["t_days_since_rain"]
    df_all["distance_km"] = np.abs(df_all["r_proj_m"]) / 1000.0

    return df_all, df_sel


# ---------------------------------------------------------
def diffusion_curve(t_days, D):
    t_sec = t_days * 86400.0
    r_m = np.sqrt(4 * D * t_sec)
    return r_m / 1000.0


# ---------------------------------------------------------
def compute_D_stats(df_sel):
    d = df_sel["Dmin_m2s"].values

    return {
        "D90": np.percentile(d, 90),
        "D95": np.percentile(d, 95),
        "D100": np.max(d),
    }


# ---------------------------------------------------------
def plot_figure(df_all, df_sel, stats):

    plt.figure(figsize=(10, 6))

    # -------------------------
    # 1. All events (background)
    # -------------------------
    plt.scatter(
        df_all["t_days"],
        df_all["distance_km"],
        s=15,
        alpha=0.25,
        label="Seismicity",
    )

    # -------------------------
    # 2. Selected envelope
    # -------------------------
    plt.scatter(
        df_sel["t_days_since_rain"],
        df_sel["distance_km"],
        s=80,
        edgecolor="black",
        linewidth=0.5,
        label="Envelope points",
        zorder=3
    )

    # -------------------------
    # 3. Diffusion curves
    # -------------------------
    t = np.linspace(0.01, PLOT_X_MAX, 500)

    for name, D in stats.items():
        r = diffusion_curve(t, D)

        plt.plot(
            t,
            r,
            linewidth=2.5,
            label=f"{name} = {D:.3f} m²/s",
        )

    # -------------------------
    # Axis
    # -------------------------
    plt.xlim(PLOT_X_MIN, PLOT_X_MAX)
    plt.ylim(PLOT_Y_MIN, PLOT_Y_MAX)

    plt.xlabel("Time since rainfall (days)", fontsize=12)
    plt.ylabel("Distance along fault (km)", fontsize=12)

    plt.title("Diffusion envelope of seismic migration", fontsize=13)

    plt.grid(True, alpha=0.2)

    # -------------------------
    # Legend (clean)
    # -------------------------
    plt.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    plt.close()

    print(f"\nFigura guardada en:\n{OUT_FIG}")


# ---------------------------------------------------------
def main():

    df_all, df_sel = load_data()

    stats = compute_D_stats(df_sel)

    print("\n=== D values ===")
    for k, v in stats.items():
        print(f"{k}: {v:.4f} m²/s  ({v*86400:.0f} m²/day)")

    plot_figure(df_all, df_sel, stats)


# ---------------------------------------------------------
if __name__ == "__main__":
    main()

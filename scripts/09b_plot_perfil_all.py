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

# ------------------------------------------------------------
# INPUT / OUTPUT
# ------------------------------------------------------------
#INFILE = ROOT_DIR / "intermediate" / "10_catalog_sanjose_onsegment.csv"
# Alternative:
INFILE = ROOT_DIR / "intermediate" / "10_catalog_sanjose_nearfault.csv"

OUTFIG = ROOT_DIR / "figures" / "11_perfil_sanjose.svg"

# ------------------------------------------------------------
# PLOT OPTIONS
# ------------------------------------------------------------
USE_COLOR_BY_TIME = True
SHOW_ZERO_LINE = True
POINT_SIZE = 28
ALPHA = 0.80
DEPTH_POSITIVE_DOWN = True


def main() -> None:
    if not INFILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada:\\n{INFILE}\\n"
            "Primero corre 10_allprojection.py"
        )

    OUTFIG.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INFILE, parse_dates=["datetime"])

    required = {"r_proj_km", "depth_km"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    if USE_COLOR_BY_TIME and "t_days_since_rain" in df.columns:
        df = df.sort_values("t_days_since_rain").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.invert_xaxis()

    if USE_COLOR_BY_TIME and "t_days_since_rain" in df.columns:
        sc = ax.scatter(
            df["r_proj_km"],
            df["depth_km"],
            c=df["t_days_since_rain"],
            s=POINT_SIZE,
            alpha=ALPHA,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Days since rain")
    else:
        ax.scatter(
            df["r_proj_km"],
            df["depth_km"],
            s=POINT_SIZE,
            alpha=ALPHA,
        )

    if SHOW_ZERO_LINE:
        ax.axvline(0.0, linewidth=1.0)

    ax.set_xlabel("Perpendicular distance to San José fault, signed (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("Projected seismicity profile perpendicular to the San José fault")

    if DEPTH_POSITIVE_DOWN:
        ax.invert_yaxis()

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(OUTFIG, dpi=300, bbox_inches="tight")
    print(f"Figura guardada en:\\n{OUTFIG}")

    print("\\nResumen")
    print("-" * 40)
    print(f"Eventos: {len(df):,}")
    print(f"r_proj_km min/max: {df['r_proj_km'].min():.3f} / {df['r_proj_km'].max():.3f}")
    print(f"depth_km min/max: {df['depth_km'].min():.3f} / {df['depth_km'].max():.3f}")


if __name__ == "__main__":
    main()

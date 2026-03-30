#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seismostats import Catalog

# ---------------------------------------------------------------------
# Allow importing config.py from repository root
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CATALOG_TULE, FIGURES_DIR


# ---------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------
FIG_GUTENBERG_DIR = FIGURES_DIR / "gutenberg_richter"
FIG_FMD = FIG_GUTENBERG_DIR / "fmd.png"
FIG_MC_VS_B = FIG_GUTENBERG_DIR / "mc_vs_b.png"
FIG_CUM_FMD = FIG_GUTENBERG_DIR / "cum_fmd.png"
FIG_CUM_FMD_B1 = FIG_GUTENBERG_DIR / "cum_fmd_b1.png"


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
MAG_MAX = 2.8
DELTA_M = 0.1
FMD_BIN = 0.1
MC_SEARCH_MIN = 1.0
MC_SEARCH_MAX = 2.8
MC_SEARCH_STEP = 0.1


def load_input_catalog(csv_path: Path) -> pd.DataFrame:
    """
    Read the working CSV catalog and apply minimal cleaning.

    Expected columns in the current catalog:
    datetime, latitude, longitude, depth_km, magnitude
    """
    df = pd.read_csv(csv_path)

    required = ["datetime", "latitude", "longitude", "depth_km", "magnitude"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"The catalog is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    df = df.dropna(
        subset=["datetime", "latitude", "longitude", "depth_km", "magnitude"]
    ).copy()

    df = df[df["magnitude"] <= MAG_MAX].copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def prepare_seismostats_catalog(df: pd.DataFrame) -> Catalog:
    """
    Convert the project dataframe into the column names expected by seismostats.
    """
    df_catalog = pd.DataFrame(
        {
            "lat": df["latitude"].astype(float),
            "long": df["longitude"].astype(float),
            "depth": df["depth_km"].astype(float),
            "magnitude": df["magnitude"].astype(float),
            "time": df["datetime"],
        }
    )

    cat = Catalog(df_catalog)
    cat.delta_m = DELTA_M

    return cat


def estimate_mc_and_b(cat: Catalog) -> tuple[float, object]:
    """
    Estimate magnitude of completeness and b-value.
    """
    mc, _ = cat.estimate_mc_maxc(fmd_bin=FMD_BIN)
    b_estimator = cat.estimate_b(mc=mc)
    return mc, b_estimator


def plot_fmd(cat: Catalog, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    cat.plot_fmd(fmd_bin=FMD_BIN)
    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.close()


def plot_mc_vs_b(cat: Catalog, output_path: Path) -> None:
    mcs = np.arange(MC_SEARCH_MIN, MC_SEARCH_MAX + MC_SEARCH_STEP, MC_SEARCH_STEP)

    plt.figure(figsize=(7, 5))
    cat.plot_mc_vs_b(mcs=mcs)
    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.close()


def plot_cumulative_fmd(
    cat: Catalog,
    mc: float,
    b_value: float,
    output_path: Path,
    mmax: float | None = None,
) -> None:
    cat.mc = mc
    cat.b_value = b_value
    cat.delta_m = DELTA_M

    if mmax is not None:
        cat.mmax = mmax

    plt.figure(figsize=(7, 5))
    cat.plot_cum_fmd()
    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.close()


def main() -> None:
    FIG_GUTENBERG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Read and clean the input catalog
    df = load_input_catalog(CATALOG_TULE)

    # 2) Build seismostats catalog
    cat = prepare_seismostats_catalog(df)

    # 3) Estimate Mc and b-value
    mc, b_estimator = estimate_mc_and_b(cat)

    print("\n=== Gutenberg-Richter summary ===")
    print(f"Input catalog : {CATALOG_TULE}")
    print(f"N events      : {len(df)}")
    print(f"delta_m       : {cat.delta_m:.2f}")
    print(f"Mc            : {mc:.2f}")
    print(f"b-value       : {b_estimator.b_value:.2f} ± {b_estimator.std:.2f}")

    # 4) Standard plots
    plot_fmd(cat, FIG_FMD)
    plot_mc_vs_b(cat, FIG_MC_VS_B)

    # 5) Cumulative FMD using the estimated b-value
    plot_cumulative_fmd(
        cat=cat,
        mc=mc,
        b_value=b_estimator.b_value,
        output_path=FIG_CUM_FMD,
        mmax=float(df["magnitude"].max()),
    )

    # 6) Optional comparison plot using fixed b = 1.0
    plot_cumulative_fmd(
        cat=cat,
        mc=mc,
        b_value=1.0,
        output_path=FIG_CUM_FMD_B1,
        mmax=float(df["magnitude"].max()),
    )

    print("\nSaved figures:")
    print(f"  - {FIG_FMD}")
    print(f"  - {FIG_MC_VS_B}")
    print(f"  - {FIG_CUM_FMD}")
    print(f"  - {FIG_CUM_FMD_B1}")


if __name__ == "__main__":
    main()

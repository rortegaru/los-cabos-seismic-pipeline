#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Allow importing config.py from repository root
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CATALOG_PROJ, FIGURES_DIR


# ---------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------
FIG_DIR = FIGURES_DIR / "darcy"
FIG_PROFILE = FIG_DIR / "profile_depth_distance.png"
FIG_TIME_DISTANCE = FIG_DIR / "time_distance.png"
FIG_DIFFUSION = FIG_DIR / "diffusion_curves.png"


# ---------------------------------------------------------------------
# Physical reference dates
# ---------------------------------------------------------------------
MAINSHOCK_DATE = pd.Timestamp("2024-09-03 00:00:00")
NETWORK_READY_DATE = pd.Timestamp("2024-08-30 00:00:00")
RAIN_DATE = pd.Timestamp("2024-09-14 00:00:00")

# Hydraulic delay before seismic response becomes observable
DELAY_DAYS = 13.0
EFFECTIVE_RAIN_DATE = RAIN_DATE + pd.Timedelta(days=DELAY_DAYS)


# ---------------------------------------------------------------------
# Analysis window
# ---------------------------------------------------------------------
ANALYSIS_END_DATE = pd.Timestamp("2025-02-08 23:59:59")


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
DEPTH_MAX = 10.0

TIME_REFERENCE = "rain"   # options: "rain", "mainshock", "network"

# Time window in hours relative to selected reference
TMIN = 0.0
TMAX = 3500.0

D_VALUES = [0.01, 0.1, 0.3, 1.7, 3.0]

DIFFUSION_CURVES = [
    {"D": 0.01, "t_end": 3500.0, "linestyle": "--", "label": "D = 0.01 m²/s"},
    {"D": 0.10, "t_end": 3000.0, "linestyle": "-",  "label": "D = 0.1 m²/s"},
    {"D": 0.30, "t_end": 1000.0, "linestyle": "-",  "label": "D = 0.3 m²/s"},
    {"D": 1.70, "t_end": 700.0,  "linestyle": "-",  "label": "D = 1.7 m²/s"},
    {"D": 3.00, "t_end": 3500.0, "linestyle": "--", "label": "D = 3.0 m²/s"},
]


# ---------------------------------------------------------------------

def median_profile_regression(df: pd.DataFrame, n_bins: int = 10):
    # Remove NaNs just in case
    df = df.dropna(subset=["distance_km", "depth_km"]).copy()

    # Create bins
    bins = np.linspace(df["distance_km"].min(), df["distance_km"].max(), n_bins + 1)
    df["bin"] = pd.cut(df["distance_km"], bins)

    # Compute median per bin
    grouped = df.groupby("bin").agg(
        distance_med=("distance_km", "median"),
        depth_med=("depth_km", "median"),
        count=("depth_km", "size")
    ).dropna()

    # Remove bins with very few points
    grouped = grouped[grouped["count"] >= 3]

    # Linear fit on medians
    coef = np.polyfit(grouped["distance_med"], grouped["depth_med"], 1)

    return grouped, coef


def debug_dataframe_state(df: pd.DataFrame, label: str) -> None:
    print(f"\n--- DEBUG: {label} ---")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if len(df) == 0:
        print("DataFrame is empty.")
        return

    print("\nFirst 5 rows:")
    print(df.head())

    for col in ["datetime", "r_proj_m", "depth_km", "t_days_since_rain"]:
        if col in df.columns:
            print(f"{col} NaN count: {df[col].isna().sum()}")
        else:
            print(f"{col}: COLUMN NOT FOUND")

    if "datetime" in df.columns:
        print(f"datetime min: {df['datetime'].min()}")
        print(f"datetime max: {df['datetime'].max()}")

    if "r_proj_m" in df.columns:
        print(f"r_proj_m min: {df['r_proj_m'].min()}")
        print(f"r_proj_m max: {df['r_proj_m'].max()}")

    if "depth_km" in df.columns:
        print(f"depth_km min: {df['depth_km'].min()}")
        print(f"depth_km max: {df['depth_km'].max()}")

    if "t_days_since_rain" in df.columns:
        print(f"t_days_since_rain min: {df['t_days_since_rain'].min()}")
        print(f"t_days_since_rain max: {df['t_days_since_rain'].max()}")


# ---------------------------------------------------------------------
def load_catalog(csv_path: Path) -> pd.DataFrame:
    print("\n=== LOADING CATALOG ===")
    print(f"Reading file: {csv_path}")
    print(f"Exists: {csv_path.exists()}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    debug_dataframe_state(df, "raw catalog after read_csv")

    required = ["datetime", "r_proj_m", "depth_km"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["datetime", "r_proj_m", "depth_km"]).copy()
    after = len(df)

    print("\nDropna summary:")
    print(f"Rows before dropna: {before}")
    print(f"Rows after dropna : {after}")
    print(f"Rows removed      : {before - after}")

    df["distance_km"] = df["r_proj_m"] / 1000.0

    debug_dataframe_state(df, "after datetime cleaning and distance conversion")
    return df


# ---------------------------------------------------------------------
def get_reference_time() -> tuple[pd.Timestamp, str]:
    if TIME_REFERENCE == "rain":
        return EFFECTIVE_RAIN_DATE, "RAIN_PLUS_DELAY"
    if TIME_REFERENCE == "mainshock":
        return MAINSHOCK_DATE, "MAINSHOCK"
    if TIME_REFERENCE == "network":
        return NETWORK_READY_DATE, "NETWORK_READY"

    raise ValueError(
        f"Unknown TIME_REFERENCE: {TIME_REFERENCE}. "
        "Use 'rain', 'mainshock', or 'network'."
    )


# ---------------------------------------------------------------------
def add_relative_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t0, label = get_reference_time()

    df["t_hours"] = (df["datetime"] - t0).dt.total_seconds() / 3600.0

    print("\n=== TIME REFERENCE ===")
    print(f"Reference label        : {label}")
    print(f"RAIN_DATE              : {RAIN_DATE}")
    print(f"DELAY_DAYS             : {DELAY_DAYS}")
    print(f"Effective reference t0 : {t0}")
    print(f"t_hours min            : {df['t_hours'].min()}")
    print(f"t_hours max            : {df['t_hours'].max()}")

    return df


# ---------------------------------------------------------------------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== APPLYING FILTERS ===")

    n0 = len(df)

    df1 = df[
        (df["depth_km"] <= DEPTH_MAX)
        & (df["datetime"] <= ANALYSIS_END_DATE)
    ].copy()

    print("\nBasic filters:")
    print(f"Input rows                 : {n0}")
    print(f"Rows after depth/end-date  : {len(df1)}")
    print(f"Removed                    : {n0 - len(df1)}")
    print(f"Analysis end date          : {ANALYSIS_END_DATE}")

    if len(df1) == 0:
        print("WARNING: No rows survived the depth/end-date filter.")
        return df1

    df1 = add_relative_time(df1)

    # Keep only events after the effective reference time
    df1 = df1[df1["t_hours"] >= 0].copy()

    print("\nAfter enforcing t_hours >= 0:")
    print(f"Rows: {len(df1)}")

    if len(df1) > 0:
        print(f"t_hours min: {df1['t_hours'].min()}")
        print(f"t_hours max: {df1['t_hours'].max()}")

    print("\nSample rows after relative time assignment:")
    cols_to_show = [
        c for c in
        ["datetime", "t_days_since_rain", "t_hours", "r_proj_m", "distance_km", "depth_km"]
        if c in df1.columns
    ]
    print(df1[cols_to_show].head(10))

    df2 = df1[
        (df1["t_hours"] >= TMIN)
        & (df1["t_hours"] <= TMAX)
    ].copy()

    print("\nTime-window filter:")
    print(f"Requested window : {TMIN} to {TMAX} hours")
    print(f"Rows after filter: {len(df2)}")

    if len(df2) == 0:
        print("WARNING: No rows survived the time-window filter.")
        print("Adjust TMIN/TMAX or change TIME_REFERENCE/DELAY_DAYS.")

    return df2


# ---------------------------------------------------------------------
def plot_profile(df: pd.DataFrame, output_path: Path) -> None:
    print(f"\nPlotting profile: {output_path}")
    print(f"Rows to plot: {len(df)}")

    plt.figure(figsize=(10, 5))

    if len(df) > 0:
        plt.scatter(
            df["distance_km"],
            df["depth_km"],
            s=20,
            alpha=0.5,
            label="Events"
        )

        # --- Median regression ---
        grouped, coef = median_profile_regression(df, n_bins=10)

        if len(grouped) > 0:
            # Plot median points
            plt.scatter(
                grouped["distance_med"],
                grouped["depth_med"],
                s=80,
                label="Median (binned)",
                zorder=3
            )

            # Regression line
            x_line = np.linspace(df["distance_km"].min(), df["distance_km"].max(), 200)
            y_line = coef[0] * x_line + coef[1]

            plt.plot(
                x_line,
                y_line,
                linewidth=2.5,
                label=f"Median trend (slope={coef[0]:.3f})"
            )

            print(f"\nMedian regression slope: {coef[0]:.4f} km/km")

    plt.gca().invert_yaxis()
    plt.xlabel("Distance (km)")
    plt.ylabel("Depth (km)")
    plt.title("Profile with robust median trend")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.close()


# ---------------------------------------------------------------------
def plot_time_distance(df: pd.DataFrame, output_path: Path) -> None:
    print(f"\nPlotting time-distance: {output_path}")
    print(f"Rows to plot: {len(df)}")

    plt.figure(figsize=(10, 5))

    if len(df) > 0:
        plt.scatter(df["t_hours"], df["distance_km"])

    plt.xlabel("Time since effective reference (hours)")
    plt.ylabel("Distance (km)")
    plt.title(
        f"Time-Distance ({TMIN:.0f} ≤ t ≤ {TMAX:.0f} h, ref={TIME_REFERENCE}, "
        f"delay={DELAY_DAYS:.1f} d)"
    )
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.close()


# ---------------------------------------------------------------------
def plot_diffusion(df: pd.DataFrame, output_path: Path) -> None:
    print(f"\nPlotting diffusion curves: {output_path}")
    print(f"Rows to plot: {len(df)}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # --------------------------------------------------
    # Conversión base
    # --------------------------------------------------
    df = df.copy()
    df["t_months"] = df["t_hours"] / (24.0 * 30.0)

    # --------------------------------------------------
    # Scatter de sismicidad
    # --------------------------------------------------
    if len(df) > 0:
        ax.scatter(
            df["t_months"],
            df["distance_km"],
            s=20,
            label="Earthquakes",
        )

    # --------------------------------------------------
    # Curvas de difusión
    # --------------------------------------------------
    for curve in DIFFUSION_CURVES:
        D = curve["D"]
        t_end = curve["t_end"]
        linestyle = curve["linestyle"]
        label = curve["label"]

        t_curve_start = max(0.0, TMIN)
        t_curve_end = min(t_end, TMAX)

        if t_curve_end <= t_curve_start:
            continue

        t_range_hours = np.linspace(t_curve_start, t_curve_end, 400)
        t_range_months = t_range_hours / (24.0 * 30.0)

        r_m = np.sqrt(4.0 * D * t_range_hours * 3600.0)
        r_km = r_m / 1000.0

        ax.plot(
            t_range_months,
            r_km,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )

    # --------------------------------------------------
    # Eje principal (meses)
    # --------------------------------------------------
    ax.set_xlabel("Time since effective reference (months)")
    ax.set_ylabel("Distance (km)")
    ax.set_xlim(TMIN / (24.0 * 30.0), TMAX / (24.0 * 30.0))

    # --------------------------------------------------
    # Eje superior (horas)
    # --------------------------------------------------
    def months_to_hours(x):
        return x * 24.0 * 30.0

    def hours_to_months(x):
        return x / (24.0 * 30.0)

    secax = ax.secondary_xaxis("top", functions=(months_to_hours, hours_to_months))
    secax.set_xlabel("Time (hours)")

    # --------------------------------------------------
    # Estética (espacio extra en título)
    # --------------------------------------------------
    ax.set_title(
        f"Apparent diffusion regimes ({TMIN:.0f} ≤ t ≤ {TMAX:.0f} h, "
        f"ref={TIME_REFERENCE}, delay={DELAY_DAYS:.1f} d)",
        pad=20  # 🔥 esto separa el título
    )

    ax.grid(True)
    ax.legend(title="Reference curves")

    fig.tight_layout()
    fig.savefig(output_path, format="png", dpi=300)
    plt.close(fig)
# ---------------------------------------------------------------------
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_catalog(CATALOG_PROJ)
    df = apply_filters(df)

    print("\n=== Darcy diffusion analysis ===")
    print(f"Input catalog          : {CATALOG_PROJ}")
    print(f"Time reference         : {TIME_REFERENCE}")
    print(f"RAIN_DATE              : {RAIN_DATE}")
    print(f"DELAY_DAYS             : {DELAY_DAYS}")
    print(f"EFFECTIVE_RAIN_DATE    : {EFFECTIVE_RAIN_DATE}")
    print(f"Analysis end           : {ANALYSIS_END_DATE}")
    print(f"N events               : {len(df)}")

    if len(df) == 0:
        print("\nWARNING: Final dataframe is empty.")
        print("The SVG files will be created, but the scatter points will be empty.")

    plot_profile(df, FIG_PROFILE)
    plot_time_distance(df, FIG_TIME_DISTANCE)
    plot_diffusion(df, FIG_DIFFUSION)

    print("\nSaved figures:")
    print(f"  - {FIG_PROFILE}")
    print(f"  - {FIG_TIME_DISTANCE}")
    print(f"  - {FIG_DIFFUSION}")


if __name__ == "__main__":
    main()

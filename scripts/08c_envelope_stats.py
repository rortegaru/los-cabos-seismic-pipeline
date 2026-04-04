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

from config import (
    DATA_INTERMEDIATE,
    FIGURES_DIR,
    PLOT_X_MIN,
    PLOT_X_MAX,
    PLOT_Y_MIN,
    PLOT_Y_MAX,
)

# ---------------------------------------------------------------------
# Input / Output
# ---------------------------------------------------------------------
INPUT_SELECTED_CSV = DATA_INTERMEDIATE / "selected_envelope_points.csv"

OUT_DIR = FIGURES_DIR / "darcy_envelope"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY_CSV = DATA_INTERMEDIATE / "selected_envelope_summary.csv"
OUT_BOOTSTRAP_CSV = DATA_INTERMEDIATE / "selected_envelope_bootstrap.csv"

FIG_HIST = OUT_DIR / "selected_envelope_dmin_hist.png"
FIG_ENVELOPE = OUT_DIR / "selected_envelope_curves.png"

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 42

# percentiles to summarize selected-point distribution
PERCENTILES = [50, 90, 95, 100]


# ---------------------------------------------------------------------
def load_selected_points(csv_path: Path) -> pd.DataFrame:
    print(f"\nReading selected points: {csv_path}")
    print(f"Exists: {csv_path.exists()}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Selected points file not found: {csv_path}\n"
            "Run 08b_select_envelope_points.py first and save your selection."
        )

    df = pd.read_csv(csv_path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    required = ["selection_order", "event_id", "t_days_since_rain", "distance_km", "Dmin_m2s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["t_days_since_rain", "distance_km", "Dmin_m2s"]).copy()

    # Only physically valid times
    df = df[df["t_days_since_rain"] > 0].copy()

    if len(df) == 0:
        raise ValueError("No valid selected points remain after filtering t_days_since_rain > 0.")

    return df


# ---------------------------------------------------------------------
def compute_summary_stats(dmin: np.ndarray) -> dict[str, float]:
    dmin = np.asarray(dmin, dtype=float)
    dmin = dmin[np.isfinite(dmin)]

    out = {
        "n_points": int(len(dmin)),
        "D50_m2s": float(np.percentile(dmin, 50)),
        "D90_m2s": float(np.percentile(dmin, 90)),
        "D95_m2s": float(np.percentile(dmin, 95)),
        "D100_m2s": float(np.max(dmin)),
        "Dmean_m2s": float(np.mean(dmin)),
        "Dstd_m2s": float(np.std(dmin, ddof=1)) if len(dmin) > 1 else 0.0,
        "Dmin_m2s": float(np.min(dmin)),
    }
    return out


# ---------------------------------------------------------------------
def bootstrap_summary(dmin: np.ndarray, n_boot: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dmin = np.asarray(dmin, dtype=float)
    dmin = dmin[np.isfinite(dmin)]

    n = len(dmin)
    if n == 0:
        raise ValueError("No finite Dmin values available for bootstrap.")

    rows = []
    for i in range(n_boot):
        sample = rng.choice(dmin, size=n, replace=True)

        rows.append({
            "iter": i + 1,
            "D50_m2s": np.percentile(sample, 50),
            "D90_m2s": np.percentile(sample, 90),
            "D95_m2s": np.percentile(sample, 95),
            "D100_m2s": np.max(sample),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
def summarize_bootstrap(df_boot: pd.DataFrame) -> pd.DataFrame:
    records = []

    for col in ["D50_m2s", "D90_m2s", "D95_m2s", "D100_m2s"]:
        vals = df_boot[col].to_numpy(dtype=float)

        records.append({
            "metric": col,
            "bootstrap_median": np.percentile(vals, 50),
            "bootstrap_mean": np.mean(vals),
            "bootstrap_ci2p5": np.percentile(vals, 2.5),
            "bootstrap_ci16": np.percentile(vals, 16),
            "bootstrap_ci84": np.percentile(vals, 84),
            "bootstrap_ci97p5": np.percentile(vals, 97.5),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
def build_human_summary(df_sel: pd.DataFrame, stats: dict[str, float], df_boot_summary: pd.DataFrame) -> pd.DataFrame:
    controlling_idx = int(np.argmax(df_sel["Dmin_m2s"].to_numpy()))
    controlling_row = df_sel.iloc[controlling_idx]

    out = {
        "n_points": stats["n_points"],
        "Dmin_m2s_min": stats["Dmin_m2s"],
        "D50_m2s": stats["D50_m2s"],
        "D90_m2s": stats["D90_m2s"],
        "D95_m2s": stats["D95_m2s"],
        "D100_m2s": stats["D100_m2s"],
        "D100_m2day": stats["D100_m2s"] * 86400.0,
        "D95_m2day": stats["D95_m2s"] * 86400.0,
        "D90_m2day": stats["D90_m2s"] * 86400.0,
        "D50_m2day": stats["D50_m2s"] * 86400.0,
        "Dmean_m2s": stats["Dmean_m2s"],
        "Dstd_m2s": stats["Dstd_m2s"],
        "controlling_event_id": controlling_row["event_id"],
        "controlling_datetime": controlling_row["datetime"],
        "controlling_t_days": controlling_row["t_days_since_rain"],
        "controlling_distance_km": controlling_row["distance_km"],
        "controlling_depth_km": controlling_row["depth_km"] if "depth_km" in controlling_row.index else np.nan,
        "controlling_Dmin_m2s": controlling_row["Dmin_m2s"],
    }

    # add bootstrap CI columns
    for _, row in df_boot_summary.iterrows():
        metric = row["metric"]
        out[f"{metric}_boot_median"] = row["bootstrap_median"]
        out[f"{metric}_boot_ci2p5"] = row["bootstrap_ci2p5"]
        out[f"{metric}_boot_ci97p5"] = row["bootstrap_ci97p5"]

    return pd.DataFrame([out])


# ---------------------------------------------------------------------
def plot_histogram(df_sel: pd.DataFrame, stats: dict[str, float], output_path: Path) -> None:
    vals = df_sel["Dmin_m2s"].to_numpy(dtype=float)

    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=min(10, max(5, len(vals))), alpha=0.8)

    plt.axvline(stats["D50_m2s"], linestyle="--", linewidth=1.8, label=f"D50 = {stats['D50_m2s']:.4f}")
    plt.axvline(stats["D90_m2s"], linestyle="-.", linewidth=1.8, label=f"D90 = {stats['D90_m2s']:.4f}")
    plt.axvline(stats["D95_m2s"], linestyle=":", linewidth=2.0, label=f"D95 = {stats['D95_m2s']:.4f}")
    plt.axvline(stats["D100_m2s"], linewidth=2.2, label=f"D100 = {stats['D100_m2s']:.4f}")

    plt.xlabel("Dmin (m²/s)")
    plt.ylabel("Count")
    plt.title("Distribution of selected-point Dmin values")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------
def diffusion_curve_km(t_days: np.ndarray, D_m2s: float) -> np.ndarray:
    """
    r = sqrt(4 D t)
    t_days -> seconds
    returns km
    """
    t_sec = np.asarray(t_days, dtype=float) * 86400.0
    r_m = np.sqrt(4.0 * D_m2s * t_sec)
    return r_m / 1000.0


# ---------------------------------------------------------------------
def plot_selected_with_envelopes(df_sel: pd.DataFrame, stats: dict[str, float], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))

    # selected points
    plt.scatter(
        df_sel["t_days_since_rain"],
        df_sel["distance_km"],
        s=70,
        alpha=0.9,
        label="Selected envelope points"
    )

    # annotate order
    for _, row in df_sel.iterrows():
        plt.annotate(
            str(int(row["selection_order"])),
            (row["t_days_since_rain"], row["distance_km"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

    # time array for curves
    tmax = max(PLOT_X_MAX, float(df_sel["t_days_since_rain"].max()) * 1.05)
    tmin = max(0.001, PLOT_X_MIN if PLOT_X_MIN > 0 else 0.001)
    t_curve = np.linspace(tmin, tmax, 600)

    for name, D in [
        ("D50", stats["D50_m2s"]),
        ("D90", stats["D90_m2s"]),
        ("D95", stats["D95_m2s"]),
        ("D100", stats["D100_m2s"]),
    ]:
        r_curve = diffusion_curve_km(t_curve, D)
        plt.plot(t_curve, r_curve, linewidth=2.0, label=f"{name} = {D:.4f} m²/s")

    plt.xlim(PLOT_X_MIN, PLOT_X_MAX)
    plt.ylim(PLOT_Y_MIN, PLOT_Y_MAX)

    plt.xlabel("Time since rain (days)")
    plt.ylabel("Projected distance (km)")
    plt.title("Selected envelope points and inferred diffusion envelopes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------
def print_console_summary(df_sel: pd.DataFrame, stats: dict[str, float], df_boot_summary: pd.DataFrame) -> None:
    controlling_idx = int(np.argmax(df_sel["Dmin_m2s"].to_numpy()))
    controlling_row = df_sel.iloc[controlling_idx]

    print("\n=== ENVELOPE SUMMARY ===")
    print(f"N selected points         : {stats['n_points']}")
    print(f"D50 (median)             : {stats['D50_m2s']:.6f} m²/s")
    print(f"D90                      : {stats['D90_m2s']:.6f} m²/s")
    print(f"D95                      : {stats['D95_m2s']:.6f} m²/s")
    print(f"D100 = max(Dmin)         : {stats['D100_m2s']:.6f} m²/s")
    print(f"D100                     : {stats['D100_m2s'] * 86400.0:.2f} m²/day")

    print("\nControlling event (max Dmin):")
    print(f"  event_id               : {controlling_row['event_id']}")
    print(f"  datetime               : {controlling_row['datetime']}")
    print(f"  t_days_since_rain      : {controlling_row['t_days_since_rain']:.6f}")
    print(f"  distance_km            : {controlling_row['distance_km']:.6f}")
    if "depth_km" in controlling_row.index:
        print(f"  depth_km               : {controlling_row['depth_km']:.6f}")
    print(f"  Dmin_m2s               : {controlling_row['Dmin_m2s']:.6f}")

    print("\nBootstrap 95% CI:")
    for _, row in df_boot_summary.iterrows():
        print(
            f"  {row['metric']:10s}: "
            f"{row['bootstrap_median']:.6f} "
            f"[{row['bootstrap_ci2p5']:.6f}, {row['bootstrap_ci97p5']:.6f}]"
        )


# ---------------------------------------------------------------------
def main() -> None:
    df_sel = load_selected_points(INPUT_SELECTED_CSV)

    # sort by user selection order, just in case
    df_sel = df_sel.sort_values("selection_order").reset_index(drop=True)

    dmin = df_sel["Dmin_m2s"].to_numpy(dtype=float)

    stats = compute_summary_stats(dmin)

    df_boot = bootstrap_summary(
        dmin,
        n_boot=BOOTSTRAP_N,
        seed=BOOTSTRAP_SEED
    )
    df_boot_summary = summarize_bootstrap(df_boot)

    df_summary = build_human_summary(df_sel, stats, df_boot_summary)

    # save outputs
    df_summary.to_csv(OUT_SUMMARY_CSV, index=False)
    df_boot.to_csv(OUT_BOOTSTRAP_CSV, index=False)

    plot_histogram(df_sel, stats, FIG_HIST)
    plot_selected_with_envelopes(df_sel, stats, FIG_ENVELOPE)

    print_console_summary(df_sel, stats, df_boot_summary)

    print("\nSaved files:")
    print(f"  - {OUT_SUMMARY_CSV}")
    print(f"  - {OUT_BOOTSTRAP_CSV}")
    print(f"  - {FIG_HIST}")
    print(f"  - {FIG_ENVELOPE}")


if __name__ == "__main__":
    main()


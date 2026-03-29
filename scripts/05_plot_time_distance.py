#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Allow importing config.py from repo root
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    CATALOG_PROJ,
    FIG_PROJECTED,
    FIG_EUCLIDEAN,
    MAINSHOCK_DATE,
    NETWORK_READY_DATE,
    RAIN_DATE,
    PLOT_X_MIN,
    PLOT_X_MAX,
    PLOT_Y_MIN,
    PLOT_Y_MAX,
)


def days_since(date_series: pd.Series, ref: pd.Timestamp) -> pd.Series:
    return (date_series - ref).dt.total_seconds() / 86400.0


def add_lollipop(ax, x: float, color: str, y_frac: float = 0.95) -> None:
    """
    Draw a vertical lollipop marker with a filled colored circle.
    """
    y_top = ax.get_ylim()[1]
    y_marker = y_top * y_frac

    ax.vlines(x, 0, y_marker, linewidth=1.6, color=color, zorder=2)
    ax.scatter(
        [x],
        [y_marker],
        s=70,
        facecolors=color,
        edgecolors="black",
        linewidths=0.6,
        zorder=4,
    )


def build_event_legend():
    """
    Create a clean detached legend for the three reference dates.
    """
    handles = [
        Line2D(
            [0], [0],
            color="tab:green",
            linewidth=1.6,
            marker="o",
            markersize=7,
            markerfacecolor="tab:green",
            markeredgecolor="black",
            markeredgewidth=0.6,
            label="Network ready (Aug 30, 2024)",
        ),
        Line2D(
            [0], [0],
            color="tab:red",
            linewidth=1.6,
            marker="o",
            markersize=7,
            markerfacecolor="tab:red",
            markeredgecolor="black",
            markeredgewidth=0.6,
            label="Main event (Sep 3, 2024)",
        ),
        Line2D(
            [0], [0],
            color="tab:blue",
            linewidth=1.6,
            marker="o",
            markersize=7,
            markerfacecolor="tab:blue",
            markeredgecolor="black",
            markeredgewidth=0.6,
            label="Rainfall (Sep 14, 2024)",
        ),
    ]
    return handles


def make_plot(df, y_col, ylabel, title, outpath):
    fig, ax = plt.subplots(figsize=(9, 5.8))

    # Scatter cloud (data)
    ax.scatter(
        df["t_days_since_rain"],
        df[y_col],
        s=16,
        alpha=0.75,
        linewidths=0,
        color="0.25",
        zorder=1,
    )

    # Axis limits and style
    ax.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
    ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
    ax.set_xlabel("Days relative to rainfall")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    # Reference times in rainfall-relative coordinates
    x_network = (NETWORK_READY_DATE - RAIN_DATE).total_seconds() / 86400.0
    x_mainshock = (MAINSHOCK_DATE - RAIN_DATE).total_seconds() / 86400.0
    x_rain = 0.0

    # Lollipops only for key dates
    add_lollipop(ax, x_network, color="tab:green")
    add_lollipop(ax, x_mainshock, color="tab:red")
    add_lollipop(ax, x_rain, color="tab:blue")

    # Detached legend box
    legend = ax.legend(
        handles=build_event_legend(),
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.7",
        fontsize=9,
        title="Reference dates",
        title_fontsize=9,
    )
    legend.set_zorder(10)

    # Interpretive note
    ax.text(
        0.99,
        0.02,
        "Negative values indicate pre-rainfall seismicity.\n"
        "The network was operating before both the main event and rainfall.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main():
    FIG_PROJECTED.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CATALOG_PROJ, parse_dates=["datetime"])

    if df.empty:
        raise ValueError("Projected catalog is empty.")

    # Distances in km
    df["r_proj_km"] = np.abs(df["r_proj_m"]) / 1000.0
    df["d_euclid_ref_km"] = df["d_euclid_ref_m"] / 1000.0

    # Recompute rainfall-relative time
    df["t_days_since_rain"] = days_since(df["datetime"], RAIN_DATE)

    # Projected plot
    make_plot(
        df,
        y_col="r_proj_km",
        ylabel="Distance normal to fault (km)",
        title="El Tule subset: projected time–distance",
        outpath=FIG_PROJECTED,
    )

    # Euclidean plot
    make_plot(
        df,
        y_col="d_euclid_ref_km",
        ylabel="Distance from reference point (km)",
        title="El Tule subset: Euclidean time–distance",
        outpath=FIG_EUCLIDEAN,
    )

    print("\nPlot summary")
    print("-" * 50)
    print(f"Events plotted: {len(df):,}")
    print(f"Saved: {FIG_PROJECTED}")
    print(f"Saved: {FIG_EUCLIDEAN}")


if __name__ == "__main__":
    main()

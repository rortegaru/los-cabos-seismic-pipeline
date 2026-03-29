#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow importing config.py from repository root
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CATALOG_HQ, CATALOG_PAPER, PAPER_START, PAPER_END


def main() -> None:
    # Ensure output directory exists
    CATALOG_PAPER.parent.mkdir(parents=True, exist_ok=True)

    # Read HQ catalog
    df = pd.read_csv(CATALOG_HQ, parse_dates=["datetime"])

    # Apply manuscript time window
    df_paper = df[
        (df["datetime"] >= PAPER_START) &
        (df["datetime"] <= PAPER_END)
    ].copy()

    # Sort for reproducibility
    df_paper = df_paper.sort_values(["datetime", "event_id"]).reset_index(drop=True)

    # Save output
    df_paper.to_csv(CATALOG_PAPER, index=False)

    # Summary
    print("\nTime-window summary")
    print("-" * 50)
    print(f"Input HQ catalog:               {CATALOG_HQ}")
    print(f"Start time:                     {PAPER_START}")
    print(f"End time:                       {PAPER_END}")
    print(f"Events in paper catalog:        {len(df_paper):,}")

    if len(df_paper) > 0:
        print(f"First event in window:          {df_paper['datetime'].min()}")
        print(f"Last event in window:           {df_paper['datetime'].max()}")

    print(f"\nSaved paper catalog to:         {CATALOG_PAPER}")


if __name__ == "__main__":
    main()

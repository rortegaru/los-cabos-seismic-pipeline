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

from config import (
    CATALOG_HQ,
    CATALOG_PAPER,
    PAPER_START,
    PAPER_END,
    DATE1,
    DATE2,
    DATE3,
    DATE4,
    DATE5,
    DATE6,
    CATALOG_DATE1,
    CATALOG_DATE2,
    CATALOG_DATE3,
    CATALOG_DATE4,
    CATALOG_DATE5,
    CATALOG_DATE6,
)


def save_window(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_path: Path,
    label: str,
) -> None:
    """Save catalog filtered between start and end dates."""
    df_window = df[
        (df["datetime"] >= start) &
        (df["datetime"] <= end)
    ].copy()

    df_window = df_window.sort_values(["datetime", "event_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_window.to_csv(output_path, index=False)

    print(f"\n{label}")
    print("-" * 50)
    print(f"Start time:                     {start}")
    print(f"End time:                       {end}")
    print(f"Events:                         {len(df_window):,}")
    if not df_window.empty:
        print(f"First event:                    {df_window['datetime'].min()}")
        print(f"Last event:                     {df_window['datetime'].max()}")
    print(f"Saved to:                       {output_path}")


def main() -> None:
    # Read HQ catalog
    df = pd.read_csv(CATALOG_HQ, parse_dates=["datetime"])

    # Main manuscript catalog
    save_window(
        df=df,
        start=PAPER_START,
        end=PAPER_END,
        output_path=CATALOG_PAPER,
        label="PAPER CATALOG",
    )

    # Progressive catalogs from PAPER_START up to each date
    date_windows = [
        ("CATALOG DATE1", DATE1, CATALOG_DATE1),
        ("CATALOG DATE2", DATE2, CATALOG_DATE2),
        ("CATALOG DATE3", DATE3, CATALOG_DATE3),
        ("CATALOG DATE4", DATE4, CATALOG_DATE4),
        ("CATALOG DATE5", DATE5, CATALOG_DATE5),
        ("CATALOG DATE6", DATE6, CATALOG_DATE6),
    ]

    for label, end_date, output_path in date_windows:
        save_window(
            df=df,
            start=PAPER_START,
            end=end_date,
            output_path=output_path,
            label=label,
        )


if __name__ == "__main__":
    main()

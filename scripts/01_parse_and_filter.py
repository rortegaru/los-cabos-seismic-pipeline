#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing config.py from repository root
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import RAW_RELOC, CATALOG_CLEAN, CATALOG_HQ, ERR_H_MAX


def parse_hypodd_reloc(filepath: str | Path) -> pd.DataFrame:
    """
    Parse a hypoDD .reloc catalog and return a cleaned DataFrame.
    """
    filepath = Path(filepath)
    rows = []
    bad_lines = []

    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        for line_number, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue

            parts = s.split()

            if len(parts) < 17:
                bad_lines.append((line_number, len(parts), s))
                continue

            try:
                row = {
                    "year": int(parts[10]),
                    "month": int(parts[11]),
                    "day": int(parts[12]),
                    "hour": int(parts[13]),
                    "minute": int(parts[14]),
                    "second": float(parts[15]),
                    "event_id": int(parts[0]),
                    "latitude": float(parts[1]),
                    "longitude": float(parts[2]),
                    "depth_km": float(parts[3]),
                    "magnitude": float(parts[16]),
                    "err_x_m": float(parts[7]),
                    "err_y_m": float(parts[8]),
                    "err_z_m": float(parts[9]),
                }
                rows.append(row)

            except Exception as exc:
                bad_lines.append((line_number, len(parts), f"{s} || ERROR: {exc}"))

    if not rows:
        raise ValueError("No valid rows could be parsed from the .reloc file.")

    df = pd.DataFrame(rows)

    # Build datetime from integer seconds + microseconds
    sec_int = df["second"].astype(int)
    microsec = ((df["second"] - sec_int) * 1_000_000).round().astype(int)

    df["datetime"] = pd.to_datetime(
        dict(
            year=df["year"],
            month=df["month"],
            day=df["day"],
            hour=df["hour"],
            minute=df["minute"],
            second=sec_int,
            microsecond=microsec,
        ),
        errors="coerce",
    )

    # Combined horizontal uncertainty
    df["err_h_m"] = np.sqrt(df["err_x_m"] ** 2 + df["err_y_m"] ** 2)

    # Sort chronologically
    df = df.sort_values(["datetime", "event_id"]).reset_index(drop=True)

    # Reorder columns
    df = df[
        [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "datetime",
            "event_id",
            "latitude",
            "longitude",
            "depth_km",
            "magnitude",
            "err_x_m",
            "err_y_m",
            "err_z_m",
            "err_h_m",
        ]
    ]

    if bad_lines:
        print(f"\nProblematic lines: {len(bad_lines)}")
        for item in bad_lines[:10]:
            print(item)

    return df


def main() -> None:
    # Ensure output directory exists
    CATALOG_CLEAN.parent.mkdir(parents=True, exist_ok=True)

    df = parse_hypodd_reloc(RAW_RELOC)

    # Save full catalog
    df.to_csv(CATALOG_CLEAN, index=False)

    # Save high-quality subset
    df_hq = df[df["err_h_m"] <= ERR_H_MAX].copy()
    df_hq.to_csv(CATALOG_HQ, index=False)

    # Summary
    print("\nGeneral summary")
    print("-" * 50)
    print(f"Raw input file:                {RAW_RELOC}")
    print(f"Total events:                  {len(df):,}")
    print(f"High-quality events:           {len(df_hq):,}")
    print(f"Retained percentage:           {100 * len(df_hq) / len(df):.2f}%")
    print(f"First event:                   {df['datetime'].min()}")
    print(f"Last event:                    {df['datetime'].max()}")
    print(f"Maximum err_h_m (full):        {df['err_h_m'].max():.2f} m")
    print(f"Maximum err_h_m (HQ subset):   {df_hq['err_h_m'].max():.2f} m")

    print(f"\nSaved full catalog to:         {CATALOG_CLEAN}")
    print(f"Saved HQ catalog to:           {CATALOG_HQ}")


if __name__ == "__main__":
    main()

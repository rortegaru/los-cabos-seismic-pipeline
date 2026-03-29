#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import subprocess
import sys

SCRIPTS = [
    "scripts/01_parse_and_filter.py",
    "scripts/02_time_window.py",
    "scripts/03_spatial_subset.py",
    "scripts/04_fault_projection.py",
    "scripts/05_plot_time_distance.py",
]

def run_script(script_path: str) -> None:
    print(f"\n>>> Running {script_path}")
    subprocess.run([sys.executable, script_path], check=True)

def main() -> None:
    for script in SCRIPTS:
        run_script(script)
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()

# Los Cabos seismic processing pipeline

This repository contains a fully reproducible workflow for generating the seismic catalogs and time–distance plots used in the Los Cabos hydrometeorological-triggering analysis.

## Overview

The workflow starts from a single raw hypoDD relocation catalog and applies:

1. parsing and catalog cleaning
2. horizontal uncertainty filtering
3. paper time-window selection
4. spatial selection within the El Tule polygon
5. fault-based geometric projection
6. time–distance plotting

All intermediate products are exported as CSV files for inspection and verification.

## Repository structure

- `data/raw/`: raw input catalog
- `data/external/`: shapefiles and fault-geometry auxiliary files
- `data/intermediate/`: generated intermediate catalogs
- `figures/`: final plots
- `scripts/`: modular processing scripts
- `run_pipeline.py`: executes the complete workflow

## Input data

The workflow requires:

- a hypoDD relocation catalog (`.reloc`)
- the El Tule polygon shapefile
- a CSV file with the two fault endpoints used for projection

## Run the full pipeline

```bash
python run_pipeline.py

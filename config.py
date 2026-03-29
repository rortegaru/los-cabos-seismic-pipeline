from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_EXTERNAL = BASE_DIR / "data" / "external"
DATA_INTERMEDIATE = BASE_DIR / "data" / "intermediate"
FIGURES_DIR = BASE_DIR / "figures"

RAW_RELOC = DATA_RAW / "hypDD2024-mar2026LosCabos_SSN.reloc"
TULE_POLYGON = DATA_EXTERNAL / "shallowpoli.shp"
FAULT_ENDPOINTS = DATA_EXTERNAL / "extremosnodes.csv"

CATALOG_CLEAN = DATA_INTERMEDIATE / "catalog_clean_complete.csv"
CATALOG_HQ = DATA_INTERMEDIATE / "catalog_hq_errh_le_280m.csv"
CATALOG_PAPER = DATA_INTERMEDIATE / "catalog_paper_2024-08-01_2025-06-10.csv"
CATALOG_TULE_FLAG = DATA_INTERMEDIATE / "catalog_paper_hq_with_tule_flag.csv"
CATALOG_TULE = DATA_INTERMEDIATE / "catalog_tule_subset.csv"
CATALOG_PROJ = DATA_INTERMEDIATE / "catalog_tule_projection.csv"

FIG_PROJECTED = FIGURES_DIR / "darcy_plot_projected.png"
FIG_EUCLIDEAN = FIGURES_DIR / "darcy_plot_euclidean.png"

ERR_H_MAX = 280.0
PAPER_START = pd.Timestamp("2024-08-01")
PAPER_END = pd.Timestamp("2025-06-10 23:59:59")
RAIN_DATE = pd.Timestamp("2024-09-13 00:00:00")

CRS_EVENTS = "EPSG:4326"
CRS_WORK = "EPSG:32612"

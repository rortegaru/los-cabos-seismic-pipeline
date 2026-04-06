from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_EXTERNAL = BASE_DIR / "data" / "external"
DATA_INTERMEDIATE = BASE_DIR / "data" / "intermediate"
FIGURES_DIR = BASE_DIR / "figures"

RAW_RELOC = DATA_RAW / "hypDD2024-mar2026LosCabos_SSN.reloc"
#TULE_POLYGON = DATA_EXTERNAL / "shallowpoli2.shp"
TULE_POLYGON = DATA_EXTERNAL / "shallowpoli.shp"
#FAULT_ENDPOINTS = DATA_EXTERNAL / "extremos2.csv"
FAULT_ENDPOINTS = DATA_EXTERNAL / "extremosnodes.csv"

CATALOG_CLEAN = DATA_INTERMEDIATE / "catalog_clean_complete.csv"
CATALOG_HQ = DATA_INTERMEDIATE / "catalog_hq_errh_le_280m.csv"
CATALOG_PAPER = DATA_INTERMEDIATE / "catalog_paper_2024-08-01_2025-06-10.csv"
CATALOG_TULE_FLAG = DATA_INTERMEDIATE / "catalog_paper_hq_with_tule_flag.csv"
CATALOG_TULE = DATA_INTERMEDIATE / "catalog_tule_subset.csv"
CATALOG_PROJ = DATA_INTERMEDIATE / "catalog_tule_projection.csv"

# Figuras estáticas
FIG_PROJECTED = FIGURES_DIR / "darcy_plot_projected.png"
FIG_EUCLIDEAN = FIGURES_DIR / "darcy_plot_euclidean.png"

ERR_H_MAX = 280.0
PAPER_START = pd.Timestamp("2024-08-01")
PAPER_END = pd.Timestamp("2025-06-10 23:59:59")
MAINSHOCK_DATE = pd.Timestamp("2024-09-03 00:00:00")
NETWORK_READY_DATE = pd.Timestamp("2024-08-30 00:00:00")
ANALYSIS_END_DATE = pd.Timestamp("2025-02-08 23:59:59")
DATE1 = pd.Timestamp("2024-08-30")
DATE2 = pd.Timestamp("2024-09-15")
DATE3 = pd.Timestamp("2024-09-30")
DATE4 = pd.Timestamp("2024-10-15")
DATE5 = pd.Timestamp("2024-10-30")
DATE6 = pd.Timestamp("2025-05-15")

# =========================
# TIME-WINDOW OUTPUTS
# =========================

CATALOG_DATE1 = DATA_INTERMEDIATE / "catalog_until_DATE1_2024-08-30.csv"
CATALOG_DATE2 = DATA_INTERMEDIATE / "catalog_until_DATE2_2024-09-15.csv"
CATALOG_DATE3 = DATA_INTERMEDIATE / "catalog_until_DATE3_2024-09-30.csv"
CATALOG_DATE4 = DATA_INTERMEDIATE / "catalog_until_DATE4_2024-10-15.csv"
CATALOG_DATE5 = DATA_INTERMEDIATE / "catalog_until_DATE5_2024-10-30.csv"
CATALOG_DATE6 = DATA_INTERMEDIATE / "catalog_until_DATE6_2025-05-15.csv"

RAIN_DATE = pd.Timestamp("2024-09-14 00:00:00")

PLOT_X_MIN = -30
PLOT_X_MAX = 300
PLOT_Y_MIN = 0.0
PLOT_Y_MAX = 5.0

LOLLIPOP_BIN_DAYS = 5

CRS_EVENTS = "EPSG:4326"
CRS_WORK = "EPSG:32612"

# =========================
# MAP SETTINGS
# =========================

MAP_LAT_RANGE = (22.9, 23.1)
MAP_LON_RANGE = (-109.8, -109.6)
MAP_DEPTH_RANGE = (2, 12)

MAP_HEIGHT = 700
MAP_ZOOM = 7
MAP_STYLE = "open-street-map"

# =========================
# OUTPUT FILES
# =========================

MAP_ANIMATION_HTML = FIGURES_DIR / "06_seismic_sequence_map.html"
MAP_ANIMATION_DAILY_CSV = DATA_INTERMEDIATE / "06_catalog_for_animation.csv"

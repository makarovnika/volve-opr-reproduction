"""Single source of truth: paths, well lists, split dates, feature roster, seed."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset location — prefer input data/ if populated, fall back to My work/
_EXCEL = "MN08Dec2025OriginalDatasetAllWells (2).xlsx"
DATASET = (
    os.path.join(ROOT, "input data", _EXCEL)
    if os.path.exists(os.path.join(ROOT, "input data", _EXCEL))
    else os.path.join(ROOT, "My work", _EXCEL)
)

RESULTS_DIR     = os.path.join(ROOT, "results")
TABLES_DIR      = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR     = os.path.join(RESULTS_DIR, "figures")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
STAGE1_DIR      = os.path.join(RESULTS_DIR, "stage1")
DATA_CARD       = os.path.join(RESULTS_DIR, "data_card.md")

SEED = 42

PRODUCERS = ["15 9-F-1 C", "15 9-F-11 H", "15 9-F-12 H", "15 9-F-14 H", "15 9-F-15 D"]
INJECTORS = ["15 9-F-4 AH", "15 9-F-5 AH"]

WELL_ABBREV = {
    "15 9-F-1 C":  "F1C",
    "15 9-F-11 H": "F11H",
    "15 9-F-12 H": "F12H",
    "15 9-F-14 H": "F14H",
    "15 9-F-15 D": "F15D",
    "15 9-F-4 AH": "F4AH",
    "15 9-F-5 AH": "F5AH",
}

# Frozen split dates (TZ-01 §2.8)
CAL_START  = "2008-02-17"
CAL_END    = "2016-09-17"
TRAIN_END  = "2015-01-25"
TEST_START = "2015-01-26"

# Producer feature channels (before per-well exclusions)
PRODUCER_CHANNELS = ["HRS", "ABHP", "ABHT", "ADPT", "AAP", "ACS", "AWHP", "AWHT", "CZ"]
# Injector channels — only HRS+WIR are populated (H3 finding)
INJECTOR_CHANNELS = ["HRS", "WIR"]
# Target columns in the wide frame
TARGETS       = ["OPR_field", "GPR_field", "WPR_field"]
TARGET_SOURCES = ["OPR", "GPR", "WPR"]

# Channels subject to zero-as-missing recoding (TZ-01 §2.4)
ZERO_AS_MISSING = ["ABHP", "ABHT", "ADPT", "AAP", "ACS", "AWHP", "AWHT"]

VAL_FRACTION = 0.15   # last fraction of train days used for validation

# Expected counts from TZ-01 §2 (for data card assertions).
# NOTE: calendar_days=3136 because pd.date_range("2008-02-17","2016-09-17") has
# 3136 endpoints; the spec says 3135 counting day-intervals rather than day-points.
# 2535 train + 601 test = 3136 confirms this interpretation.
EXPECTED = {
    "calendar_days":               3136,
    "train_rows":                  2535,
    "test_rows":                   601,
    "hrs_gt_24":                   13,
    "allocation_artifacts":        2,
    "small_negative_wpr":          2,
    "inconsistent_day":            5,
    "f12_abhp_zero_producing":     1906,
    "zero_production_days_train":  133,
}

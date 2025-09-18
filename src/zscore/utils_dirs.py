from pathlib import Path

# base folder
BASE_DIR = Path(__file__).resolve().parents[2]

# high-level folders
SRC_DR_DIR = BASE_DIR / "src" / "zscore"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# treebank mrg files
TREEBANK_DIR = BASE_DIR / "data" / "treebank_3"
TREEBANK_MRG_DIR = BASE_DIR / "data" / "treebank_3" / "parsed" / "mrg" / "swbd"
TREEBANK_SAMPLE_MRG_FILE = TREEBANK_MRG_DIR / "4" / "sw4004.mrg"

# treebank txt files
TREEBANK_PROCESSED_DIR = BASE_DIR / "data" / "treebank_3_flat"
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
VAL_FRACTION = 0.15
N_HELD_OUT_INTENTS = 7
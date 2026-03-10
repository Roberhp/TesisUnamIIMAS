
from pathlib import Path

SEED = 42
LANGUAGE = "english"
STOP_WORDS = LANGUAGE

TEXT_COLUMN = 'statement'
LABEL_COLUMN='status'

# Carpeta donde se guardarán modelos individuales
MODELS_DIR = "models"



BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "Combined_Data.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
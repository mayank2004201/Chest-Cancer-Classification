from pathlib import Path
from pathlib import Path

CONFIG_FILE_PATH = Path(__file__).parent / "config" / "config.yaml"
# CONFIG_FILE_PATH = Path("config/config.yaml")

CONFIG_FILE_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

PARAMS_FILE_PATH = Path("params.yaml")

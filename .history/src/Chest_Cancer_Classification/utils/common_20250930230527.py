import os 
from box.exceptions import BoxValueError
import yaml
from Chest_Cancer_Classification import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """_summary_

    Args:
        path_to_yaml (str): path like input

    Returns:
        ConfigBox: config box type
    """
    
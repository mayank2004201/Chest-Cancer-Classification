import os 
from box.exceptions import BoxValueError
import yaml
from Chest_Cancer_Classification import logger
import json
import joblib
from ensure import ensure_annotations
from box import config_box
from pathlib import Path
from typing import Any
import base64



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> C
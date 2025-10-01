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
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input
    
    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file : {path_to_yaml} loaded succesfully.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty.")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list,verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path to directories
        ignore_log (bool, optional): ignores if multiple dirs are to be created . Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f"Created directory at :{path}")

@ensure_annotations
def load_json(path:Path) -> ConfigBox:
    """load json file data

    Args:
        path (Path): path to json file 

    Returns:
        ConfigBox: data as class attributes instead of dicts
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"json file loaded succesfully from: {path}")

    
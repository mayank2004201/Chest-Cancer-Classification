# import os 
# from box.exceptions import BoxValueError
# import yaml
# from Chest_Cancer_Classification import logger
# import json
# import joblib
# from pydantic import validate_arguments
# from box import box
# from pathlib import Path
# from typing import Any
# import base64


# @validate_arguments
# def read_yaml(path_to_yaml: Path) -> box:
#     """reads yaml file and returns

#     Args:
#         path_to_yaml (str): path like input
    
#     Raises:
#         ValueError: if yaml file is empty
#         e: empty file

#     Returns:
#      box: box type
#     """
#     try:
#         with open(path_to_yaml) as yaml_file:
#             content = yaml.safe_load(yaml_file)
#             logger.info(f"yaml file : {path_to_yaml} loaded succesfully.")
#             return box(content)
#     except BoxValueError:
#         raise ValueError("yaml file is empty.")
#     except Exception as e:
#         raise e
    
# @validate_arguments
# def create_directories(path_to_directories: list,verbose=True):
#     """create list of directories

#     Args:
#         path_to_directories (list): list of path to directories
#         ignore_log (bool, optional): ignores if multiple dirs are to be created . Defaults to False.
#     """
#     for path in path_to_directories:
#         os.makedirs(path,exist_ok=True)
#         if verbose:
#             logger.info(f"Created directory at :{path}")

# @validate_arguments
# def load_json(path:Path) -> box:
#     """load json file data

#     Args:
#         path (Path): path to json file 

#     Returns:
#      box: data as class attributes instead of dicts
#     """
#     with open(path) as f:
#         content = json.load(f)

#     logger.info(f"json file loaded succesfully from: {path}")
#     return box(content)

# @validate_arguments
# def save_bin(data:Any,path:Path):
#     """save binary file 

#     Args:
#         data (Any): data to be saved as binary 
#         path (Path): path to binary file 
#     """
#     joblib.dump(value=data,filename=path)
#     logger.info(f"Binary file saved at :{path}")

# @validate_arguments
# def load_bin(path:Path) -> Any:
#     """load binary data

#     Args:
#         path (Path): path to binary file

#     Returns:
#         Any: objects stored in the file 
#     """
#     data = joblib.load(path)
#     logger.info(f"binary file loaded from :{path}")
    
# @validate_arguments
# def get_size(path: Path) -> str:
#     """get size in kb

#     Args:
#         path (Path): path to the file 

#     Returns:
#         str: size in kb 
#     """
#     size_in_kb = round(os.path.getsize(path)/1024)
#     return f"~{size_in_kb} KB"

# def decodeImage(imgstring,filename):
#     imgdata = base64.b64decode(imgstring)
#     with open(filename,'wb') as f:
#         f.write(imgdata)
#         f.close()

# def encodeImageIntoBase64(croppedImagePath):
#     with open(croppedImagePath,'rb') as f:
#         return base64.b64encode(f.read())

import os
from box.exceptions import BoxValueError
from box import ConfigBox
import yaml
from Chest_Cancer_Classification import logger
import json
import joblib
from pydantic import validate_arguments
from pathlib import Path
from typing import Any
import base64

@validate_arguments
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads YAML file and returns a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the YAML file.
    
    Raises:
        ValueError: If the YAML file is empty.
        Exception: For other errors during file reading.

    Returns:
        ConfigBox: YAML content with dot notation access.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError("YAML file is empty.")
            logger.info(f"YAML file: {path_to_yaml} loaded successfully.")
            return ConfigBox(content)  # Fixed: box -> ConfigBox
    except FileNotFoundError as e:
        logger.error(f"File not found: {path_to_yaml}")
        raise e
    except BoxValueError:
        raise ValueError("YAML file is empty.")
    except Exception as e:
        logger.error(f"Error loading YAML file {path_to_yaml}: {str(e)}")
        raise e

@validate_arguments
def create_directories(path_to_directories: list, verbose: bool = True):
    """Create a list of directories.

    Args:
        path_to_directories (list): List of directory paths.
        verbose (bool, optional): Log creation if True. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@validate_arguments
def load_json(path: Path) -> ConfigBox:
    """Load JSON file data.

    Args:
        path (Path): Path to JSON file.

    Returns:
        ConfigBox: JSON data with dot notation access.
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)  # Fixed: box -> ConfigBox

@validate_arguments
def save_bin(data: Any, path: Path):
    """Save data as a binary file.

    Args:
        data (Any): Data to save.
        path (Path): Path to binary file.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@validate_arguments
def load_bin(path: Path) -> Any:
    """Load binary data.

    Args:
        path (Path): Path to binary file.

    Returns:
        Any: Object stored in the file.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data

@validate_arguments
def get_size(path: Path) -> str:
    """Get file size in KB.

    Args:
        path (Path): Path to the file.

    Returns:
        str: Size in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~{size_in_kb} KB"

@validate_arguments
def decodeImage(imgstring: str, filename: str):
    """Decode base64 image string and save to file.

    Args:
        imgstring (str): Base64-encoded image string.
        filename (str): Path to save the image.
    """
    imgdata = base    
import os 
from pathlib import Path
import logging

logging.basicConfig(Level=logging.INFO,format='[%(asctime)s]: %(message)s')

project_name = "Chest Cancer Classification"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/____init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb", 
    "templates/index.html"
]
    
for filepath in list_of_files:
    filepath = 
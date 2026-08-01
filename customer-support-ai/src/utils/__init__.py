import os
import sys
import yaml
from pathlib import Path
from src.logger import logging
from src.exception import CustomException

def read_yaml(path_to_yaml: Path) -> dict:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return content
    except Exception as e:
        raise CustomException(e, sys)

def create_directories(path_to_directories: list, ignore_log=False):
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if not ignore_log:
                logging.info(f"created directory at: {path}")
    except Exception as e:
        raise CustomException(e, sys)

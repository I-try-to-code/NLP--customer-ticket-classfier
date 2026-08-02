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

import joblib

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

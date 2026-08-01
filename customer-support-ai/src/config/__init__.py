import sys
from pathlib import Path
from src.constants import *
from src.utils import read_yaml, create_directories
from src.entity import DataIngestionConfig, DataTransformationConfig
from src.logger import logging
from src.exception import CustomException

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, schema_filepath=SCHEMA_FILE_PATH):
        try:
            self.config = read_yaml(Path(config_filepath))
            self.schema = read_yaml(Path(schema_filepath))

            create_directories([self.config['artifacts_root']])
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            config = self.config['data_ingestion']
            create_directories([config['root_dir']])

            data_ingestion_config = DataIngestionConfig(
                root_dir=Path(config['root_dir']),
                source_file=Path(config['source_file']),
                local_data_file=Path(config['local_data_file'])
            )
            return data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            config = self.config['data_transformation']
            create_directories([config['root_dir']])

            data_transformation_config = DataTransformationConfig(
                root_dir=Path(config['root_dir']),
                data_path=Path(config['data_path']),
                transformed_data_path=Path(config['transformed_data_path'])
            )
            return data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)

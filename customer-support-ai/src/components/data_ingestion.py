import os
import sys
import shutil
from pathlib import Path
from src.logger import logging
from src.exception import CustomException
from src.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self) -> Path:
        logging.info("Entered the data ingestion component")
        try:
            source_file = self.config.source_file
            
            if not os.path.exists(source_file):
                raise Exception(f"Source file {source_file} does not exist. Please place the dataset there.")
                
            logging.info(f"Source file {source_file} exists. Starting data copy.")
            
            os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
            shutil.copy(source_file, self.config.local_data_file)
            
            logging.info(f"Data successfully copied to {self.config.local_data_file}")
            
            return self.config.local_data_file
        except Exception as e:
            raise CustomException(e, sys)

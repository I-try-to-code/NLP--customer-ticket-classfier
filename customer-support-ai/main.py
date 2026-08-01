import sys
from src.logger import logging
from src.exception import CustomException
from src.config import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

def main():
    try:
        logging.info(">>>>>> Data Ingestion phase started <<<<<<")
        
        # 1. Initialize configuration manager to load yamls
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        
        # 2. Feed configuration to the ingestion component and execute
        data_ingestion = DataIngestion(config=data_ingestion_config)
        saved_dataset_path = data_ingestion.initiate_data_ingestion()
        
        logging.info(f"Data ingestion completed. Dataset is securely saved at: {saved_dataset_path}")
        logging.info(">>>>>> Data Ingestion phase completed successfully <<<<<<\n")
        
        
        logging.info(">>>>>> Data Transformation phase started <<<<<<")
        
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        transformed_dataset_path = data_transformation.initiate_data_transformation()
        
        logging.info(f"Data transformation completed. Transformed dataset saved at: {transformed_dataset_path}")
        logging.info(">>>>>> Data Transformation phase completed successfully <<<<<<\n")
        
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()

import os
from pathlib import Path

base_dir = Path("c:/Users/Ayan/Documents/GitHub/AIML/NLP- customer ticket classfier/customer-support-ai")

dirs = [
    "artifacts",
    "configs",
    "notebooks",
    "logs",
    "src/components",
    "src/pipeline",
    "src/utils",
    "src/entity",
    "src/config",
]

for d in dirs:
    os.makedirs(base_dir / d, exist_ok=True)

files = {
    "configs/config.yaml": """artifacts_root: artifacts

data_ingestion:
    root_dir: artifacts/data_ingestion
    source_file: data/sample_dataset.csv
    local_data_file: artifacts/data_ingestion/dataset.csv
""",
    "configs/schema.yaml": "",
    "src/__init__.py": "",
    "src/components/__init__.py": "",
    "src/pipeline/__init__.py": "",
    "src/utils/__init__.py": """import os
import sys
import yaml
import logging
from src.exception import CustomException
from pathlib import Path

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
""",
    "src/entity/__init__.py": """from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_file: Path
    local_data_file: Path
""",
    "src/config/__init__.py": """from src.constants import *
from src.utils import read_yaml, create_directories
from src.entity import DataIngestionConfig
import logging
from src.exception import CustomException
import sys
from pathlib import Path

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
""",
    "src/exception.py": """import sys

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
""",
    "src/logger.py": """import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(os.path.dirname(logs_path), exist_ok=True)

LOG_FILE_PATH = logs_path

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
""",
    "src/constants.py": """from pathlib import Path

CONFIG_FILE_PATH = Path("configs/config.yaml")
SCHEMA_FILE_PATH = Path("configs/schema.yaml")
""",
    "src/components/data_ingestion.py": """import os
import sys
import shutil
from pathlib import Path
import logging
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
                raise Exception(f"Source file {source_file} does not exist.")
                
            logging.info(f"Source file {source_file} exists. Starting data copy.")
            
            os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
            shutil.copy(source_file, self.config.local_data_file)
            
            logging.info(f"Data copied to {self.config.local_data_file}")
            
            return self.config.local_data_file
        except Exception as e:
            raise CustomException(e, sys)
""",
    "app.py": "",
    "main.py": """from src.logger import logging
from src.exception import CustomException
from src.config import ConfigurationManager
from src.components.data_ingestion import DataIngestion
import sys

def main():
    try:
        logging.info(">>>>>> Data Ingestion phase started <<<<<<")
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        
        data_ingestion = DataIngestion(config=data_ingestion_config)
        saved_dataset_path = data_ingestion.initiate_data_ingestion()
        
        logging.info(f"Data ingestion completed. Dataset saved at: {saved_dataset_path}")
        logging.info(">>>>>> Data Ingestion phase completed <<<<<<\\n")
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
""",
    "setup.py": """import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

REPO_NAME = "customer-support-ai"
AUTHOR_USER_NAME = "Ayan"
SRC_REPO = "src"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    description="A small python package for NLP app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
""",
    ".gitignore": """logs/
__pycache__/
artifacts/
.venv/
""",
    "README.md": "# Customer Support AI\n",
}

for rel_path, content in files.items():
    file_path = base_dir / rel_path
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

print("Scaffold complete.")

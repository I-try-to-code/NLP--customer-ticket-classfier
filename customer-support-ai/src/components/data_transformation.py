import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self):
        logging.info("Entered the data transformation component")
        try:
            # Read Dataset
            logging.info(f"Reading data from {self.config.data_path}")
            df = pd.read_csv(self.config.data_path)
            
            # Filter English rows
            logging.info("Filtering only English language rows")
            df = df[df["language"] == "en"].reset_index(drop=True)  
            
            # Fill Missing Subject
            logging.info("Filling missing values in 'subject' and 'body'")
            if 'subject' in df.columns:
                df['subject'] = df['subject'].fillna('')
            else:
                df['subject'] = ''
                
            if 'body' in df.columns:
                df['body'] = df['body'].fillna('')
            else:
                df['body'] = ''
            
            # Combine Subject + Body and override body column
            logging.info("Combining Subject and Body into 'body' column")
            df['body'] = df['subject'] + " " + df['body']
            
            # Drop original subject column to clean up
            if 'subject' in df.columns:
                df.drop(columns=['subject'], inplace=True)
            
            import re
            import contractions
            
            def basic_clean(text: str) -> str:
                if not isinstance(text, str): return text
                text = text.lower()
                text = re.sub(r'<.*?>', ' ', text) # Drop HTML
                text = re.sub(r'http\S+|www\.\S+', ' ', text) # Drop URLs
                text = re.sub(r'\S+@\S+', ' ', text) # Drop Emails
                text = contractions.fix(text)
                text = text.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
                return re.sub(r'\s+', ' ', text).strip()
                
            logging.info("Applying basic text cleaning (HTML/URLs) for Transformers")
            df['body'] = df['body'].apply(basic_clean)
            
            # Save Clean Text
            df.to_csv(self.config.transformed_data_path, index=False)
            logging.info(f"Saved transformed dataset to {self.config.transformed_data_path}")
            
            return self.config.transformed_data_path
            
        except Exception as e:
            raise CustomException(e, sys)

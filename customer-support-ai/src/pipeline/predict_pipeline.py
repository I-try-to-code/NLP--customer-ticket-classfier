import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.utils.nlp_pipeline import NLPPreprocessingPipeline

class PredictPipeline:
    def __init__(self, model_path: str, vectorizer_path: str, label_encoder_path: str) -> None:
        """
        Initialize inference pipeline by loading saved artifacts.
        """
        try:
            logging.info("Loading inference artifacts...")
            self.model = load_object(model_path)
            self.vectorizer = load_object(vectorizer_path)
            self.label_encoder = load_object(label_encoder_path)
            self.nlp_pipeline = NLPPreprocessingPipeline()
            logging.info("Inference artifacts loaded successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, raw_text: str) -> str:
        """
        Takes raw string input, processes it through the entire pipeline, 
        and returns the decoded predicted class.
        """
        try:
            # 1. Apply full NLP preprocessing
            cleaned_text = self.nlp_pipeline.transform(raw_text)
            
            # 2. Vectorize using loaded TF-IDF (must be passed as an iterable)
            vectorized_text = self.vectorizer.transform([cleaned_text])
            
            # 3. Model Inference
            prediction_encoded = self.model.predict(vectorized_text)
            
            # 4. Decode Prediction
            decoded_label = self.label_encoder.inverse_transform(prediction_encoded)[0]
            
            return decoded_label
        except Exception as e:
            raise CustomException(e, sys)
